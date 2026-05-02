"""
Conversione annotazioni Gemini (JSONL batch) → dataset COCO + train/val split.

Formato input JSONL (una riga per immagine):
  {
    "key": "req_<filename_stem>",
    "response": {
      "candidates": [{
        "content": {"parts": [{"text": "[{\"bbox\":[ymin,xmin,ymax,xmax],\"class\":\"...\"},...]"}]},
        "finishReason": "STOP"
      }]
    }
  }

Formato bbox Gemini: [ymin, xmin, ymax, xmax] normalizzati su scala 0–1000.
Formato bbox COCO:   [x, y, width, height] in pixel assoluti.
"""
import json
import random
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


# ── Parsing ────────────────────────────────────────────────────────────────

def parse_gemini_response_line(line: str) -> Tuple[Optional[str], List[dict]]:
    """
    Parsifica una riga JSONL di risposta Gemini.

    Returns:
        (filename_stem, predictions) dove predictions è una lista di
        {"bbox": [ymin,xmin,ymax,xmax], "class": str}.
        Restituisce (None, []) in caso di errore o risposta vuota.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None, []

    key = data.get("key", "")
    # La chiave ha formato "req_<filename_stem>" — rimuove il prefisso
    stem = key[4:] if key.startswith("req_") else key

    candidates = data.get("response", {}).get("candidates", [])
    if not candidates:
        return stem, []

    candidate = candidates[0]
    if candidate.get("finishReason") not in ("STOP", None):
        return stem, []

    parts = candidate.get("content", {}).get("parts", [])
    if not parts:
        return stem, []

    text = parts[0].get("text", "")
    try:
        preds = json.loads(text)
        if not isinstance(preds, list):
            return stem, []
        return stem, preds
    except (json.JSONDecodeError, TypeError):
        return stem, []


def gemini_bbox_to_coco(
    bbox_norm: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converte bbox Gemini [ymin, xmin, ymax, xmax] (0–1000) in
    formato COCO [x, y, width, height] (pixel assoluti).
    """
    ymin, xmin, ymax, xmax = map(float, bbox_norm)
    x  = xmin / 1000 * img_w
    y  = ymin / 1000 * img_h
    bw = (xmax - xmin) / 1000 * img_w
    bh = (ymax - ymin) / 1000 * img_h
    return x, y, bw, bh


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    """
    Indicizza ricorsivamente tutte le immagini in images_dir (compresi sottocartelle).

    Returns:
        {stem: path} — mappa dal nome senza estensione al percorso assoluto.
    """
    index: Dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in images_dir.rglob(ext):
            index[p.stem] = p
    return index


def _get_image_size(
    images_dir: Path,
    stem: str,
    image_index: Optional[Dict[str, Path]] = None,
) -> Optional[Tuple[int, int]]:
    """Prova a leggere (width, height) dell'immagine. Restituisce None se mancante."""
    if image_index is not None:
        path = image_index.get(stem)
        if path is None:
            return None
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return None

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = images_dir / (stem + ext)
        if path.exists():
            try:
                with Image.open(path) as img:
                    return img.size
            except Exception:
                return None
    return None


# ── Conversione ────────────────────────────────────────────────────────────

def jsonl_to_coco(
    jsonl_path: str | Path,
    images_dir: str | Path,
    filter_class: Optional[str] = None,
    skip_empty: bool = False,
    image_index: Optional[Dict[str, Path]] = None,
) -> dict:
    """
    Converte un singolo file JSONL di predizioni Gemini in un dataset COCO.

    Args:
        jsonl_path:   percorso al file JSONL
        images_dir:   directory con le immagini originali (per leggere dimensioni)
        filter_class: se impostato ("chart" o "non_chart"), include solo quella classe
        skip_empty:   se True, esclude immagini senza annotazioni
        image_index:  indice pre-costruito {stem: path} da build_image_index(); se
                      fornito evita la ricerca flat su disco ad ogni riga

    Returns:
        dict COCO con chiavi "info", "images", "annotations", "categories"
    """
    images_dir = Path(images_dir)
    coco = _empty_coco()
    image_id = ann_id = 1
    skipped = missing = 0

    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Parsing {Path(jsonl_path).name}"):
        line = line.strip()
        if not line:
            continue

        stem, preds = parse_gemini_response_line(line)
        if stem is None:
            skipped += 1
            continue

        size = _get_image_size(images_dir, stem, image_index)
        if size is None:
            missing += 1
            continue
        img_w, img_h = size

        annotations = []
        for p in preds:
            cls = p.get("class", "")
            if filter_class and cls != filter_class:
                continue
            bbox_raw = p.get("bbox")
            if not bbox_raw or len(bbox_raw) != 4:
                continue
            x, y, bw, bh = gemini_bbox_to_coco(bbox_raw, img_w, img_h)
            annotations.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 0,
                "bbox":        [x, y, bw, bh],
                "area":        bw * bh,
                "iscrowd":     0,
            })
            ann_id += 1

        if skip_empty and not annotations:
            continue

        coco["images"].append({
            "id":        image_id,
            "file_name": stem + ".jpg",
            "width":     img_w,
            "height":    img_h,
        })
        coco["annotations"].extend(annotations)
        image_id += 1

    if skipped:
        print(f"  Righe saltate (parse error): {skipped}")
    if missing:
        print(f"  Immagini non trovate su disco: {missing}")

    return coco


def jsonl_dir_to_coco(
    jsonl_dir: str | Path,
    images_dir: str | Path,
    filter_class: Optional[str] = None,
    skip_empty: bool = False,
) -> dict:
    """
    Converte una directory di file JSONL in un unico dataset COCO.
    Le immagini duplicate (stessa filename_stem in più JSONL) vengono deduplicate.

    Indicizza ricorsivamente images_dir una sola volta per gestire strutture a
    sottocartelle (es. PMCimgs_01 … PMCimgs_42).
    """
    jsonl_dir  = Path(jsonl_dir)
    images_dir = Path(images_dir)
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Nessun file .jsonl trovato in {jsonl_dir}")

    print(f"Indicizzazione immagini in {images_dir} ...")
    image_index = build_image_index(images_dir)
    print(f"  Trovate {len(image_index):,} immagini.")

    merged = _empty_coco()
    seen_stems: set[str] = set()
    image_id = ann_id = 1

    for jf in jsonl_files:
        print(f"Elaboro {jf.name}...")
        partial = jsonl_to_coco(jf, images_dir, filter_class, skip_empty, image_index)

        for img in partial["images"]:
            stem = Path(img["file_name"]).stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)

            new_img_id = image_id
            old_img_id = img["id"]
            img = {**img, "id": new_img_id}
            merged["images"].append(img)

            for ann in partial["annotations"]:
                if ann["image_id"] == old_img_id:
                    merged["annotations"].append({
                        **ann,
                        "id":       ann_id,
                        "image_id": new_img_id,
                    })
                    ann_id += 1

            image_id += 1

    print(f"\nTotale: {len(merged['images'])} immagini, {len(merged['annotations'])} annotazioni")
    return merged


# ── Train/Val split ────────────────────────────────────────────────────────

def split_coco(
    coco: dict,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Divide un dataset COCO in train e val.

    Returns:
        (train_coco, val_coco)
    """
    images = list(coco["images"])
    rng = random.Random(seed)
    rng.shuffle(images)

    n_val = max(1, int(len(images) * val_ratio))
    val_images   = images[:n_val]
    train_images = images[n_val:]

    val_ids   = {img["id"] for img in val_images}
    train_ids = {img["id"] for img in train_images}

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns   = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    train = {**coco, "images": train_images, "annotations": train_anns}
    val   = {**coco, "images": val_images,   "annotations": val_anns}

    print(f"Train: {len(train_images)} immagini, {len(train_anns)} annotazioni")
    print(f"Val:   {len(val_images)} immagini, {len(val_anns)} annotazioni")

    return train, val


# ── I/O ───────────────────────────────────────────────────────────────────

def save_coco(coco: dict, output_path: str | Path) -> None:
    """Salva un dataset COCO su file JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Salvato: {output_path}  ({size_mb:.1f} MB)")


# ── Helper ────────────────────────────────────────────────────────────────

def _empty_coco() -> dict:
    return {
        "info": {
            "description":  "PMC charts dataset",
            "version":      "1.0",
            "year":         date.today().year,
            "contributor":  "gemini_3flashpreview-generated",
            "date_created": str(date.today()),
        },
        "licenses":    [],
        "images":      [],
        "annotations": [],
        "categories":  [{"id": 0, "name": "subimage"}],
    }
