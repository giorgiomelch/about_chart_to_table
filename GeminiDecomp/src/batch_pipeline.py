"""
Pipeline completa per generare predizioni Gemini su un dataset di immagini.

Flusso:
  1. create_batch_requests()  → carica immagini su Google Files API + crea JSONL richieste
  2. run_batch_inference()    → invia il job a Gemini Batch API e scarica risultati
  3. predictions_to_labelstudio() → converte JSONL predizioni in formato LabelStudio

Per le funzioni di basso livello (API Gemini) si delega a src/gemini_client.py.
"""
import json
from pathlib import Path
from typing import Optional

from src.gemini_client import (
    DEFAULT_MODEL,
    create_batch_file,
    run_batch_job,
    download_batch_jsonl,
)
from src.coco_converter import parse_gemini_response_line


# ── Step 1 — Creazione richieste batch ────────────────────────────────────

def create_batch_requests(
    images_dir: str | Path,
    output_jsonl: str | Path,
    model: str = DEFAULT_MODEL,
) -> bool:
    """
    Scansiona una directory di immagini, carica ogni file su Google Files API
    e produce un JSONL con le richieste pronte per il batch job.

    Args:
        images_dir:   directory con le immagini da analizzare
        output_jsonl: percorso output del file JSONL
        model:        modello Gemini da usare

    Returns:
        True se almeno una richiesta è stata generata.
    """
    return create_batch_file(
        dataset_path=str(images_dir),
        jsonl_output_path=str(output_jsonl),
        gemini_model=model,
    )


# ── Step 2 — Esecuzione e download risultati ──────────────────────────────

def run_batch_inference(
    requests_jsonl: str | Path,
    output_jsonl: str | Path,
    model: str = DEFAULT_MODEL,
) -> None:
    """
    Invia il file JSONL di richieste a Gemini Batch API, attende il completamento
    con polling automatico e scarica il JSONL dei risultati.

    Args:
        requests_jsonl: JSONL creato da create_batch_requests()
        output_jsonl:   percorso dove salvare i risultati
        model:          modello Gemini da usare
    """
    run_batch_job(
        jsonl_input_path=str(requests_jsonl),
        final_output_path=str(output_jsonl),
        gemini_model=model,
    )


def download_results(job_name: str, output_jsonl: str | Path) -> None:
    """
    Scarica i risultati di un batch job già completato dato il suo job_name.
    Utile se run_batch_inference() è stato interrotto dopo il completamento
    ma prima del download.
    """
    download_batch_jsonl(
        job_name=job_name,
        local_filename=str(output_jsonl),
    )


# ── Step 3 — Conversione in formato LabelStudio ───────────────────────────

def predictions_to_labelstudio(
    jsonl_path: str | Path,
    output_json: str | Path,
    images_base_path: str = "/data/local-files/?d=",
) -> None:
    """
    Converte un JSONL di predizioni Gemini nel formato JSON di LabelStudio
    (pronto per l'import come "predictions").

    Le coordinate vengono convertite da scala 0–1000 a percentuale (0–100).

    Args:
        jsonl_path:       JSONL con risposte Gemini
        output_json:      percorso file JSON output per LabelStudio
        images_base_path: prefisso URL per le immagini in LabelStudio
    """
    jsonl_path  = Path(jsonl_path)
    output_json = Path(output_json)

    tasks = []
    errors = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stem, preds = parse_gemini_response_line(line)
            if stem is None:
                errors += 1
                continue

            result_items = []
            for p in preds:
                bbox = p.get("bbox")
                cls  = p.get("class", "")
                if not bbox or len(bbox) != 4:
                    continue

                ymin, xmin, ymax, xmax = map(float, bbox)
                result_items.append({
                    "value": {
                        "x":              xmin / 10,
                        "y":              ymin / 10,
                        "width":          (xmax - xmin) / 10,
                        "height":         (ymax - ymin) / 10,
                        "rectanglelabels": [cls],
                    },
                    "from_name": "label",
                    "to_name":   "image",
                    "type":      "rectanglelabels",
                })

            tasks.append({
                "data": {"image": f"{images_base_path}{stem}.jpg"},
                "predictions": [{"result": result_items}],
            })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Salvato {len(tasks)} task in {output_json}")
    if errors:
        print(f"  Righe con errore di parsing: {errors}")
