"""
eprint_fetcher.py — Download delle immagini originali dai sorgenti LaTeX arXiv.

Utilizza l'endpoint e-print di arXiv che restituisce un archivio tar gzip
contenente i sorgenti LaTeX dell'articolo, incluse le figure originali.

Formati supportati:
  • PNG, JPG, GIF, SVG, TIFF, BMP — salvati direttamente
  • PDF, EPS — convertiti in PNG tramite pdftoppm (poppler-utils)
    Skip automatico di PDF con più di PDF_EPS_MAX_PAGES pagine (allegati).
"""

import gzip
import io
import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Tuple

import requests

from config import EPRINT_BASE_URL, IMAGE_EXTENSIONS_IN_TAR, PDF_EPS_DPI, PDF_EPS_MAX_PAGES
from throttler import Throttler

logger = logging.getLogger(__name__)

IMAGES_DIR = Path("images")

# Estensioni che richiedono conversione tramite pdftoppm
_CONVERT_EXTENSIONS = {".pdf", ".eps"}
# Estensioni salvabili direttamente
_DIRECT_EXTENSIONS = IMAGE_EXTENSIONS_IN_TAR - _CONVERT_EXTENSIONS


def _pdf_page_count(pdf_path: Path) -> int:
    """Restituisce il numero di pagine di un PDF. Ritorna 0 in caso di errore."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


def _convert_to_png(src_path: Path, out_dir: Path, base_name: str) -> list:
    """
    Converte un PDF o EPS in uno o più PNG tramite pdftoppm.

    Args:
        src_path: Path al file PDF/EPS sorgente.
        out_dir: Cartella dove scrivere i PNG.
        base_name: Prefisso per i file di output (es. '2308.04689_03').

    Returns:
        Lista di Path ai PNG generati.
    """
    out_prefix = str(out_dir / base_name)
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(PDF_EPS_DPI), str(src_path), out_prefix],
            capture_output=True, timeout=30, check=True
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timeout conversione: %s", src_path.name)
        return []
    except subprocess.CalledProcessError as exc:
        logger.warning("pdftoppm fallito per %s: %s", src_path.name, exc.stderr.decode(errors="replace").strip())
        return []

    # pdftoppm produce: base_name-1.png, base_name-2.png, ...
    # oppure base_name.png se pagina singola
    results = sorted(out_dir.glob(f"{base_name}*.png"))
    return results


def fetch_eprint_images(arxiv_id: str, throttler: Throttler) -> Tuple[int, str]:
    """
    Scarica e decomprimi il tar sorgente di un articolo, estraendo le immagini.

    Args:
        arxiv_id: Identificatore arXiv (es. '2308.04689').
        throttler: Istanza del rate-limiter (thread-safe).

    Returns:
        Tupla (n_immagini_salvate, status) dove status è:
          "done"       — almeno un'immagine estratta
          "no_images"  — tar valido ma senza immagini nei formati supportati
          "no_tar"     — risposta non è un tar valido (es. PDF diretto)
          "no_source"  — 404, sorgente non disponibile
    """
    url = f"{EPRINT_BASE_URL}/{arxiv_id}"
    logger.info("E-print download: %s", url)

    # --- Richiesta HTTP throttolata ---
    try:
        response = throttler.get(url)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 0
        if status_code == 404:
            logger.info("[%s] Sorgente non disponibile (404).", arxiv_id)
            return 0, "no_source"
        logger.error("[%s] Errore HTTP %d — saltato.", arxiv_id, status_code)
        return 0, "no_source"
    except requests.RequestException as exc:
        logger.error("[%s] Errore di rete: %s — saltato.", arxiv_id, exc)
        return 0, "no_source"

    # --- Decompressione e apertura del tar ---
    try:
        raw_bytes = io.BytesIO(response.content)
        tar = tarfile.open(fileobj=raw_bytes, mode="r:*")
    except Exception:
        try:
            raw_bytes.seek(0)
            decompressed = io.BytesIO(gzip.decompress(response.content))
            tar = tarfile.open(fileobj=decompressed, mode="r:*")
        except Exception as exc:
            logger.info("[%s] Non è un tar valido (%s) — saltato.", arxiv_id, type(exc).__name__)
            return 0, "no_tar"

    output_dir = IMAGES_DIR / arxiv_id
    output_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        try:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                member_path = Path(member.name)

                # Sicurezza: salta path con traversal
                if ".." in member_path.parts:
                    logger.warning("[%s] Path traversal: %s — saltato.", arxiv_id, member.name)
                    continue

                ext = member_path.suffix.lower()
                if ext not in IMAGE_EXTENSIONS_IN_TAR:
                    continue

                file_obj = tar.extractfile(member)
                if file_obj is None:
                    continue
                raw_data = file_obj.read()

                # --- Formati diretti: salva subito ---
                if ext in _DIRECT_EXTENSIONS:
                    n_saved += 1
                    out_path = output_dir / f"{arxiv_id}_{n_saved:02d}{ext}"
                    try:
                        out_path.write_bytes(raw_data)
                        logger.debug("[%s] Salvato: %s", arxiv_id, out_path.name)
                    except OSError as exc:
                        logger.warning("[%s] Impossibile scrivere %s: %s", arxiv_id, out_path.name, exc)
                        n_saved -= 1

                # --- PDF / EPS: converti tramite pdftoppm ---
                elif ext in _CONVERT_EXTENSIONS:
                    tmp_src = tmp_dir / f"{arxiv_id}_src{ext}"
                    tmp_src.write_bytes(raw_data)

                    # Skip PDF con troppo pagine (allegati, non figure)
                    if ext == ".pdf":
                        pages = _pdf_page_count(tmp_src)
                        if pages > PDF_EPS_MAX_PAGES:
                            logger.debug(
                                "[%s] PDF con %d pagine skippato (max %d): %s",
                                arxiv_id, pages, PDF_EPS_MAX_PAGES, member.name
                            )
                            continue

                    base_name = f"{arxiv_id}_{n_saved + 1:02d}_conv"
                    png_files = _convert_to_png(tmp_src, tmp_dir, base_name)

                    for png_src in png_files:
                        n_saved += 1
                        final_path = output_dir / f"{arxiv_id}_{n_saved:02d}.png"
                        shutil.move(str(png_src), str(final_path))
                        logger.debug("[%s] Convertito: %s", arxiv_id, final_path.name)

        finally:
            tar.close()

    if n_saved == 0:
        try:
            output_dir.rmdir()
        except OSError:
            pass
        logger.info("[%s] Nessuna immagine trovata.", arxiv_id)
        return 0, "no_images"

    logger.info("[%s] Estratte %d immagini.", arxiv_id, n_saved)
    return n_saved, "done"
