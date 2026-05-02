"""
eprint_main.py — Entry point per il download immagini via e-print arXiv.

Utilizzo:
    # Run completo (8.000 articoli, 1 worker):
    python eprint_main.py

    # Con 2 worker concorrenti (~2x più veloce):
    python eprint_main.py --workers 2

    # Test rapido su ID specifici:
    python eprint_main.py --test-ids 2308.04689 2401.12345 2210.03065
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from api_client import fetch_all_ids
from config import EPRINT_PROGRESS_FILE
from eprint_fetcher import fetch_eprint_images, IMAGES_DIR
from throttler import Throttler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt))

    file_handler = logging.FileHandler("eprint_download.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gestione del file di progresso
# ---------------------------------------------------------------------------

def _load_progress() -> Dict[str, str]:
    path = Path(EPRINT_PROGRESS_FILE)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Impossibile leggere il file di progresso: %s", exc)
    return {}


def _save_progress(progress: Dict[str, str]) -> None:
    try:
        with open(EPRINT_PROGRESS_FILE, "w", encoding="utf-8") as fh:
            json.dump(progress, fh, indent=2, sort_keys=True)
    except OSError as exc:
        logger.error("Impossibile aggiornare il file di progresso: %s", exc)


def _already_has_images(arxiv_id: str) -> bool:
    """Controlla se la cartella images/{id}/ esiste e contiene almeno un file."""
    img_dir = IMAGES_DIR / arxiv_id
    return img_dir.is_dir() and any(img_dir.iterdir())


# ---------------------------------------------------------------------------
# Logica principale
# ---------------------------------------------------------------------------

def _process_id(arxiv_id: str, throttler: Throttler) -> tuple:
    """Funzione eseguita da ogni worker. Restituisce (arxiv_id, n_images, status)."""
    n, status = fetch_eprint_images(arxiv_id, throttler)
    return arxiv_id, n, status


def run(ids: List[str], n_workers: int) -> None:
    progress = _load_progress()

    # Filtra gli ID già processati o con immagini su disco
    pending = []
    for arxiv_id in ids:
        if arxiv_id in progress:
            logger.debug("[%s] Skip (già in progress: %s).", arxiv_id, progress[arxiv_id])
            continue
        if _already_has_images(arxiv_id):
            logger.debug("[%s] Skip (immagini già presenti su disco).", arxiv_id)
            progress[arxiv_id] = "done"
            continue
        pending.append(arxiv_id)

    total = len(ids)
    already_done = total - len(pending)
    logger.info(
        "Totale ID: %d | Già completati: %d | Da processare: %d | Worker: %d",
        total, already_done, len(pending), n_workers,
    )

    throttler = Throttler()
    counters = {"done": 0, "no_images": 0, "no_tar": 0, "no_source": 0}
    completed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_id, arxiv_id, throttler): arxiv_id
                for arxiv_id in pending
            }

            for future in as_completed(futures):
                arxiv_id, n_images, status = future.result()
                progress[arxiv_id] = status
                counters[status] = counters.get(status, 0) + 1
                completed_count += 1

                logger.info(
                    "[%d/%d] %s → %s (%d immagini)",
                    completed_count, len(pending), arxiv_id, status, n_images,
                )

                if completed_count % 20 == 0:
                    _save_progress(progress)
    finally:
        throttler.close()
        _save_progress(progress)

    total_images_found = sum(
        len(list((IMAGES_DIR / aid).iterdir()))
        for aid, st in progress.items()
        if st == "done" and (IMAGES_DIR / aid).is_dir()
    )

    logger.info(
        "\n--- Riepilogo ---\n"
        "  Done (con immagini):  %d\n"
        "  No images (tar vuoto): %d\n"
        "  No tar (solo PDF):    %d\n"
        "  No source (404):      %d\n"
        "  Totale immagini:      %d",
        counters.get("done", 0) + already_done,
        counters.get("no_images", 0),
        counters.get("no_tar", 0),
        counters.get("no_source", 0),
        total_images_found,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Scarica immagini da sorgenti e-print arXiv."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Numero di worker concorrenti (default: 1, max consigliato: 2)",
    )
    parser.add_argument(
        "--test-ids",
        nargs="+",
        metavar="ID",
        help="Elabora solo questi ID specifici (per test rapidi)",
    )
    parser.add_argument(
        "--ids-file",
        metavar="FILE",
        help="File di testo con un ID arXiv per riga. Salta la query API.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers deve essere >= 1")
    if args.workers > 3:
        logger.warning(
            "Con %d worker il rate effettivo potrebbe scendere sotto 3s/richiesta "
            "e triggherare il throttle di arXiv. Usare con cautela.",
            args.workers,
        )

    if args.test_ids:
        ids = args.test_ids
        logger.info("Modalità test: %d ID specificati.", len(ids))
    elif args.ids_file:
        ids_path = Path(args.ids_file)
        if not ids_path.exists():
            logger.error("File non trovato: %s", args.ids_file)
            sys.exit(1)
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logger.info("Caricati %d ID da %s.", len(ids), args.ids_file)
    else:
        logger.info("Recupero ID da API arXiv...")
        throttler = Throttler()
        try:
            ids = fetch_all_ids(throttler)
        finally:
            throttler.close()
        logger.info("Recuperati %d ID.", len(ids))

    run(ids, n_workers=args.workers)


if __name__ == "__main__":
    main()
