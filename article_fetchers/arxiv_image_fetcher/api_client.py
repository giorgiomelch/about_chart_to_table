"""
api_client.py — Interrogazione dell'API Atom di arXiv con paginazione.

Recupera i metadati (in blocchi di BATCH_SIZE record) fino a TOTAL_DOCUMENTS
e restituisce la lista degli ID documento.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Generator, List

from config import API_BASE_URL, BATCH_SIZE, SEARCH_QUERY, TOTAL_DOCUMENTS
from throttler import Throttler

logger = logging.getLogger(__name__)

# Namespace Atom usato dalle risposte arXiv
_ATOM_NS = "http://www.w3.org/2005/Atom"


def _extract_ids_from_feed(xml_text: str) -> List[str]:
    """
    Analizza la risposta Atom XML e restituisce una lista di ID arXiv.

    Gli <id> nel feed hanno la forma:
        http://arxiv.org/abs/2308.04689v1
    Estraiamo solo la parte finale (es. '2308.04689').

    Args:
        xml_text: Corpo della risposta XML.

    Returns:
        Lista di ID (senza versione).
    """
    root = ET.fromstring(xml_text)
    ids: List[str] = []

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        id_elem = entry.find(f"{{{_ATOM_NS}}}id")
        if id_elem is None or not id_elem.text:
            continue
        raw_id: str = id_elem.text.strip()
        # Estrai solo l'ID numerico, rimuovendo versione e path
        # es. "http://arxiv.org/abs/2308.04689v1" -> "2308.04689"
        arxiv_id = raw_id.rstrip("/").split("/")[-1]
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]
        ids.append(arxiv_id)

    return ids


def _fetch_batch(
    throttler: Throttler,
    start: int,
    max_results: int,
) -> List[str]:
    """
    Recupera un singolo batch di metadati dall'API arXiv.

    Args:
        throttler: Istanza del rate-limiter da usare per la richiesta.
        start: Indice di partenza (0-based).
        max_results: Numero massimo di record da richiedere in questo batch.

    Returns:
        Lista di ID estratti dal batch.

    Raises:
        requests.RequestException: In caso di errore HTTP non recuperabile.
    """
    params = {
        "search_query": SEARCH_QUERY,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    logger.info(
        "Richiesta API — start=%d, max_results=%d (totale richiesto: %d)",
        start,
        max_results,
        TOTAL_DOCUMENTS,
    )

    response = throttler.get(API_BASE_URL, params=params)
    ids = _extract_ids_from_feed(response.text)
    logger.info("Ricevuti %d ID in questo batch.", len(ids))
    return ids


def fetch_all_ids(throttler: Throttler) -> List[str]:
    """
    Recupera *tutti* gli ID arXiv per la query configurata, fino a TOTAL_DOCUMENTS.

    Utilizza la paginazione (parametri start / max_results) e rispetta
    il throttling tramite l'istanza Throttler fornita.

    Args:
        throttler: Istanza del rate-limiter condivisa con il resto del sistema.

    Returns:
        Lista deduplicata di ID arXiv (lunghezza <= TOTAL_DOCUMENTS).
    """
    all_ids: List[str] = []
    start: int = 0

    while len(all_ids) < TOTAL_DOCUMENTS:
        remaining = TOTAL_DOCUMENTS - len(all_ids)
        batch_size = min(BATCH_SIZE, remaining)

        batch_ids = _fetch_batch(throttler, start=start, max_results=batch_size)

        if not batch_ids:
            logger.warning(
                "Batch vuoto ricevuto a start=%d: il feed potrebbe non avere "
                "abbastanza documenti per la query '%s'.",
                start,
                SEARCH_QUERY,
            )
            break

        all_ids.extend(batch_ids)
        start += len(batch_ids)

        logger.info(
            "Totale ID raccolti finora: %d / %d", len(all_ids), TOTAL_DOCUMENTS
        )

    # Rimuove eventuali duplicati mantenendo l'ordine
    seen: set = set()
    unique_ids: List[str] = []
    for arxiv_id in all_ids:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            unique_ids.append(arxiv_id)

    logger.info("Raccolta ID completata. Totale unici: %d", len(unique_ids))
    return unique_ids[:TOTAL_DOCUMENTS]
