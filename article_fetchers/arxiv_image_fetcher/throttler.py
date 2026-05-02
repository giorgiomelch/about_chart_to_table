"""
throttler.py — Controllo del flusso delle richieste HTTP.

Garantisce:
  • un ritardo deterministico di almeno REQUEST_DELAY_SECONDS tra ogni chiamata;
  • gestione dell'header Retry-After in caso di risposta 503 e 429.
"""

import time
import logging
import threading
from typing import Optional

import requests

from config import REQUEST_DELAY_SECONDS, RETRY_AFTER_DEFAULT, USER_AGENT

logger = logging.getLogger(__name__)


class Throttler:
    """Wrapper su requests.Session che impone il rate-limit richiesto da arXiv."""

    def __init__(self) -> None:
        self._session: requests.Session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_time: float = 0.0
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Esegue una GET rispettando il ritardo minimo.
        Ritenta automaticamente in caso di 429, 503 e timeout di rete.

        Args:
            url: URL da richiedere.
            **kwargs: Argomenti aggiuntivi passati a requests.Session.get().

        Returns:
            L'oggetto Response con status code 2xx o 3xx.

        Raises:
            requests.HTTPError: Per errori HTTP diversi da 429 e 503.
            requests.ConnectionError: Per errori di connessione non-timeout.
        """
        retry_wait = RETRY_AFTER_DEFAULT

        while True:
            self._wait_if_needed()

            try:
                response = self._session.get(url, timeout=30, **kwargs)
            except requests.Timeout as exc:
                # Timeout di lettura/connessione: aspetta e riprova con backoff
                logger.warning(
                    "Timeout per %s — attesa di %d s prima di riprovare. (%s)",
                    url, retry_wait, exc.__class__.__name__,
                )
                time.sleep(retry_wait)
                retry_wait = min(retry_wait * 2, 120)
                continue
            except requests.ConnectionError as exc:
                logger.error("Errore di connessione per %s: %s", url, exc)
                raise

            # 429 Too Many Requests e 503 Service Unavailable: backoff esponenziale
            if response.status_code in (429, 503):
                wait_secs = max(self._parse_retry_after(response), retry_wait)
                logger.warning(
                    "%d %s per %s — attesa di %d s prima di riprovare.",
                    response.status_code,
                    response.reason,
                    url,
                    wait_secs,
                )
                time.sleep(wait_secs)
                retry_wait = min(wait_secs * 2, 120)
                continue  # riprova

            retry_wait = RETRY_AFTER_DEFAULT  # reset su risposta ok
            response.raise_for_status()
            return response

    def close(self) -> None:
        """Chiude la sessione HTTP sottostante."""
        self._session.close()

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _wait_if_needed(self) -> None:
        """
        Dorme il tempo necessario per rispettare REQUEST_DELAY_SECONDS.

        Thread-safe: il lock viene acquisito per leggere/aggiornare
        _last_request_time, ma rilasciato durante il sleep per non bloccare
        gli altri worker mentre si aspetta.
        """
        while True:
            with self._lock:
                elapsed = time.monotonic() - self._last_request_time
                remaining = REQUEST_DELAY_SECONDS - elapsed
                if remaining <= 0:
                    # Slot disponibile: prenotiamo subito aggiornando il timestamp
                    # così gli altri worker vedranno il nuovo valore.
                    self._last_request_time = time.monotonic()
                    return
            # Slot occupato: dormiamo fuori dal lock per non bloccare gli altri.
            logger.debug("Throttle: attesa di %.2f s prima della prossima richiesta.", remaining)
            time.sleep(remaining)

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> int:
        """
        Legge l'header Retry-After dalla risposta 429/503.

        Returns:
            Numero di secondi da attendere (intero >= 1).
        """
        raw: Optional[str] = response.headers.get("Retry-After")
        if raw is not None:
            try:
                seconds = int(raw)
                if seconds > 0:
                    return seconds
            except ValueError:
                pass  # potrebbe essere una data HTTP — usiamo il default
        return RETRY_AFTER_DEFAULT
