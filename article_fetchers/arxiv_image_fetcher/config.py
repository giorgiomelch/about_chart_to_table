"""
config.py — Costanti e configurazione globale del progetto.
"""

# ---------------------------------------------------------------------------
# Ricerca
# ---------------------------------------------------------------------------
SEARCH_QUERY: str = "cat:cs.DB"
TOTAL_DOCUMENTS: int = 8000
BATCH_SIZE: int = 1000          # record per richiesta API (max consentito da arXiv)

# ---------------------------------------------------------------------------
# Endpoint arXiv  (sempre export.arxiv.org per non caricare il sito pubblico)
# ---------------------------------------------------------------------------
API_BASE_URL: str = "https://export.arxiv.org/api/query"
HTML_BASE_URL: str = "https://export.arxiv.org/html"

# ---------------------------------------------------------------------------
# HTTP – User-Agent con email di contatto come richiesto da arXiv
# ---------------------------------------------------------------------------
USER_AGENT: str = (
    "arxiv-cs-db-fetcher/1.0 "
    "(educational research tool; contact: melchiorrigiorgio@gmail.com)"
)

# ---------------------------------------------------------------------------
# Throttling  (arXiv raccomanda >= 3 s tra richieste consecutive)
# ---------------------------------------------------------------------------
REQUEST_DELAY_SECONDS: float = 5.0      # attesa minima tra ogni richiesta
                                        # alzata a 5s per ridurre i 429 sulle immagini
RETRY_AFTER_DEFAULT: int = 20           # secondi da attendere se Retry-After mancante

# ---------------------------------------------------------------------------
# Percorsi locali
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "html_documents"
LOG_FILE: str = "download.log"
PROGRESS_FILE: str = "progress.json"    # tiene traccia degli ID già scaricati

# ---------------------------------------------------------------------------
# E-print source  (approccio alternativo: sorgenti LaTeX con figure originali)
# ---------------------------------------------------------------------------
EPRINT_BASE_URL: str = "https://arxiv.org/e-print"
IMAGE_EXTENSIONS_IN_TAR: set = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".tiff", ".bmp", ".pdf", ".eps"}
PDF_EPS_MAX_PAGES: int = 3        # skip PDF con più di N pagine (evita allegati multi-pagina)
PDF_EPS_DPI: int = 150            # risoluzione conversione PDF/EPS → PNG
EPRINT_PROGRESS_FILE: str = "eprint_progress.json"   # dict {id: "done"|"no_source"|"no_tar"|"no_images"}
