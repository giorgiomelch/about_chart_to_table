from Bio import Entrez

ENTREZ_EMAIL = "your_email@"
ENTREZ_API_KEY = "" 

NUM_ARTICLES = 8_000
OUTPUT_DIR = "Ultimi_Articoli_PMC"

# Intervalli tra le richieste (secondi) per rispettare i limiti di PMC
OA_API_SLEEP = 1.0      # tra una chiamata OA API e la successiva
DOWNLOAD_SLEEP = 3.0    # tra un download di pacchetto e il successivo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".gif", ".svg", ".eps"}

Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY
