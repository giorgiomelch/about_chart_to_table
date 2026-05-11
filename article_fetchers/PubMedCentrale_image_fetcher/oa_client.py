import xml.etree.ElementTree as ET
import requests

from config import IMAGE_EXTENSIONS


_S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PMC-HiRes-Fetcher/1.0 (mailto:youremail@stud.uniroma3.it)"})


def list_article_images(pmc_id: str) -> list[tuple[str, str]]:
    """
    Interroga il bucket S3 PMC Open Access e restituisce la lista di
    (nome_file, url_https) per ogni immagine presente nell'articolo.
    Ritorna una lista vuota se l'articolo non è disponibile o non ha immagini.
    """
    params = {"prefix": f"PMC{pmc_id}.", "max-keys": "500"}
    try:
        response = _SESSION.get(_S3_BASE, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [S3] Errore di rete per PMC{pmc_id}: {e}")
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        print(f"  [S3] Risposta XML non valida per PMC{pmc_id}: {e}")
        return []

    images = []
    for item in root.findall("s3:Contents", _S3_NS):
        key = item.findtext("s3:Key", default="", namespaces=_S3_NS)
        ext = _ext(key)
        if ext in IMAGE_EXTENSIONS:
            filename = key.split("/")[-1]
            url = f"{_S3_BASE}/{key}"
            images.append((filename, url))

    return images


def _ext(filename: str) -> str:
    import os
    return os.path.splitext(filename)[1].lower()
