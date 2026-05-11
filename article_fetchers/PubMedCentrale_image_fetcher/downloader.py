import os
import requests


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PMC-HiRes-Fetcher/1.0 (mailto:youremail@stud.uniroma3.it)"})


def download_images(pmc_id: str, images: list[tuple[str, str]], output_root: str) -> int:
    """
    Scarica le immagini indicate (lista di (nome_file, url)) nella cartella
    `{output_root}/PMC{pmc_id}/`.
    Ritorna il numero di immagini salvate, oppure 0 se la cartella esiste già.
    """
    article_dir = os.path.join(output_root, f"PMC{pmc_id}")

    if os.path.isdir(article_dir) and os.listdir(article_dir):
        print(f"  Saltato: PMC{pmc_id} (già presente su disco).")
        return 0

    os.makedirs(article_dir, exist_ok=True)
    count = 0

    for filename, url in images:
        dest_path = os.path.join(article_dir, filename)
        try:
            response = _SESSION.get(url, timeout=30)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            count += 1
        except requests.RequestException as e:
            print(f"  [Download] Errore scaricando {filename}: {e}")

    if count == 0:
        os.rmdir(article_dir)

    return count
