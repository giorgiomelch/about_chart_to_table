import os
import time
import requests
from Bio import Entrez
from bs4 import BeautifulSoup
from urllib.parse import urljoin

Entrez.email = "gio.melchiorri@stud.uniroma3.it"

NUMERO_ARTICOLI = 8_000


def crea_cartella(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ottieni_ultimi_articoli(max_results):
    """
    Recupera gli ID degli articoli Open Access ordinati per data (più recenti prima).
    """
    print(f"Ricerca degli ultimi {max_results} articoli Open Access pubblicati...")
    try:
        handle = Entrez.esearch(
            db="pmc", 
            term="open access[filter]", 
            sort="date", 
            retmax=max_results
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"Errore critico durante la ricerca: {e}")
        return []
    

def salva_svg_inline(svg_soup, percorso_file):
    """
    Salva un oggetto BeautifulSoup <svg> in un file .svg su disco.
    Aggiunge l'header XML se mancante per garantire compatibilità.
    """
    try:
        # Ottieni il contenuto stringa del tag
        
        contenuto_svg = str(svg_soup)
        
        # Aggiungi dichiarazione XML se non c'è, per renderlo apribile dai software di grafica
        if not contenuto_svg.startswith("<?xml"):
            contenuto_svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + contenuto_svg

        with open(percorso_file, "w", encoding="utf-8") as f:
            f.write(contenuto_svg)
        return True
    except Exception as e:
        print(f"  Errore salvataggio SVG inline: {e}")
        return False
    
def scarica_immagini_da_articolo(pmc_id, root_folder):
    """
    Scarica le immagini. Ritorna True se ha scaricato almeno un file, False altrimenti.
    """
    folder_name = f"PMC{pmc_id}"
    percorso_articolo = os.path.join(root_folder, folder_name)

    if os.path.exists(percorso_articolo) and os.listdir(percorso_articolo):
        print(f"Saltato: PMC{pmc_id} (già presente).")
        return False

    url_articolo = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.ncbi.nlm.nih.gov/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'}

    try:
        response = requests.get(url_articolo, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Errore HTTP {response.status_code} per PMC{pmc_id}")
            return False

        soup = BeautifulSoup(response.content, "html.parser")
        figure = soup.find_all("figure")

        if not figure:
            return False

        crea_cartella(percorso_articolo)
        print(f"Processando PMC{pmc_id}: trovate {len(figure)} immagini.")

        count_img = 0
        
        for i, fig in enumerate(figure):
            prefix = f"PMC{pmc_id}_fig_{i+1}"

            # --- 1. Cerca immagini standard (JPG, PNG, GIF, SVG linkati) ---
            img_tags = fig.find_all("img")
            for j, img_tag in enumerate(img_tags):
                if "src" in img_tag.attrs:
                    img_url = img_tag["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin("https://www.ncbi.nlm.nih.gov", img_url)

                    # Estensione e pulizia query string
                    estensione = os.path.splitext(img_url)[1].split('?')[0] or ".jpg"
                    
                    # Nome file: se c'è più di un'immagine nella stessa figura, usiamo un sotto-indice
                    suffix = f"_{j+1}" if len(img_tags) > 1 else ""
                    nome_file = f"{prefix}{suffix}{estensione}"
                    path_completo = os.path.join(percorso_articolo, nome_file)

                    try:
                        r_img = requests.get(img_url, headers=headers, timeout=10)
                        time.sleep(0.2)
                        with open(path_completo, 'wb') as f:
                            f.write(r_img.content)
                        count_img += 1
                    except Exception:
                        pass

            # --- 2. Cerca SVG Inline (Codice <svg> nel div) ---
            svg_tags = fig.find_all("svg")
            for k, svg_tag in enumerate(svg_tags):
                # Assegna un nome distinto per gli SVG inline
                suffix_svg = f"_inline_{k+1}"
                nome_file_svg = f"{prefix}{suffix_svg}.svg"
                path_completo_svg = os.path.join(percorso_articolo, nome_file_svg)
                
                if salva_svg_inline(svg_tag, path_completo_svg):
                    count_img += 1
                    print(f"  - Salvato SVG Inline: {nome_file_svg}")

        if count_img == 0:
            os.rmdir(percorso_articolo) # Pulisce se vuoto
            return False
            
        return True

    except Exception as e:
        print(f"Errore su PMC{pmc_id}: {e}")
        return False


def main():
    root_output = "Ultimi_Articoli_PMC"
    crea_cartella(root_output)

    ids = ottieni_ultimi_articoli(NUMERO_ARTICOLI)
    print(f"Trovati {len(ids)} ID. Inizio download immagini...")

    articoli_scaricati = 0

    for pmc_id in ids:
        successo = scarica_immagini_da_articolo(pmc_id, root_output)
        if successo:
            articoli_scaricati += 1
        
        time.sleep(2)

    print(f"\n--- Operazione Completata ---")
    print(f"Analizzati: {len(ids)} articoli.")
    print(f"Articoli con immagini scaricate: {articoli_scaricati}")

if __name__ == "__main__":
    main()
