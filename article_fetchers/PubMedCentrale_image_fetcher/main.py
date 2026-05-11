import os
import time

import config
from search import search_oa_articles
from oa_client import list_article_images
from downloader import download_images


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    ids = search_oa_articles(config.NUM_ARTICLES)
    if not ids:
        print("Nessun articolo trovato. Interruzione.")
        return

    print(f"\nInizio download per {len(ids)} articoli...\n")

    stats = {"scaricati": 0, "saltati": 0, "senza_immagini": 0, "errori": 0}

    for i, pmc_id in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] PMC{pmc_id}")

        images = list_article_images(pmc_id)
        time.sleep(config.OA_API_SLEEP)

        if not images:
            stats["senza_immagini"] += 1
            continue

        result = download_images(pmc_id, images, config.OUTPUT_DIR)
        time.sleep(config.DOWNLOAD_SLEEP)

        if result > 0:
            print(f"  Salvate {result} immagini.")
            stats["scaricati"] += 1
        elif result == 0:
            stats["saltati"] += 1
        else:
            stats["errori"] += 1

    print("\n--- Riepilogo ---")
    print(f"  Articoli analizzati   : {len(ids)}")
    print(f"  Scaricati             : {stats['scaricati']}")
    print(f"  Già presenti          : {stats['saltati']}")
    print(f"  Senza immagini in S3  : {stats['senza_immagini']}")
    print(f"  Errori                : {stats['errori']}")


if __name__ == "__main__":
    main()
