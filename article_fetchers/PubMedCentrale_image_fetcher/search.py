from Bio import Entrez
from config import NUM_ARTICLES


def search_oa_articles(max_results: int = NUM_ARTICLES) -> list[str]:
    """
    Restituisce gli ID PMC degli articoli Open Access più recenti.
    Biopython gestisce internamente il rate limiting di Entrez.
    """
    print(f"Ricerca degli ultimi {max_results} articoli Open Access su PMC...")
    try:
        handle = Entrez.esearch(
            db="pmc",
            term="open access[filter]",
            sort="date",
            retmax=max_results,
        )
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]
        print(f"Trovati {len(ids)} articoli.")
        return ids
    except Exception as e:
        print(f"Errore durante la ricerca Entrez: {e}")
        return []
