# article_fetchers

This folder contains the two modules for automatically downloading scientific articles and extracting their figures.

| Module | Source | 
|--------|--------|
| [arxiv_image_fetcher](arxiv_image_fetcher/README.md) | arXiv (cs.DB) 
| [PubMedCentrale_image_fetcher](PubMedCentrale_image_fetcher/README.md) | PubMed Central 

Images are saved to the [`articles_images/`](../articles_images/README.md) folder at the repository root.

---

## General flow

```
arXiv API / PMC OA API
        ↓
Download articles (HTML / e-print bundles)
        ↓
Extract embedded images
        ↓
articles_images/arXiv/images/
articles_images/PubMedCentral/PMC_raw_images/
```

The extracted images are then pre-processed (cropping, normalisation) and passed to [deformable-detr-infer](../deformable-detr-infer/README.md) for compound figure decomposition.

See the README of each sub-module for configuration and usage details.
