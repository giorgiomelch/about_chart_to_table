# arxiv_image_fetcher

Downloads articles from the **cs.DB** category on arXiv and extracts their embedded images via the arXiv e-print API.


---

## Structure

```
arxiv_image_fetcher/
├── eprint_main.py      # Entry point
├── eprint_fetcher.py   # Download and image extraction logic
├── api_client.py       # arXiv API client (fetches article ID list)
├── config.py           # Configuration: URLs, paths, limits
└── throttler.py        # Rate limiter to comply with arXiv policies
```

---

## Configuration

Before running, review the parameters in `config.py`:

| Parameter | Description |
|-----------|-------------|
| `EPRINT_BASE_URL` | Base URL for the arXiv e-print API |
| `IMAGES_DIR` | Output folder for extracted images |
| `EPRINT_PROGRESS_FILE` | JSON file used to save progress |
| Rate limit | The `Throttler` inserts pauses between requests |

---

## Usage

```bash
cd article_fetchers/arxiv_image_fetcher

# Full download (~8,000 articles, 1 worker)
python eprint_main.py

# With 2 concurrent workers (~2x faster)
python eprint_main.py --workers 2

# Quick test on specific IDs
python eprint_main.py --test-ids 2308.04689 2401.12345 2210.03065
```

---

## Resuming interrupted runs

Progress is saved to `EPRINT_PROGRESS_FILE` (see `config.py`). If the process is interrupted, already-downloaded articles are automatically skipped on restart.

A detailed log is written to `eprint_download.log`.

---

## Output

```
articles_images/
└── arXiv/
    └── images/          ← extracted images (PNG, JPG, …)
```

Images are then cropped/normalised and passed to [deformable-detr-infer](../../deformable-detr-infer/README.md).
