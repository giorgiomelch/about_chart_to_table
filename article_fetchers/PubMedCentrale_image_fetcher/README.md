# PubMedCentrale_image_fetcher

Downloads articles from **PubMed Central (PMC) Open Access** and extracts their images via the PMC OA API.

---

## Structure

```
PubMedCentrale_image_fetcher/
├── main.py         # Entry point
├── search.py       # PMC OA article search via API
├── oa_client.py    # PMC API client: lists images for each article
├── downloader.py   # Downloads images from PMC S3 servers
└── config.py       # Configuration: URLs, paths, sleep intervals
```

---

## Configuration

Review the parameters in `config.py` before running:

| Parameter | Description |
|-----------|-------------|
| `NUM_ARTICLES` | Number of articles to download |
| `OUTPUT_DIR` | Destination folder for images |
| `OA_API_SLEEP` | Pause (seconds) between PMC API calls |
| `DOWNLOAD_SLEEP` | Pause (seconds) between article downloads |

---

## Usage

```bash
cd article_fetchers/PubMedCentrale_image_fetcher

python main.py
```

A summary is printed on completion:

```
--- Summary ---
  Articles processed : 8000
  Downloaded         : 7412
  Already present    : 341
  No images in S3    : 198
  Errors             : 49
```

---

## Resuming interrupted runs

Already-downloaded articles are detected automatically: if the output folder already contains images for a given PMC ID, the download is skipped (counted as "Already present").

---

## Output

```
articles_images/
└── PubMedCentral/
    └── PMC_raw_images/   ← original images downloaded from PMC
```

Images are then cropped and passed to [deformable-detr-infer](../../deformable-detr-infer/README.md).
