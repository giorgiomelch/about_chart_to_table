# GeminiDecomp

Valutazione di **Gemini Flash 3** sul task di *compound figure separation*: dato un'immagine scientifica composta da più figure (grafici, tabelle, schemi), il modello deve localizzare e classificare ogni subfigura con un bounding box.

Le classi rilevate sono due:
- **`chart`** — rappresentazioni di dati numerici (grafici a barre, scatter plot, istogrammi, curve, heatmap, ecc.)
- **`non_chart`** — tutto il resto (tabelle, fotografie, schemi, flowchart, illustrazioni)

---

## Cosa fa questo progetto

### 1. Valutazione delle performance (notebook principale)

`notebooks/evaluate_performance.ipynb` calcola Precision, Recall e F1 su un test set di **998 immagini** (piemmeci_test) al variare della soglia IoU da 0.1 a 0.95.

Risultati a IoU = 0.5:

| Classe      | Precision | Recall | F1     |
|-------------|-----------|--------|--------|
| chart       | 0.9629    | 0.9535 | 0.9582 |
| non_chart   | 0.9067    | 0.8829 | 0.8947 |
| **all**     | **0.9458**| **0.9372** | **0.9415** |

mF1 medio su tutte le soglie IoU [0.1 → 0.95]: **0.8306**

Produce quattro grafici salvati in `results/`:
- `metrics_vs_iou.png` — curve P/R/F1 al variare della soglia
- `heatmap_f1.png` — heatmap F1 per classe × soglia IoU
- `iou_histogram.png` — distribuzione degli IoU score sui match corretti (TP)
- `pr_curve.png` — curva Precision-Recall parametrizzata per soglia IoU

### 2. Visualizzazione errori

`notebooks/visualize_errors.ipynb` mostra le immagini in cui il modello ha sbagliato con bounding box colorati:

| Colore   | Significato                        |
|----------|------------------------------------|
| Verde    | GT box rilevata correttamente (TP) |
| Arancio  | GT box non rilevata (FN — missed)  |
| Blu      | Predizione corretta (TP)           |
| Rosso    | Predizione errata (FP — falso allarme) |

### 3. Dataset COCO (~32K immagini)

Le predizioni di Gemini su tutto il dataset PMCimages (~32.000 immagini) sono già convertite in formato COCO e pronte per il training:

- `data/coco_all.json` — dataset completo (31.999 immagini, ~43 MB)
- `data/annotations/instances_train.json` — 28.800 immagini (90%)
- `data/annotations/instances_val.json` — 3.199 immagini (10%)

`notebooks/prepare_dataset.ipynb` documenta il pipeline completo che ha generato questi file (JSONL batch → COCO → train/val split).

### 4. Nuove predizioni batch

Per rieseguire le predizioni su un nuovo set di immagini:

```bash
export GEMINI_API_KEY="la_tua_chiave"

python scripts/run_batch_predictions.py \
    --input  /path/to/images \
    --output /path/to/results
```

Lo script trova automaticamente le sottocartelle di primo livello in `--input`, lancia un batch job Gemini separato per ognuna, le monitora in parallelo e scarica i risultati. Se interrotto, riprende da dove si era fermato grazie a `_state.json`.

---

## Struttura

```
GeminiDecomp/
├── config.py                        # path ai dati, classi, soglie IoU
├── environment.yml                  # ambiente conda
│
├── data/
│   ├── ground_truth.json            # 998 annotazioni manuali (piemmeci_test)
│   ├── gemini_predictions.json      # 998 predizioni Gemini (piemmeci_test)
│   ├── coco_all.json                # dataset COCO completo (~32K immagini)
│   └── annotations/
│       ├── instances_train.json     # split train (90%)
│       └── instances_val.json       # split val  (10%)
│
├── batch_predictions/               # JSONL raw delle predizioni Gemini su PMCimages
│   └── batch_pred__NN.jsonl         # un file per ogni batch (~1000 immagini ciascuno)
│
├── notebooks/
│   ├── evaluate_performance.ipynb   # metriche + grafici
│   ├── visualize_errors.ipynb       # visualizzazione FP/FN per immagine
│   └── prepare_dataset.ipynb        # pipeline JSONL → COCO → train/val split
│
├── scripts/
│   └── run_batch_predictions.py     # lancia predizioni batch su una cartella
│
├── src/
│   ├── gemini_client.py             # client Gemini API (upload, batch job, download)
│   ├── batch_pipeline.py            # wrapper alto livello per la pipeline batch
│   ├── coco_converter.py            # conversione JSONL Gemini → formato COCO
│   ├── data_loader.py               # carica GT e predizioni dal formato LabelStudio
│   ├── iou_matching.py              # IoU + Hungarian matching (scipy)
│   ├── metrics.py                   # Precision/Recall/F1, sweep IoU, mF1
│   └── visualization.py             # tutti i grafici matplotlib/seaborn
│
└── results/                         # grafici generati dai notebook
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate gemini_decomp
jupyter notebook
```

Per le predizioni batch serve anche la variabile d'ambiente `GEMINI_API_KEY`.

---

## Formato dati

**Ground truth / predizioni** (`data/*.json`): formato LabelStudio export, coordinate come percentuale (0–100) rispetto alle dimensioni originali dell'immagine.

**COCO** (`data/annotations/*.json`, `data/coco_all.json`): formato COCO standard, bounding box `[x, y, width, height]` in pixel assoluti. Categoria unica: `subimage` (id=0).

**JSONL batch Gemini** (`batch_predictions/*.jsonl`): una riga per immagine, bbox nel formato `[ymin, xmin, ymax, xmax]` normalizzato su scala 0–1000.
