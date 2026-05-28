# PaperUnPlotBench

The main benchmark of the project: **950 real charts** extracted from scientific literature (PMC + arXiv) paired with **950 synthetic charts** (one-to-one correspondence), organised into 10 categories, with ground-truth annotations and an evaluation pipeline for 8 models.

---

## Dataset

### Composition

| Split | Source | Charts |
|-------|--------|--------|
| `PMCharts` | PubMed Central (real) | 500 (50 per category) |
| `arXiv` | arXiv (real) | 450 (50 per category, except bubble: 0) |
| `PMCharts_synthetic` | Synthetic from PMCharts GT | 500 |
| `arXiv_synthetic` | Synthetic from arXiv GT | 450 |

### Categories (10)

`bar` · `box` · `bubble` · `errorpoint` · `heatmap` · `histogram` · `line` · `pie` · `radar` · `scatter`

### Folder structure

```
PaperUnPlotBench/
├── data/
│   ├── images/
│   │   ├── PMCharts/           ← 500 real PMC charts
│   │   │   ├── bar/
│   │   │   ├── box/
│   │   │   └── ...
│   │   ├── arXiv/              ← 450 real arXiv charts
│   │   ├── PMCharts_synthetic/ ← 500 synthetic charts
│   │   └── arXiv_synthetic/    ← 450 synthetic charts
│   └── groundtruth/            ← JSON annotations (one per image)
├── src/
│   ├── models/                 ← model interfaces
│   ├── evaluation/             ← NMS evaluation pipeline
│   └── utils/                  ← prompts, JSON schema, helpers
├── chart_factory/
│   └── generate_from_gt.py     ← synthetic chart generation
├── notebooks/                  ← analysis, correlations, cost estimation
├── outputs/
│   ├── predictions/            ← model predictions (JSON)
│   ├── metrics/                ← NMS metrics per model/category
│   └── reports/                ← HTML comparison reports (GT vs predictions)
└── run_benchmark.py            ← main entry point
```

### Ground-truth format

Each JSON file in `data/groundtruth/` follows this schema:

```json
{
  "chart_title": "Effect of temperature on yield",
  "x_axis_label": "Temperature (°C)",
  "y_axis_label": "Yield (%)",
  "x_axis": {
        "min": 20,
        "max": 100,
        "is_log": false
    },
  "y_axis": {
        "min": 0,
        "max": 100,
        "is_log": false
    },
  "data_points": [
    {"series_name": "Method A", "x_value": "20", "y_value": 45.2},
    {"series_name": "Method A", "x_value": "40", "y_value": 67.8}
  ]
}
```

---

## Running the benchmark

### Requirements

```bash
pip install torch torchvision transformers pillow anthropic openai google-generativeai
```

API keys required (commercial models only):

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

### Main commands

```bash
cd PaperUnPlotBench

# Inference with Gemini 2.5 Flash on PMCharts
python run_benchmark.py --model gemini --tier gemini-2.5-flash --dataset PMCharts

# Multiple open-weight models on all datasets
python run_benchmark.py --model qwen,internvl --dataset all

# Inference + evaluation + HTML reports
python run_benchmark.py --model deplot --evaluate --report

# Evaluation only (predictions must already exist in outputs/predictions/)
python run_benchmark.py --evaluate --metrics --report

# On Google Colab with data on Drive
python run_benchmark.py --model qwen --dataset PMCharts \
    --drive-path /content/drive/MyDrive/MyProject \
    --data-path /content/drive/MyDrive/MyProject/data
```

### Available models

| Key | Model | Type |
|-----|-------|------|
| `gemini` | Gemini 2.5 Flash (default) | API |
| `openai` | GPT-4o (default) | API |
| `qwen` | Qwen2-VL 2B (default) | Open-weight |
| `internvl` | InternVL2.5-2B (default) | Open-weight |
| `phi` | Phi-3.5-Vision | Open-weight |
| `deplot` | DePlot / Pix2Struct | Task-specific |

Use `--tier` to select a different variant (e.g. `--tier 7B` for Qwen, `--tier gpt-4o-mini` for OpenAI).

---

## Metric: NMS (Normalized Mapping Similarity)

NMS extends RMS (Relative Mapping Similarity) with:

1. **Axis-range normalisation**: error is divided by the axis range, not by the absolute target value
2. **Logarithmic axis support**: distance computed in log-space
3. **Optimal matching**: Hungarian algorithm on `(x_header, y_header, value)` triples
4. **Output**: Precision, Recall, F1 per model / category / split

---

## Analysis notebooks

```bash
cd PaperUnPlotBench/notebooks

# Dataset statistics (data points per category, value distributions)
jupyter notebook dataset_analysis.ipynb

# Cross-model performance correlation
jupyter notebook correlation_analysis.ipynb

# API inference cost estimation
jupyter notebook cost_estimation.ipynb

# Debug missing predictions
jupyter notebook missing_predictions.ipynb
```

---

## Output files

| Folder | Contents |
|--------|----------|
| `outputs/predictions/` | JSON files with each model's predictions |
| `outputs/metrics/` | NMS scores (P/R/F1) per model, category and split |
| `outputs/reports/` | HTML reports with visual comparison of predictions vs GT |

Files in `outputs/` are not tracked in git.

---

## Generating synthetic charts

Synthetic charts are generated from ground-truth JSON files by `chart_factory/generate_from_gt.py`:

```bash
python chart_factory/generate_from_gt.py \
    --gt-dir data/groundtruth/ \
    --out-dir data/images/PMCharts_synthetic/
```

---

## Key results

| Model | NMS F1 (PMCharts) |
|-------|-------------------|
| Gemini 2.5 Flash | **91.2%** |
| GPT-4o | ~87% |
| Qwen2-VL 2B | ~58% |
| InternVL2 2B | ~52% |
| Phi-3.5 Vision | ~49% |
| DePlot | ~31% (bar/line/pie only) |
