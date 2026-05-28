# about_chart_to_table

The goal of the work is to study the real-world distribution of chart types in scientific literature (PubMed Central and arXiv), build a representative benchmark for the **chart-to-table extraction** task, and evaluate state-of-the-art vision-language models (LVLMs) on it.

---

## Pipeline overview

```
article_fetchers/
    Download articles from PMC and arXiv
            ↓
deformable-detr-infer/   +   GeminiDecomp/
    Decompose compound figures (multi-panel) into individual charts
            ↓
chart_classifier/
    Classify into 24 categories (SwinV2-Base, 98.46% accuracy) to get the chart type distribution
            ↓
annotate_charts/
    Manually annotate ground-truth for 950 charts (10 categories)
            ↓
PaperUnPlotBench/
    Benchmark: 950 real charts + 950 synthetic charts
    Evaluate 8 models with the NMS metric
```

---

## Quick-start: running the benchmark

```bash
cd PaperUnPlotBench

# Inference with Gemini 2.5 Flash on real PMCharts
python run_benchmark.py --model gemini --tier gemini-2.5-flash --dataset PMCharts

# Multiple open-weight models + evaluation + HTML reports
python run_benchmark.py --model qwen,internvl --dataset all --evaluate --report

# Evaluation only (requires predictions to already exist)
python run_benchmark.py --evaluate --metrics --report
```

See [PaperUnPlotBench/README.md](PaperUnPlotBench/README.md) for full details.

---

## Repository structure

```
about_chart_to_table/
├── article_fetchers/             # Download articles from PMC and arXiv
│   ├── arxiv_image_fetcher/
│   └── PubMedCentrale_image_fetcher/
├── articles_images/              # Image storage (not tracked in git)
│   ├── arXiv/
│   └── PubMedCentral/
├── chart_classifier/             # Chart type classifier (SwinV2-Base)
│   ├── about_data_management/    # Dataset collection and synthetic generation
│   └── about_model/              # Training, inference, checkpoints
├── deformable-detr-infer/        # Compound figure decomposition (DETR)
├── GeminiDecomp/                 # Gemini evaluation for decomposition
├── annotate_charts/              # Ground-truth annotation tools
├── PaperUnPlotBench/                # Main benchmark
```

---

## Evaluation metric: NMS (Normalized Mapping Similarity)

The benchmark introduces **NMS**, an extension of RMS (Relative Mapping Similarity) that:
- Normalises the error by the axis range (not the absolute target value)
- Supports logarithmic axes (distance computed in log-space)
- Uses the Hungarian algorithm for optimal matching between predictions and ground-truth
- Returns Precision, Recall and F1

---

## Requirements

Requirements vary per module. In general:

- Python 3.10+
- PyTorch 2.x with CUDA (for open-weight models)
- API keys for commercial models

Each subfolder contains its own README with installation instructions.
