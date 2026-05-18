# chart_classifier

Module for automatically classifying the chart type of scientific images. Given a set of images (from [articles_images/](../articles_images/README.md)), it assigns each one to one of **24 chart categories**.

**Result**: 98.46% accuracy on the test set.

---

## Architecture

The classifier is based on **SwinV2-Base** (Swin Transformer V2, Base variant), fine-tuned on ~67,000 labelled images from 9+ public datasets combined with synthetically generated charts.

---

## Structure

```
chart_classifier/
├── about_data_management/    # Dataset collection and synthetic generation
│   ├── chart_factory/        # Scripts to generate synthetic charts (18+ types)
│   ├── datasets/             # Aggregated public datasets
│   └── ...
└── about_model/              # Training and inference code
    ├── src/
    │   ├── train.py          # Training loop
    │   ├── inference.py      # Batch inference on a folder
    │   ├── model.py          # SwinV2-Base wrapper
    │   ├── dataset.py        # PyTorch Dataset
    │   └── config.py         # Parameters (IMG_SIZE, CLASS_NAMES, SAVE_DIR, …)
    └── training_output/      # Saved checkpoints
```

See the README of each sub-folder for details:

- [about_data_management/README.md](about_data_management/README.md) — how the training dataset was built
- [about_model/README.md](about_model/README.md) — how to train the model and run inference

---

## Supported categories (24)

area, bar, box, bubble, chord, confusion matrix, contour, errorpoint, heatmap, histogram, line, manhattan, pie, quiver, radar, scatter, surface, treemap, venn, violin, and additional minor categories.
