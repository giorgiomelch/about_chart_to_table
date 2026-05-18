# deformable-detr-infer

Module for **compound figure decomposition**: given a scientific image containing multiple side-by-side charts (a multi-panel figure), it detects and crops each sub-panel as a separate image.

**Performance**: Precision 0.927 · Recall 0.930 · F1 0.928 @ IoU=0.5

---

## Architecture

- **Deformable DETR** (Detection Transformer with deformable attention)
- Backbone: **ResNet-50**
- The training of the model is based on the official [Deformable DETR GitHub repository](https://github.com/fundamentalvision/Deformable-DETR), cloned and adapted for compound figure detection
- Trained on ~1,000 manually annotated compound figures + ~32,000 weakly labelled by Gemini 3 Flash

---

## Structure

```
deformable-detr-infer/
├── infer.py                    # Entry point (single image or folder)
├── args_config.py              # Model argument configuration
├── checkpoints/
│   └── checkpoint05.pth        # Pre-trained checkpoint (tracked in git)
├── models/
│   ├── deformable_detr.py      # Main model architecture
│   ├── deformable_transformer.py
│   └── backbone.py             # ResNet-50 feature extractor
└── datasets/
    └── transforms.py           # Image pre-processing
```

---

## Requirements

```bash
pip install torch torchvision pillow
```

To compile Deformable DETR's custom multi-scale attention operators:

```bash
cd models/ops
python setup.py build install
```

---

## Usage

```bash
cd deformable-detr-infer

# Single image
python infer.py --input inputs/figure.jpg

# Entire folder
python infer.py --input inputs/

# With custom options
python infer.py \
    --input inputs/ \
    --output outputs/ \
    --threshold 0.4 \
    --checkpoint checkpoints/checkpoint05.pth
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Image file or folder of images |
| `--output` | `outputs/` | Folder where cropped sub-panels are saved |
| `--threshold` | 0.5 | Confidence threshold for detections |
| `--checkpoint` | `checkpoints/checkpoint05.pth` | Path to the model checkpoint |

---

## Output

For each input image a sub-folder is created inside `--output` containing the individual cropped panels as separate files. The resulting images are then classified by [chart_classifier](../chart_classifier/README.md).

---

## Comparison with GeminiDecomp

See [GeminiDecomp/README.md](../GeminiDecomp/README.md) for Gemini 3 Flash results as a zero-shot alternative (F1=0.94 @ IoU=0.5 but with API cost).
