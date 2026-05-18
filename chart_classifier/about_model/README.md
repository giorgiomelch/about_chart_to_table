# about_model

Implementation of the chart classifier based on **SwinV2-Base**. 

---

## Structure

```
about_model/
├── src/
│   ├── train.py          # Training loop with weighted loss
│   ├── inference.py      # Batch inference on a folder or single image
│   ├── evaluate.py       # Metric computation on the validation set
│   ├── model.py          # ChartClassifier wrapper (SwinV2-Base)
│   ├── dataset.py        # PyTorch Dataset + transforms
│   ├── config.py         # Parameters (IMG_SIZE, CLASS_NAMES, SAVE_DIR, …)
│   └── utils.py          # Seed, miscellaneous helpers
└── training_output/      # Saved .pth checkpoints
```

---

## Requirements

```bash
pip install torch torchvision timm tqdm
```

A GPU (CUDA) is strongly recommended. The code also runs on CPU but is significantly slower.

---

## Training

```bash
cd about_model/src

python train.py
```

Training uses:
- **Weighted Cross-Entropy Loss** to handle class imbalance
- **CosineAnnealingLR** as the learning rate scheduler
- **Mixed precision** (AMP) to speed up GPU training

Parameters (learning rate, batch size, epochs, data paths) are configured in `config.py`.

Checkpoints are saved to `training_output/`; the checkpoint tracked in git is `training_output/dice_1000.pth`.

---

## Batch inference

```bash
cd about_model/src

# Classify all images in a folder
python inference.py --target /path/to/images/

# With a specific checkpoint and custom batch size
python inference.py --target /path/to/images/ --model_path ../training_output/dice_1000.pth --batch_size 32

# Single image
python inference.py --target /path/to/chart.png
```

**Output**: a CSV file at `pred_results/<folder_name>_<checkpoint_name>.csv` with columns `filename, prediction, confidence`.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | required | Folder or image file path |
| `--batch_size` | 2 | DataLoader batch size |
| `--model_path` | see config.py | Path to the .pth checkpoint |
| `--workers` | 8 | Number of DataLoader workers |

---

## Supported classes

The 24 categories are defined in `config.py` under `CLASS_NAMES` and correspond to the sub-folders of the training dataset.
