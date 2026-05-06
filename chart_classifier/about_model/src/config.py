import torch

# --- PERCORSI ---
DATA_DIR = "./data"
SAVE_DIR = "./training_output"
BEST_MODEL_NAME = "chart_classifier_fra/best_model.pth"

# --- DATI ---
IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_CLASSES = 24

# --- IPERPARAMETRI TRAINING ---
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP = 5.0
GRAD_CHECKPOINTING = True

# --- NORMALIZZAZIONE ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['area', 'bar', 'box', 'bubble', 'chord', 'contour', 'diagram', 'errorpoint', 'heatmap', 'histogram', 'image', 'line', 'manhattan', 'map', 'molecule', 'pie', 'quiver', 'radar_polar', 'scatter', 'surface3d', 'table', 'treemap', 'venn', 'violin']