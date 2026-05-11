from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Dati valutazione piemmeci_test ─────────────────────────────────────────
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth_testset.json"
PREDICTIONS_PATH  = DATA_DIR / "gemini_predictions_on_testset.json"
IMAGES_DIR = Path("PMC_images_testset_decompose_task")

# ── Dataset COCO (annotazioni Gemini su tutto PMCimages) ───────────────────
COCO_ALL_PATH   = DATA_DIR / "coco_all.json"
COCO_TRAIN_PATH = DATA_DIR / "annotations" / "instances_train.json"
COCO_VAL_PATH   = DATA_DIR / "annotations" / "instances_val.json"

# Directory con le immagini PMCimages (sorgente per COCO dataset)
PMCIMAGES_DIR = Path("/home/giorgiomelch/VS_Workspace/GitHub/chart_distribution_over_pubmed/data/PMCimages")

# Raw JSONL batch predictions (originali, non copiati — troppo grandi)
RAW_BATCH_PREDICTIONS_DIR = Path("GeminiDecomp/batch_predictions")

# ── Valutazione ────────────────────────────────────────────────────────────
CLASSES = ["chart", "non_chart"]
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
IOU_MAIN = 0.5

# ── Batch inference ────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-3-flash-preview"
COCO_VAL_RATIO = 0.1
COCO_SEED      = 42
