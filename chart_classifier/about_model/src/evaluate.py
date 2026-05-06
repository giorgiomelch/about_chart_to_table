import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

import config
from model import ChartClassifier
from dataset import safe_pil_loader, get_val_transforms
from training_plotter import TrainingPlotter


class TestImageFolder(datasets.ImageFolder):
    """ImageFolder che restituisce (image, label, path) per tracciare i filename nel CSV."""

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        image = self.loader(path)
        if image is None:
            image = torch.zeros((3, config.IMG_SIZE, config.IMG_SIZE))
        elif self.transform:
            image = self.transform(image)
        return image, label_idx, path


def evaluate(args):
    if not os.path.isdir(args.test_dir):
        print(f"[ERR] Cartella test non trovata: {args.test_dir}")
        return

    # 1. Dataset
    dataset = TestImageFolder(
        root=args.test_dir,
        transform=get_val_transforms(),
        loader=safe_pil_loader,
    )
    class_names = dataset.classes
    print(f"[INFO] Test set: {len(dataset)} immagini | Classi: {class_names}")

    # 2. Modello
    print(f"[INFO] Caricamento modello da: {args.model_path}")
    device = config.DEVICE
    model = ChartClassifier(checkpoint_path=args.model_path).to(device)
    model.eval()

    # 3. DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(str(device) == "cuda"),
    )

    # 4. Inferenza
    all_preds = []
    all_labels = []
    all_paths = []
    all_confs = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Valutazione", unit="batch"):
            images = images.to(device)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())
            all_paths.extend(paths)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 5. Metriche
    accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\n[RISULTATI] Accuracy: {accuracy:.2f}% su {len(all_labels)} immagini\n")

    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    print(report)

    # 6. Salvataggio output
    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}% su {len(all_labels)} immagini\n\n")
        f.write(report)
    print(f"[INFO] Report salvato in: {report_path}")

    csv_path = os.path.join(args.output_dir, "detailed_results.csv")
    with open(csv_path, "w") as f:
        f.write("filename,true_label,predicted_label,confidence\n")
        for path, true_idx, pred_idx, conf in zip(all_paths, all_labels, all_preds, all_confs):
            filename = os.path.basename(path)
            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]
            f.write(f"{filename},{true_label},{pred_label},{conf:.4f}\n")
    print(f"[INFO] Risultati dettagliati salvati in: {csv_path}")

    plotter = TrainingPlotter(save_dir=args.output_dir, class_names=class_names)
    plotter.save_confusion_matrix(list(all_preds), list(all_labels), tag="test_set")
    print(f"[INFO] Confusion matrix salvata in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione automatica sul test set in data/test/")
    parser.add_argument("--test_dir", type=str, default=os.path.join(config.DATA_DIR, "test"),
                        help="Cartella test in formato ImageFolder (default: data/test/)")
    parser.add_argument("--model_path", type=str, default=os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME),
                        help="Percorso ai pesi .pth del modello")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size per inferenza")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                        help="Numero di workers DataLoader")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory dove salvare i risultati")

    args = parser.parse_args()
    evaluate(args)
