import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import argparse
import glob
from tqdm import tqdm
import config
from model import ChartClassifier
from dataset import get_val_transforms, safe_pil_loader

class InferenceDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = safe_pil_loader(img_path)
        
        if image is None:
             image = torch.zeros((3, config.IMG_SIZE, config.IMG_SIZE))
        else:
            if self.transform:
                image = self.transform(image)
                
        return image, img_path


def run_batch_inference(model, dataloader, device, class_names):
    model.eval()
    results = [] # Lista di tuple: (filename, label, confidence)

    print(f"[INFO] Avvio inferenza su {len(dataloader.dataset)} immagini...")
    
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader, unit="batch"):
            batch_images = batch_images.to(device)
            
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                outputs = model(batch_images)
                probs = F.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probs, 1)

            # Porta i risultati su CPU per salvarli
            predicted_indices = predicted_indices.cpu().numpy()
            confidences = confidences.cpu().numpy()

            for i, path in enumerate(batch_paths):
                idx = predicted_indices[i]
                conf = confidences[i]
                label = class_names[idx] if class_names else str(idx)
                
                filename = os.path.basename(path)
                results.append((filename, label, conf))
    
    return results

def predict_directory(target_data, batch_size=16, model_path=None, num_workers=4):
    device = config.DEVICE
    
    # 1. Recupera lista file
    file_list = []
    if os.path.isfile(target_data):
        file_list = [target_data]
    elif os.path.isdir(target_data):
        exts = ('**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.bmp')
        print(f"[INFO] Scansione ricorsiva cartelle in: {target_data}...")
        for ext in exts:
            file_list.extend(glob.glob(os.path.join(target_data, ext), recursive=True))
            file_list.extend(glob.glob(os.path.join(target_data, ext.upper()), recursive=True))
        file_list = sorted(list(set(file_list)))
        if not file_list:
            print(f"[ERR] Nessuna immagine trovata in {target_data} o nelle sue sottocartelle.")
            return
    else:
        print(f"[ERR] Percorso non valido: {target_data}")
        return

    # 2. Carica Modello
    checkpoint_path = model_path if model_path else os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME)
    print(f"[INFO] Load model: {checkpoint_path}")
    
    try:
        model = ChartClassifier(checkpoint_path=checkpoint_path).to(device)
    except Exception as e:
        print(f"[ERR] Errore caricamento modello: {e}")
        return

    # 3. Prepara DataLoader
    dataset = InferenceDataset(file_list, transform=get_val_transforms())
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    class_names = config.CLASS_NAMES

    # 5. Esegui
    results = run_batch_inference(model, dataloader, device, class_names)
    if not os.path.exists("pred_results"):
        os.makedirs("pred_results")
        print(f"[INFO] Creata cartella di output: pred_results")
    source_name = os.path.basename(os.path.normpath(target_data))
    model_filename = os.path.basename(checkpoint_path)
    model_name_stem = os.path.splitext(model_filename)[0]
    output_csv = f"pred_results/{source_name}_{model_name_stem}.csv"
    with open(output_csv, "w") as f:
        f.write("filename,prediction,confidence\n")
        for filename, label, conf in results:
            f.write(f"{filename},{label},{conf:.4f}\n")
    print(f"[INFO] Risultati salvati in {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predici per un'immagine o una cartella contenente immagini le classi di grafici")
    
    parser.add_argument("--target", type=str, required=True, help="Percorso alla cartella madre o immagine")
    
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per inferenza")
    parser.add_argument("--model_path", type=str, default=None, help="Percorso specifico ai pesi .pth (opzionale)")
    parser.add_argument("--workers", type=int, default=8, help="Numero di workers per il dataloader")

    args = parser.parse_args()

    predict_directory(
        target_data=args.target,
        batch_size=args.batch_size,
        model_path=args.model_path,
        num_workers=args.workers
    )