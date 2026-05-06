import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
import config
import argparse
from dataset import get_dataloaders
from utils import seed_everything
from model import ChartClassifier
from training_plotter import TrainingPlotter
import numpy as np
from collections import Counter

def get_class_weights(dataset, class_names, device):
    """Calcola i pesi delle classi basandosi sulla distribuzione nel dataset.
      Formula: peso_i = totale_campioni / (numero_classi * campioni_classe_i)"""
    # Estrae i label dal dataset (funziona con ImageFolder)
    targets = dataset.targets
    counts = Counter(targets)
    # Ordina i conteggi in base all'indice della classe per garantire la corrispondenza
    class_indices = sorted(counts.keys())
    n_samples = [counts[i] for i in class_indices]
    total_samples = sum(n_samples)
    n_classes = len(class_indices)
    # Calcolo pesi con bilanciamento standard
    weights = [total_samples / (n_classes * n) for n in n_samples]
    print(f"[INFO] Pesi calcolati: {weights}")
    for i, name in enumerate(class_names):
        print(f"   - Classe {i} ({name}): Peso {weights[i]:.4f}")
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch_idx, device):
    """Esegue un'epoca di training: Forward + Backward + Update"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Barra di progresso
    loop = tqdm(loader, desc=f"Epoch [{epoch_idx+1}] ", leave=False, mininterval=0.5, smoothing=0.1, ncols=110)
    
    for batch_idx, (images, labels) in enumerate(loop):
        # Sposta i dati sulla GPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        # Forward Pass
        with autocast(device_type=device):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping 
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        # Step dell'optimizer
        scaler.step(optimizer)
        scaler.update()

        # Statistiche
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Aggiorna la barra
        loop.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Valuta il modello sul set di validazione"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast(device_type=device):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def resume_training(args, model, optimizer, scheduler, scaler):
    """    Carica il checkpoint e RESTITUISCE epoch e best_acc.    """
    if os.path.isfile(args.resume):
        print(f"==> Caricamento checkpoint da: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        
        # Ripristina epoch e best_acc
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        
        # Ripristina stati
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"==> Training ripreso dall'epoca {start_epoch}")
        return start_epoch, best_acc
    else:
        print(f"==> ERRORE: Nessun checkpoint trovato in {args.resume}")
        return 0, 0.0
    
def save_checkpoint(state, filename):
    print(f"=> Salvataggio checkpoint: {filename}")
    torch.save(state, filename)

def train(args):
    seed_everything(42)
    
    print(f"[INFO] Avvio training su: {args.device}")
    print(f"[INFO] Parametri: LR={args.lr}, BS={args.batch_size}, Epochs={args.epochs}")
    
    # Creazione cartella pesi
    experiment_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Path file
    best_model_path = os.path.join(experiment_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(experiment_dir, "last_checkpoint.pth")
    
    # 1. Dati e Modello
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    model = ChartClassifier(args.model_path).to(args.device)
    
    # 2. Configurazione Training
    weights = get_class_weights(train_loader.dataset, class_names, args.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion_val = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Scaler per Mixed Precision
    scaler = GradScaler("cuda")
    
    # Variabili per monitoraggio
    best_acc = 0.0
    plotter = TrainingPlotter(save_dir=experiment_dir, class_names=class_names)

    start_epoch = 0
    if args.resume:
        start_epoch, best_acc = resume_training(args, model, optimizer, scheduler, scaler)
        
    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, args.device)
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion_val, args.device)
        
        # Update LR
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        
        print(f" -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}% | LR: {curr_lr:.2e}")
        plotter.update(train_loss, val_loss, train_acc, val_acc)
        # --- LOGICA SALVATAGGIO ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Nuovo record! Modello salvato in {best_model_path}")
            plotter.save_confusion_matrix(preds, labels, tag=args.name)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'scaler': scaler.state_dict()
        }
        save_checkpoint(checkpoint, last_checkpoint_path)
    print("\n[DONE] Training terminato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--name", type=str, default="swinv2_chart", help="Nome opzionale per taggare l'addestramento")
    parser.add_argument("--save_dir", type=str, default=config.SAVE_DIR, help="Cartella dove salvare pesi e andamenti dell'addestramento")
    parser.add_argument("--model_path", type=str, default=None, help="Percorso specifico ai pesi .pth (opzionale solo per secondo fine tuning)")
    parser.add_argument("--resume", type=str, default=None, help="Percorso al checkpoint .pth da cui riprendere")
    parser.add_argument("--data_dir", type=str, default=config.DATA_DIR, help="Percorso dataset")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Dimensione del batch")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="Numero workers")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning Rate iniziale")
    parser.add_argument("--wd", type=float, default=config.WEIGHT_DECAY, help="Weight Decay")
    parser.add_argument("--grad_ckpoint", type=bool, default=config.GRAD_CHECKPOINTING, help="Gradient Checkpointing attivo o non")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Dispositivo (cuda/cpu)")

    args = parser.parse_args()
    
    train(args)