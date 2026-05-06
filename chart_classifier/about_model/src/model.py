import torch
import torch.nn as nn
import timm
import config
from pathlib import Path

class ChartClassifier(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(ChartClassifier, self).__init__()
        
        model_name = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
        
        print(f"Caricamento modello: {model_name}...")

        use_pretrained = checkpoint_path is None

        # Creazione del modello base
        self.model = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=config.NUM_CLASSES,
            img_size=config.IMG_SIZE
        )
        self.model.set_grad_checkpointing(config.GRAD_CHECKPOINTING)

        if checkpoint_path:
            if not Path(checkpoint_path).exists():
                print(f"[ERR] Il file checkpoint non esiste: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
            else:
                self.load_checkpoint(checkpoint_path)

    def forward(self, x):
        return self.model(x)
    
    def load_checkpoint(self, path):
        """Carica i pesi da un file .pth"""
        print(f"Caricamento pesi dal file: {path}")
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=torch.device(device_name))
        
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        print("Pesi caricati correttamente.")