import io
import struct
import torch
from torchvision import datasets, transforms
from PIL import Image
import config
import random

class ImageResize:
    """
    Ridimensiona l'immagine mantenendo l'aspect ratio e aggiunge padding (bianco default)
    per renderla quadrata.
    """
    def __init__(self, target_size, scale_range=None, p_scale=0.5, fill_color=(255, 255, 255)):
        self.target_size = target_size
        self.scale_range = scale_range
        self.p_scale = p_scale
        self.fill_color = fill_color

    def __call__(self, img):
        original_width, original_height = img.size
        max_side = max(original_width, original_height)
        # Determina la scala (per train con probabilità p_scale casuale, fissa a 1.0 per val)
        if self.scale_range is not None and random.random() < self.p_scale:
            random_scale = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            random_scale = 1.0
            
        eff_target = int(self.target_size * random_scale)
        ratio = min(1.0, eff_target / max_side)
        new_width = max(1, int(original_width * ratio))
        new_height = max(1, int(original_height * ratio))
        # ridimensiona solo per rimpicciolire, mai per ingrandire
        if ratio < 1.0:
            img = img.resize((new_width, new_height), Image.BICUBIC)
        new_img = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
def _pil_open_to_rgb(f):
    with Image.open(f) as img:
        img = img.convert('RGBA')
        white_background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        white_background.paste(img, (0, 0), img)
        return white_background.convert('RGB')


_PNG_CRITICAL_CHUNKS = {b'IHDR', b'PLTE', b'IDAT', b'IEND'}

def _strip_png_metadata(data):
    """Rimuove i chunk non critici (es. zTXt con CRC errato) da un PNG."""
    out = bytearray(data[:8])  # PNG signature
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_end = pos + 12 + length
        if chunk_type in _PNG_CRITICAL_CHUNKS:
            out.extend(data[pos:chunk_end])
        pos = chunk_end
    return bytes(out)


def safe_pil_loader(path):
    """
    Loader che apre l'immagine e converte subito la trasparenza in bianco.
    Fallback che strappa i metadati PNG corrotti (es. chunk zTXt di draw.io).
    """
    try:
        with open(path, 'rb') as f:
            return _pil_open_to_rgb(f)
    except Exception:
        pass

    try:
        with open(path, 'rb') as f:
            data = _strip_png_metadata(f.read())
        return _pil_open_to_rgb(io.BytesIO(data))
    except Exception:
        return None

def get_train_transforms():
    """Restituisce le trasformazioni per train"""
    train_transforms = transforms.Compose([
        ImageResize(config.IMG_SIZE),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            fill=255
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomInvert(p=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5))
        ], p=0.05),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    return train_transforms

def get_val_transforms():
    """Restituisce le trasformazioni per val"""
    val_transforms = transforms.Compose([
        ImageResize(config.IMG_SIZE, scale_range=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return val_transforms

def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    
    try:
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tf, loader=safe_pil_loader)
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_tf, loader=safe_pil_loader)
    except FileNotFoundError as e:
        print(f"Errore: Cartelle non trovate in {data_dir}. Verifica il percorso in config.py")
        raise e

    print(f"Classi trovate: {train_dataset.classes}")
    
    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes