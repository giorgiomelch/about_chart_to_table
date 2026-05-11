"""
Inferenza Deformable DETR — singola immagine o cartella intera.

Uso:
  # Singola immagine
  python infer.py --input inputs/image.jpg

  # Cartella intera
  python infer.py --input inputs/

  # Opzioni aggiuntive
  python infer.py --input inputs/ --output outputs/ --threshold 0.4 --checkpoint checkpoints/checkpoint05.pth
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import datasets.transforms as T
from args_config import get_args_parser as get_model_args_parser
from models import build_model

CHECKPOINT_DEFAULT = "checkpoints/checkpoint05.pth"
NUM_CLASSES_DEFAULT = 1
THRESHOLD_DEFAULT = 0.5
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

COLORS = [
    (0, 114, 189),
    (217, 83, 25),
    (237, 177, 32),
    (126, 47, 142),
    (119, 172, 48),
]


def load_model(checkpoint_path, num_classes, device):
    parser = get_model_args_parser()
    args = parser.parse_args([])
    args.num_classes = num_classes
    args.two_stage = True
    args.with_box_refine = True
    args.device = str(device)

    model, _, _ = build_model(args)
    model.to(device)

    print(f"Caricamento pesi da {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def get_transform():
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(model, transform, image_path, device, threshold):
    im_pil = Image.open(image_path).convert("RGB")
    w_orig, h_orig = im_pil.size

    img_tensor, _ = transform(im_pil, target=None)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]

    scores = logits.sigmoid().max(dim=-1)[0]
    keep = scores > threshold
    keep_scores = scores[keep].cpu().tolist()

    abs_boxes = []
    for box in boxes[keep].cpu():
        cx, cy, bw, bh = box.tolist()
        abs_boxes.append((
            (cx - bw / 2) * w_orig,
            (cy - bh / 2) * h_orig,
            (cx + bw / 2) * w_orig,
            (cy + bh / 2) * h_orig,
        ))

    return im_pil, keep_scores, abs_boxes


def draw_and_save(im_pil, scores, boxes, output_path):
    draw = ImageDraw.Draw(im_pil)
    for i, (score, (xmin, ymin, xmax, ymax)) in enumerate(zip(scores, boxes)):
        color = COLORS[i % len(COLORS)]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin + 4, ymin + 2), f"{score:.2f}", fill=color)
    im_pil.save(output_path)


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Deformable DETR — inferenza")
    parser.add_argument("--input", required=True, help="Immagine singola o cartella")
    parser.add_argument("--output", default="outputs/", help="Cartella output (default: outputs/)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES_DEFAULT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help="Soglia di confidenza (default 0.5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.num_classes, device)
    transform = get_transform()

    images = collect_images(args.input)
    if not images:
        print(f"Nessuna immagine trovata in: {args.input}")
        return

    print(f"Immagini da processare: {len(images)}")
    for img_path in images:
        try:
            im_pil, scores, boxes = predict(model, transform, img_path, device, args.threshold)
        except Exception as e:
            print(f"  ERRORE su {img_path.name}: {e}")
            continue

        out_path = output_dir / img_path.name
        draw_and_save(im_pil, scores, boxes, out_path)
        print(f"  {img_path.name}: {len(scores)} detection(s) -> {out_path}")

    print(f"\nDone. Risultati salvati in: {output_dir}")


if __name__ == "__main__":
    main()
