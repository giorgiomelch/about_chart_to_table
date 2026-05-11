import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from pathlib import Path
from huggingface_hub import snapshot_download
from src.utils.prompts import PROMPT2CHARTCLASS, PROMPT_AreaLineBarHistogram
from src.config import WEIGHTS_DIR, IMAGES_DIR, PREDICTIONS_DIR


MAX_SIDE_LIMIT = 768

QWEN_MODELS = {
    "2B": "Qwen/Qwen2-VL-2B-Instruct",
    "7B": "Qwen/Qwen2-VL-7B-Instruct"
}

# --- GESTIONE MODELLO ---

def get_local_path(tier):
    """Restituisce il percorso locale in base alla taglia scelta."""
    return WEIGHTS_DIR / f"Qwen2-VL-{tier}-Instruct"

def download_qwen_model(tier="2B"):
    """Scarica il modello specificato se non è già presente localmente."""
    if tier not in QWEN_MODELS:
        raise ValueError(f"Taglia non supportata. Scegli tra: {list(QWEN_MODELS.keys())}")
        
    repo_id = QWEN_MODELS[tier]
    output_path = get_local_path(tier)
    
    if output_path.exists() and any(output_path.glob("*.safetensors")):
        print(f"Modello {tier} già presente in {output_path}. Skip download.")
        return str(output_path)
        
    print(f"Inizio download del modello {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_path),
        ignore_patterns=["*.pt", "*.bin"]
    )
    print(f"Download completato e salvato in {output_path}")
    return str(output_path)

def setup_qwen2_vl(tier="2B", use_4bit=True):
    """
    Carica il modello Qwen2-VL a 4-bit specificato dal parametro tier.
    """
    model_id = download_qwen_model(tier)
    
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True
    }
    if use_4bit:
        print(f"\nCaricamento di Qwen2-VL ({tier}) a 4-bit in corso...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"\nCaricamento di Qwen2-VL ({tier}) a piena precisione ({dtype}) in corso...")
        model_kwargs["torch_dtype"] = dtype

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28, 
        max_pixels=MAX_SIDE_LIMIT * MAX_SIDE_LIMIT
    )
    
    return model, processor

# --- INFERENZA ---

def extract_table_qwen(model, processor, image_path, prompt_text):
    """
    Esegue l'inferenza su una singola immagine.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=10_000,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


def run_batch_inference(model, processor, tier="2B"):
    """
    Itera su tutte le immagini, seleziona il prompt in base alla chart class, 
    genera i JSON e replica l'alberatura delle cartelle.
    """
    input_base_dir = IMAGES_DIR
    output_base_dir = PREDICTIONS_DIR / f"Qwen{tier}"
    
    if not input_base_dir.exists():
        print(f"Errore: La directory di input {input_base_dir} non esiste.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for img_path in input_base_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            
            relative_path = img_path.relative_to(input_base_dir)
            json_output_path = output_base_dir / relative_path.with_suffix('.json')
            
            if json_output_path.exists():
                print(f"Skip: {json_output_path.name} (già processato).")
                continue
            
            # --- SELEZIONE DINAMICA DEL PROMPT ---
            # Verifica che il percorso abbia la profondità necessaria
            if len(relative_path.parts) >= 2:
                chart_class = relative_path.parts[1]
            else:
                print(f"Attenzione: Impossibile determinare la classe per {relative_path}. Salto.")
                continue
            
            # Recupera il prompt dal dizionario (con fallback al prompt generico se la classe non esiste nel dict)
            prompt_scelto = PROMPT2CHARTCLASS.get(chart_class, PROMPT_AreaLineBarHistogram)
            
            print(f"Elaborazione: {relative_path} [Classe: {chart_class}]...")
            json_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Esegue l'inferenza passando il prompt dinamico
                raw_result = extract_table_qwen(model, processor, img_path, prompt_scelto)
                
                # Pulizia sintattica
                clean_json = raw_result.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                # Validazione strutturale
                try:
                    json.loads(clean_json)
                    json_output_path.write_text(clean_json, encoding="utf-8")
                except json.JSONDecodeError as e:
                    print(f"  [ATTENZIONE] Il modello ha generato un JSON non valido per {img_path.name}: {e}")
                    fallback_content = json.dumps({
                        "error": "JSONDecodeError",
                        "raw_model_output": clean_json
                    }, indent=4, ensure_ascii=False)
                    json_output_path.write_text(fallback_content, encoding="utf-8")
                
            except Exception as e:
                print(f"Errore critico durante l'elaborazione di {img_path}: {e}")
                critical_error_content = json.dumps({
                    "error": "CriticalInferenceError",
                    "details": str(e)
                }, indent=4, ensure_ascii=False)
                json_output_path.write_text(critical_error_content, encoding="utf-8")

def ask_qwen(tier="2B", quantizzazione=True):
    modello, processore = setup_qwen2_vl(tier=tier, use_4bit=quantizzazione)
    
    run_batch_inference(modello, processore, tier=tier)
    print(f"\nQuen{tier}: inferenza batch completata.")

if __name__ == "__main__":
    ask_qwen()