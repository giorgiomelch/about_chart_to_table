import torch
import json
import gc
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
from huggingface_hub import snapshot_download

try:
    from src.utils.prompts import PROMPT2CHARTCLASS, PROMPT_AreaLineBarHistogram
    from src.config import WEIGHTS_DIR, IMAGES_DIR, PREDICTIONS_DIR
except ImportError:
    print("Attenzione: Moduli src non trovati. Verranno usati prompt e path di fallback.")
    PROMPT2CHARTCLASS = {}
    PROMPT_AreaLineBarHistogram = "Extract the data from this chart into a JSON format."
    WEIGHTS_DIR = Path("/content/drive/MyDrive/sota/weights")
    IMAGES_DIR = Path("/content/sota_chart_to_table/data/images")
    PREDICTIONS_DIR = Path("/content/drive/MyDrive/sota/outputs/predictions")

PHI_MODELS = {
    "3.5-Vision": "microsoft/Phi-3.5-vision-instruct"
}

# --- GESTIONE MODELLO ---

def get_local_path(tier):
    return WEIGHTS_DIR / f"Phi-{tier}"

def download_phi_model(tier="3.5-Vision"):
    if tier not in PHI_MODELS:
        raise ValueError(f"Taglia non supportata. Scegli tra: {list(PHI_MODELS.keys())}")
        
    repo_id = PHI_MODELS[tier]
    output_path = get_local_path(tier)
    
    if output_path.exists() and any(output_path.glob("*.safetensors")):
        print(f"Modello {tier} già presente in {output_path}. Skip download.")
        return str(output_path)
        
    print(f"Inizio download del modello {repo_id} su Google Drive...")
    # Crea le cartelle su Drive se non esistono
    output_path.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_path),
        ignore_patterns=["*.pt", "*.bin"] # Scarica solo i safetensors
    )
    print(f"Download completato e salvato in {output_path}")
    return str(output_path)

def setup_phi_vision(tier="3.5-Vision"):
    model_id = download_phi_model(tier)
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Rilevamento automatico del miglior formato supportato dall'hardware
    # bfloat16 o float16 garantiscono l'uso della versione NON quantizzata.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"\nCaricamento di Phi ({tier}) a precisione {dtype} (NON quantizzato) in corso...")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "_attn_implementation": "eager" # Necessario per Colab T4 senza Flash Attention
        # Nessun parametro 'quantization_config' presente: il modello è caricato al 100% nei pesi originali
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        **model_kwargs
    ).eval().cuda()
    
    return model, processor

# --- INFERENZA ---

def extract_table_phi(model, processor, image_path, prompt_text):
    image = Image.open(image_path).convert('RGB')
    
    messages = [
        {"role": "user", "content": f"<|image_1|>\n{prompt_text}"}
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt, 
        images=[image], 
        return_tensors="pt"
    ).to(model.device)

    generation_args = {
        "max_new_tokens": 15_000,
        "do_sample": False,
        "eos_token_id": processor.tokenizer.eos_token_id
    }

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            **generation_args
        )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    
    response = processor.batch_decode(
        generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    return response

def run_batch_inference(model, processor, tier="3.5-Vision"):
    input_base_dir = IMAGES_DIR
    output_base_dir = PREDICTIONS_DIR / f"Phi{tier}"
    
    if not input_base_dir.exists():
        print(f"Errore: La directory di input {input_base_dir} non esiste sul tuo Drive.")
        print("Crea la cartella e inserisci le immagini prima di procedere.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for img_path in input_base_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            
            relative_path = img_path.relative_to(input_base_dir)
            json_output_path = output_base_dir / relative_path.with_suffix('.json')
            
            if json_output_path.exists():
                print(f"Skip: {json_output_path.name} (già processato).")
                continue
            
            if len(relative_path.parts) >= 2:
                chart_class = relative_path.parts[1]
            else:
                print(f"Attenzione: Impossibile determinare la classe per {relative_path}. Salto.")
                continue
            
            try:
                prompt_scelto = PROMPT2CHARTCLASS.get(chart_class, PROMPT_AreaLineBarHistogram)
            except NameError:
                prompt_scelto = "Extract the data from this chart into a JSON format."
            
            print(f"Elaborazione: {relative_path} [Classe: {chart_class}]...")
            json_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                raw_result = extract_table_phi(model, processor, img_path, prompt_scelto)
                
                clean_json = raw_result.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

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

def ask_phi(tier="3.5-Vision"):
    modello, processore = setup_phi_vision(tier=tier)
    run_batch_inference(modello, processore, tier=tier)
    print(f"\nPhi {tier}: inferenza batch completata.")

if __name__ == "__main__":
    # Libera memoria residua prima di iniziare
    torch.cuda.empty_cache()
    gc.collect()
    ask_phi()