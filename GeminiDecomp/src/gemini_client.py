import os
from google import genai
from google.genai import types
import json
import PIL.Image
import time
from pathlib import Path
from tqdm import tqdm
    
PROMPT_TEXT = """
Role:
You are an automated visual annotation system specialized in Compound Figure Separation for scientific and technical documents.
Task:
Detect and annotate every single subfigure contained in the image to enable precise statistical counting of figure types.

Definition of a Figure:
A figure is a visually bounded and semantically self-contained graphical region intended to convey information independently from the surrounding text or figures.
Figures include: charts, plots, tables, diagrams, schematics, flowcharts, microscopy images, maps, and illustrations.
Maximize granularity. If a figure is composed of multiple smaller plots, each plot must be detected separately.

Subfigure Definition and Separation Rules:
1. Grid and Matrix Plots (Aggressive Separation):
   - If a figure contains a grid of plots (e.g., scatter plot matrix, facet grid, distinct panels), you MUST create a separate bounding box for EACH individual plot cell.
   - Do this even if the internal plots lack their own axis labels or ticks.
   - Do this even if they are visually aligned or touching.
2. Insets (Nested Figures):
   - If a main figure contains a smaller figure (inset) embedded within it (e.g., a zoom-in graph inside a larger plot), create a separate bounding box for the inset AND a separate bounding box for the main plot.
3. Shared Elements and Legends:
   - Axis Labels: Always include axis labels (numbers and titles) that are immediately adjacent to the plot.
   - Legends:
     1. Integrated/Close Legends: If the legend is inside the plot area or immediately adjacent (touching/very close), INCLUDE it in the bounding box.
     2. Distant/Shared Legends: If a legend is positioned far away, centered between multiple subfigures, or if including it would require drawing a large box full of whitespace/overlapping other figures, EXCLUDE the legend.
     PRIORITY: It is better to have a tight box around the plot than a loose box containing a distant legend.

Non-segmentable Internal Elements:
General images, icons, thumbnails, or example illustrations that are embedded within a larger diagram, schematic, or flowchart must not be segmented.
EXCEPTION: If the embedded element is a data-driven "chart" (e.g., a small bar plot inside a flowchart node), you MUST segment it. Charts always take priority over this non-segmentation rule.

Decorative or Auxiliary Graphics:
Ignore logos, icons, arrows, callouts, or decorative elements unless they are essential to the interpretation of a figure. Do not create bounding boxes for purely illustrative or ornamental graphics.

Bounding Box Specification:
- Scope: Enclose the graphical content AND strictly associated local text (axis labels, ticks, local titles).
- Exclude surrounding whitespace where possible.
- Coordinates must be normalized to a 0–1000 scale (Origin: Top-Left).
- Format: [ymin, xmin, ymax, xmax]
- Use integer values rounded to the nearest integer.

Classification:
Assign exactly one of the following classes to each bounding box:
- "chart":
  Any data-driven graphical representation such as bar charts, line plots, scatter plots, histograms, box plots, pie charts, radar charts, heatmaps.
- "non_chart":
  Any figure that is not a numeric data plot, including tables, photographs, diagrams, flowcharts, maps, illustrations and schematics.

Output Format:
Return a single valid JSON array and nothing else.
If no figures are detected, return an empty array: []
Each element in the array must have the following structure:
[
  {
    "bbox": [ymin, xmin, ymax, xmax],
    "class": "chart" | "non_chart",
  }
]
"""

PROMPT_TEXT_OLD_3GEN = """
Role: You are an automated visual annotation system for Document Layout Analysis.

Task: Given a single document image, detect and annotate all visual figures that are not part of the continuous main body text.

Definition of a Figure:
A figure is any visually bounded, self-contained graphical region intended to convey information independently from the running text. This includes charts, plots, tables, diagrams, photographs, flowcharts, and schematics.
Exclude:
- Running body text
- Section headers
- Footnotes
- Page numbers
- Decorative separators
- Background watermarks

Captions:
If a figure caption is visually attached to a figure, EXCLUDE the caption area from the bounding box.

Detection Rules:
1. Detect every distinct figure present in the image.
2. If a figure contains multiple clearly separated sub-figures (e.g., labeled a, b, c), create one bounding box per sub-figure.
4. Ignore icons, logos, or decorative graphics unless they are part of a figure.

Bounding Box Specification:
- Use bounding boxes tightly enclosing the visible pixels of the figure only.
- Exclude surrounding whitespace where possible.
- Coordinates must be normalized to a 0–1000 scale.
- Format: [ymin, xmin, ymax, xmax]
- Use integer values rounded to the nearest integer.

Classification:
Assign exactly one of the following classes to each bounding box:

- "chart":
  Any data-driven graphical representation such as bar charts, line plots, scatter plots, histograms, box plots, pie charts, radar charts, heatmaps.

- "non_chart":
  Any figure that is not a data plot, including tables, photographs, diagrams, flowcharts, maps, illustrations, or schematics.

Confidence:
- Provide a confidence score between 0.0 and 1.0.
- The confidence represents certainty in both correct localization and correct classification.
- Use conservative estimates; avoid values above 0.95 unless the figure is unambiguous.

Output Format:
Return a single valid JSON array and nothing else.

If no figures are detected, return an empty array: []

Each element in the array must have the following structure:

[
  {
    "bbox": [ymin, xmin, ymax, xmax],
    "class": "chart" | "non_chart",
    "confidence": float
  }
]
"""

DEFAULT_MODEL = "gemini-3-flash-preview"

DEFAULT_API_KEY_ENV_VAR = "GEMINI_API_KEY" #per gemini free aggiungi _FREE



def get_client(gemini_key=DEFAULT_API_KEY_ENV_VAR):
    """Inizializza il client tramite variabili ambiente."""
    api_key = os.environ.get(gemini_key)
    if not api_key:
        raise EnvironmentError(f"Variabile {gemini_key} non impostata.")
    return genai.Client(api_key=api_key)

def get_mime_type(file_path):
    """Restituisce il mime type corretto per l'API di Gemini."""
    suffix = file_path.suffix.lower()
    if suffix in ['.jpg', '.jpeg']: return 'image/jpeg'
    if suffix == '.png': return 'image/png'
    if suffix == '.webp': return 'image/webp'
    return 'image/jpeg' # Fallback


def create_batch_file(dataset_path, jsonl_output_path, gemini_model=DEFAULT_MODEL, recursive=False):
    """
    1. Scansiona la cartella (ricorsivamente se recursive=True).
    2. Carica ogni immagine su Google File API (con progress bar).
    3. Crea il file JSONL con i riferimenti URI.
    """
    client = get_client()
    dataset_dir = Path(dataset_path)

    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    scanner = dataset_dir.rglob("*") if recursive else dataset_dir.glob("*")
    image_files = [f for f in scanner if f.is_file() and f.suffix.lower() in valid_extensions]
    if not image_files:
        print(f"Nessuna immagine trovata in {dataset_path}")
        return False
    print(f"Rilevate {len(image_files)} immagini. Inizio caricamento su File API...")
    
    requests = []

    with tqdm(total=len(image_files), desc="Caricamento Cloud", unit="img") as pbar:
        for img_path in image_files:
            try:
                # A. Caricamento dell'immagine sul server Gemini
                uploaded_file = client.files.upload(file=str(img_path))
                # B. Creazione dell'oggetto richiesta
                mime_type = get_mime_type(img_path)
                request_entry = {
                    "key": f"req_{img_path.stem}",
                    "request": {
                        "model": f"models/{gemini_model}",
                        "contents": [{
                            "parts": [
                                {"text": PROMPT_TEXT},
                                {"fileData": {
                                    "mimeType": mime_type,
                                    "fileUri": uploaded_file.uri
                                }}
                            ]
                        }],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0
                        }
                    }
                }
                requests.append(request_entry)

            except Exception as e:
                tqdm.write(f"Errore su file {img_path.name}: {e}")
            pbar.update(1)

    # C. Salvataggio del file JSONL
    if requests:
        with open(jsonl_output_path, 'w', encoding='utf-8') as f:
            for entry in requests:
                f.write(json.dumps(entry) + '\n')
        print(f"\nFile Batch generato con successo: {jsonl_output_path} ({len(requests)} richieste pronte)")
        return True
    else:
        print("\nNessuna richiesta generata. Controlla gli errori.")
        return False


def download_batch_jsonl(job_name, local_filename="results_full.jsonl", gemini_key=DEFAULT_API_KEY_ENV_VAR):
    client = get_client(gemini_key)
    
    print(f"Recupero stato del job: {job_name}")
    try:
        batch_job = client.batches.get(name=job_name)
    except Exception as e:
        print(f"Errore API durante il recupero del job: {e}")
        return

    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Impossibile scaricare. Il job non è completato. Stato: {batch_job.state.name}")
        if hasattr(batch_job, 'error'):
            print(f"Errore riportato: {batch_job.error}")
        return

    if hasattr(batch_job, 'dest') and batch_job.dest and batch_job.dest.file_name:
        remote_file_name = batch_job.dest.file_name
        print(f"File remoto individuato: {remote_file_name}")

        try:
            print("Inizio download...")
            file_content = client.files.download(file=remote_file_name)
            
            with open(local_filename, 'wb') as f_out:
                f_out.write(file_content)
            
            print(f"Salvataggio completato. File disponibile in: {os.path.abspath(local_filename)}")
            
        except Exception as e:
            print(f"Errore critico durante il download o la scrittura: {e}")

    elif hasattr(batch_job, 'dest') and batch_job.dest and batch_job.dest.inlined_responses:
        print("Attenzione: I risultati sono 'inline' e non in un file. Questa funzione non supporta il salvataggio inline.")
    
    else:
        print("Errore: Il job risulta completato, ma non è stato trovato alcun output (né file, né inline).")


def submit_batch_job(jsonl_input_path, display_name="Medical_Labeling_Batch", gemini_model=DEFAULT_MODEL, gemini_key=DEFAULT_API_KEY_ENV_VAR) -> str:
    """
    Carica il JSONL e avvia il batch job senza attendere il completamento.

    Returns:
        job_name (str) — identificatore del job da usare per polling e download.
    """
    client = get_client(gemini_key)
    print(f"Caricamento file di definizioni batch: {jsonl_input_path}...")
    batch_resource_file = client.files.upload(
        file=jsonl_input_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type='application/jsonl')
    )
    job = client.batches.create(
        model=gemini_model,
        src=batch_resource_file.name,
        config={'display_name': display_name}
    )
    return job.name


def run_batch_job(jsonl_input_path, final_output_path, gemini_model=DEFAULT_MODEL, gemini_key=DEFAULT_API_KEY_ENV_VAR):
    """
    1. Carica il file JSONL.
    2. Avvia il Job.
    3. Monitora lo stato.
    4. Scarica i risultati.
    """
    client = get_client(gemini_key)
    display_name = "Medical_Labeling_Batch"

    print(f"Caricamento file di definizioni batch: {jsonl_input_path}...")
    try:
        batch_resource_file = client.files.upload(
            file=jsonl_input_path,
            config=types.UploadFileConfig(display_name=display_name, mime_type='application/jsonl')
            )
    except Exception as e:
        print(f"Errore critico upload JSONL: {e}")
        return
    
    print("Invio richiesta Batch Job...")
    try:
        job = client.batches.create(
            model=gemini_model,
            src=batch_resource_file.name,
            config={'display_name':display_name}
        )
    except Exception as e:
        print(f"Errore avvio job: {e}")
        return
    
    job_id = job.name
    print(f"Job creato: {job_id}")

    completed_states = set(['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'])

    print(f"Polling status for job: {job_id}")
    batch_job = client.batches.get(name=job_id)
    prev_status = None
    while True:
        batch_job = client.batches.get(name=job_id)
        current_state = batch_job.state.name
        if not current_state == prev_status:
            print(f"Job status: {current_state}")
            prev_status = current_state

        if current_state in completed_states:
            break
        time.sleep(60)

    if current_state == 'JOB_STATE_SUCCEEDED':
        print("Job completato.")
        download_batch_jsonl(job_id, final_output_path, gemini_key)
    else:
        print("Il job è fallito.")
        if hasattr(batch_job, 'error'):
            print(f"Errore API: {batch_job.error}")



def analyze_image(image_path, output_json_path, model=DEFAULT_MODEL, gemini_key=DEFAULT_API_KEY_ENV_VAR):
    try:
        img = PIL.Image.open(image_path)
    except Exception as e:
        print(f"Errore nell'apertura dell'immagine: {e}")
        return
    
    #carica l'api key dalle variabili d'ambiente
    client = get_client(gemini_key)

    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT_TEXT, img],
            config=config,
        )

        extracted_data = response.parsed if response.parsed else json.loads(response.text)

        # Creazione della struttura finale del file
        final_output = {
            "source_file": os.path.basename(image_path),
            "model_used": DEFAULT_MODEL,
            "analysis": extracted_data
        }
        
        # Salvataggio in un nuovo file JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print(f"Processo completato. File salvato in: {output_json_path}")

    except Exception as e:
        print(f"Errore durante la generazione o il salvataggio: {e}")

