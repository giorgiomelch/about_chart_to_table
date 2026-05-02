#!/usr/bin/env python3
"""
Esegue predizioni Gemini Batch su tutte le immagini in una cartella.

Strategia a tre fasi:
  1. SUBMIT   — Per ogni sottocartella di primo livello di <input>, carica le
                immagini su Google Files API, costruisce il JSONL delle richieste
                e avvia il batch job. I job vengono avviati tutti prima di aspettare
                qualsiasi completamento.
  2. POLL     — Interroga tutti i job avviati finché ognuno non raggiunge uno
                stato finale (SUCCEEDED / FAILED / CANCELLED / EXPIRED).
  3. DOWNLOAD — Scarica i risultati dei job completati con successo.

Il file <output>/_state.json salva lo stato di ogni job: se il processo viene
interrotto, alla riesecuzione le fasi già completate vengono saltate.

Uso:
    python scripts/run_batch_predictions.py \\
        --input  /path/to/images \\
        --output /path/to/results \\
        [--model gemini-3-flash-preview] \\
        [--key-env GEMINI_API_KEY] \\
        [--poll-interval 60]

Esempio per PMCimages:
    python scripts/run_batch_predictions.py \\
        --input  /data/PMCimages \\
        --output /data/batch_results
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gemini_client import (
    DEFAULT_MODEL,
    DEFAULT_API_KEY_ENV_VAR,
    create_batch_file,
    submit_batch_job,
    download_batch_jsonl,
    get_client,
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_groups(root: Path) -> dict[str, Path]:
    """
    Restituisce {group_name: directory_path} per ogni gruppo di immagini.

    - Ogni sottocartella di primo livello che contiene immagini (a qualsiasi
      profondità) diventa un gruppo separato → un batch job.
    - Se root contiene immagini direttamente al primo livello, viene aggiunto
      un gruppo con il nome della directory radice stessa.
    """
    groups: dict[str, Path] = {}

    direct_images = [
        f for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]
    if direct_images:
        groups[root.name] = root

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        has_images = any(
            f.suffix.lower() in VALID_EXTENSIONS
            for f in subdir.rglob("*") if f.is_file()
        )
        if has_images:
            groups[subdir.name] = subdir

    return groups


def count_images(path: Path) -> int:
    return sum(
        1 for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    )


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def save_state(state: dict, state_path: Path) -> None:
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


# ── fasi principali ───────────────────────────────────────────────────────────

def phase_submit(
    groups: dict[str, Path],
    state: dict,
    output_dir: Path,
    work_dir: Path,
    model: str,
    key_env: str,
) -> None:
    """
    Per ogni gruppo non ancora avviato:
      1. Carica immagini su Google Files API e crea il JSONL richieste.
      2. Avvia il batch job e salva il job_id nello state file.
    """
    to_submit = {name: path for name, path in groups.items() if name not in state}
    if not to_submit:
        print("Fase SUBMIT: tutti i job già avviati in precedenza.")
        return

    print(f"\n=== Fase SUBMIT ({len(to_submit)} gruppi) ===")
    for group_name, group_dir in to_submit.items():
        requests_jsonl = work_dir / f"requests_{group_name}.jsonl"

        print(f"\n[{group_name}] Creazione JSONL richieste...")
        ok = create_batch_file(
            dataset_path=str(group_dir),
            jsonl_output_path=str(requests_jsonl),
            gemini_model=model,
            recursive=True,
        )
        if not ok:
            print(f"[{group_name}] Nessuna immagine trovata, salto.")
            continue

        print(f"[{group_name}] Invio batch job...")
        try:
            job_id = submit_batch_job(
                jsonl_input_path=str(requests_jsonl),
                display_name=f"batch_{group_name}",
                gemini_model=model,
                gemini_key=key_env,
            )
        except Exception as e:
            print(f"[{group_name}] Errore submit: {e}")
            continue

        print(f"[{group_name}] Job avviato: {job_id}")
        state[group_name] = {
            "job_id":      job_id,
            "status":      "JOB_STATE_PENDING",
            "result_path": str(output_dir / f"batch_result_{group_name}.jsonl"),
        }
        save_state(state, output_dir / "_state.json")


def phase_poll(
    state: dict,
    output_dir: Path,
    key_env: str,
    poll_interval: int,
) -> None:
    """
    Interroga tutti i job finché non raggiungono uno stato finale.
    Aggiorna lo state file ad ogni cambio di stato.
    """
    client = get_client(key_env)
    pending = {
        name: info for name, info in state.items()
        if info["status"] not in COMPLETED_STATES
    }

    if not pending:
        print("\nFase POLL: tutti i job già in stato finale.")
        return

    print(f"\n=== Fase POLL ({len(pending)} job attivi, intervallo: {poll_interval}s) ===")
    while pending:
        for name in list(pending):
            try:
                job = client.batches.get(name=pending[name]["job_id"])
                new_status = job.state.name
            except Exception as e:
                print(f"[{name}] Errore polling: {e}")
                continue

            if new_status != state[name]["status"]:
                print(f"[{name}] {state[name]['status']} → {new_status}")
                state[name]["status"] = new_status
                save_state(state, output_dir / "_state.json")

            if new_status in COMPLETED_STATES:
                del pending[name]

        if pending:
            print(f"  {len(pending)} job ancora in corso... (attendo {poll_interval}s)")
            time.sleep(poll_interval)

    print("Tutti i job hanno raggiunto uno stato finale.")


def phase_download(state: dict, output_dir: Path, key_env: str) -> None:
    """Scarica i risultati dei job completati con successo."""
    print("\n=== Fase DOWNLOAD ===")
    for name, info in state.items():
        result_path = info["result_path"]

        if Path(result_path).exists():
            print(f"[{name}] Già scaricato, salto.")
            continue

        if info["status"] != "JOB_STATE_SUCCEEDED":
            print(f"[{name}] Stato finale: {info['status']} — nessun file da scaricare.")
            continue

        print(f"[{name}] Download → {result_path}")
        try:
            download_batch_jsonl(
                job_name=info["job_id"],
                local_filename=result_path,
                gemini_key=key_env,
            )
        except Exception as e:
            print(f"[{name}] Errore download: {e}")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini Batch predictions ricorsive su una cartella.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",         required=True,
                        help="Cartella radice con le immagini")
    parser.add_argument("--output",        required=True,
                        help="Cartella output per i risultati JSONL")
    parser.add_argument("--model",         default=DEFAULT_MODEL,
                        help=f"Modello Gemini (default: {DEFAULT_MODEL})")
    parser.add_argument("--key-env",       default=DEFAULT_API_KEY_ENV_VAR,
                        help=f"Nome variabile d'ambiente API key (default: {DEFAULT_API_KEY_ENV_VAR})")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Secondi tra un poll e il successivo (default: 60)")
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    work_dir   = output_dir / "_work"
    state_path = output_dir / "_state.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    # Scoperta gruppi
    groups = find_groups(input_dir)
    if not groups:
        print("Nessuna immagine trovata nella cartella specificata.")
        sys.exit(1)

    print(f"Cartella input : {input_dir}")
    print(f"Cartella output: {output_dir}")
    print(f"Trovati {len(groups)} gruppi di immagini:")
    for name, path in groups.items():
        print(f"  {name:25s}  {count_images(path):6d} immagini")

    state = load_state(state_path)

    phase_submit(groups, state, output_dir, work_dir, args.model, args.key_env)
    phase_poll(state, output_dir, args.key_env, args.poll_interval)
    phase_download(state, output_dir, args.key_env)

    # Riepilogo finale
    succeeded = sum(1 for i in state.values() if i["status"] == "JOB_STATE_SUCCEEDED")
    failed    = sum(1 for i in state.values() if i["status"] != "JOB_STATE_SUCCEEDED")
    print(f"\n=== Riepilogo: {succeeded} job completati, {failed} falliti/annullati ===")
    if failed:
        for name, info in state.items():
            if info["status"] != "JOB_STATE_SUCCEEDED":
                print(f"  {name}: {info['status']}")


if __name__ == "__main__":
    main()
