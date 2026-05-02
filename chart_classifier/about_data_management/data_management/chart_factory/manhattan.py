"""
Manhattan Plot Synthetic Generator
===================================
Genera migliaia di Manhattan Plot sintetici per dataset di classificazione.
Ottimizzato per stabilità e qualità su grandi batch.

Uso:
    python manhattan_plot_generator.py --num 100 --output ./output
    python manhattan_plot_generator.py --num 4000 --output ./manhattan_plots --workers 4
"""

import os
import sys
import argparse
import random
import gc
import logging
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo: niente finestre, niente crash
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  GENERATORI DI STRINGHE COMPLETAMENTE RANDOM
# ─────────────────────────────────────────────
import string as _string

# Alfabeti disponibili per la generazione
_UPPER   = _string.ascii_uppercase          # A-Z
_LOWER   = _string.ascii_lowercase          # a-z
_DIGITS  = _string.digits                   # 0-9
_SPECIAL = "-_.:/()"                        # caratteri speciali ammessi
_ALL     = _UPPER + _LOWER + _DIGITS + _SPECIAL

# Caratteri ammessi per le etichette dell'asse X (più leggibili)
_XLABEL_CHARS = _UPPER + _LOWER + _DIGITS + "-_."


def _rand_word(rng, min_len=2, max_len=12, alphabet=None):
    """Genera una parola di lunghezza casuale in [min_len, max_len]."""
    if alphabet is None:
        alphabet = _ALL
    length = rng.randint(min_len, max_len + 1)
    return ''.join(rng.choices(alphabet, k=length))


def _rand_words(rng, min_words=1, max_words=5, min_len=2, max_len=10, alphabet=None):
    """Genera una sequenza di parole casuali separate da spazio."""
    n = rng.randint(min_words, max_words + 1)
    return ' '.join(_rand_word(rng, min_len, max_len, alphabet) for _ in range(n))


def _rand_title(rng):
    """
    Titolo casuale: lunghezza variabile, struttura sintattica libera.
    Può avere separatori come –, :, (, ), numeri, ecc.
    """
    separators = [' – ', ': ', ' | ', ' / ', ' > ', ' ']
    parts = []
    n_parts = rng.randint(1, 5)
    for _ in range(n_parts):
        part_type = rng.choice(['word', 'code', 'number', 'mixed'])
        if part_type == 'word':
            parts.append(_rand_words(rng, 1, 4, 2, 12, _UPPER + _LOWER))
        elif part_type == 'code':
            parts.append(_rand_word(rng, 2, 8, _UPPER + _DIGITS + '-_'))
        elif part_type == 'number':
            parts.append(str(rng.randint(1, 9999)))
        else:
            parts.append(_rand_word(rng, 3, 14, _ALL))
    sep = rng.choice(separators)
    return sep.join(parts)


def _rand_axis_label(rng):
    """Label asse: stringa leggibile di lunghezza variabile."""
    styles = [
        lambda: _rand_words(rng, 1, 4, 3, 10, _UPPER + _LOWER + _DIGITS),
        lambda: _rand_word(rng, 4, 20, _ALL),
        lambda: f"{_rand_word(rng, 2, 8, _UPPER + _LOWER)} ({_rand_word(rng, 1, 4, _UPPER + _DIGITS)})",
        lambda: ' '.join([_rand_word(rng, 2, 8, _UPPER + _LOWER + _DIGITS),
                            _rand_word(rng, 1, 5, _DIGITS + '.'),
                            _rand_word(rng, 1, 3, _UPPER)]),
    ]
    return rng.choice(styles)()


def _rand_xlabel_tick(rng):
    """
    Etichetta singola per un tick dell'asse X.
    Lunghezza >= 1, caratteri alfanumerici + separatori.
    Nessun limite superiore fisso: può arrivare a 15+ caratteri.
    """
    styles = [
        # Stile numerico puro
        lambda: str(rng.randint(1, 9999)),
        # Stile alfanumerico corto
        lambda: _rand_word(rng, 1, 4, _UPPER + _DIGITS),
        # Stile lungo tipo codice
        lambda: _rand_word(rng, 4, 12, _XLABEL_CHARS),
        # Stile con separatore interno
        lambda: f"{_rand_word(rng, 1, 5, _UPPER + _LOWER)}{rng.choice('-_.')}{_rand_word(rng, 1, 6, _DIGITS + _UPPER)}",
        # Stile numeri romani simulati (costruiti da lettere romane)
        lambda: ''.join(rng.choices('IVXLCDM', k=rng.randint(1, 6))),
        # Stile misto lungo
        lambda: _rand_word(rng, 5, 15, _XLABEL_CHARS),
        # Stile con prefisso random + numero
        lambda: f"{_rand_word(rng, 1, 5, _UPPER + _LOWER)}{rng.randint(1, 999)}",
        # Solo lettere uppercase
        lambda: _rand_word(rng, 1, 6, _UPPER),
        # Codice con punto/slash
        lambda: f"{_rand_word(rng, 2, 6, _UPPER)}.{rng.randint(1, 99)}",
    ]
    return rng.choice(styles)()


def _rand_ylabel(rng):
    """Label asse Y: può essere testo puro o stringa LaTeX simulata."""
    if rng.random() < 0.3:
        # LaTeX-like
        inner = _rand_word(rng, 2, 8, _LOWER + _DIGITS + '-')
        return rf"$-\log_{{10}}(\mathrm{{{inner}}})$"
    else:
        return _rand_axis_label(rng)

# ─────────────────────────────────────────────
#  PALETTE COLORI
# ─────────────────────────────────────────────
COLOR_SCHEMES = [
    # (colore_pari, colore_dispari)
    ("#4878CF", "#6ACC65"),   # blu / verde
    ("#333333", "#6baed6"),   # grigio scuro / azzurro
    ("#888888", "#4292c6"),   # grigio / blu medio
    ("#D62728", "#1F77B4"),   # rosso / blu
    ("#7B4173", "#A55194"),   # viola scuro / viola chiaro
    ("#3182bd", "#9ecae1"),   # blu / blu chiaro
    ("#636363", "#969696"),   # grigio scuro / grigio chiaro
    ("#74C476", "#31A354"),   # verde chiaro / verde scuro
    ("#FD8D3C", "#E6550D"),   # arancio chiaro / arancio scuro
    ("#9467BD", "#C5B0D5"),   # viola / lavanda
]

THRESHOLD_COLORS = ["#D62728", "#888888", "#E31A1C", "#6A3D9A", "#FF7F00"]


# ─────────────────────────────────────────────
#  FUNZIONE PRINCIPALE DI GENERAZIONE
# ─────────────────────────────────────────────
def generate_manhattan_plot(output_path: Path, seed: int = None):
    """
    Genera un singolo Manhattan Plot sintetico e lo salva.
    Ritorna True se successo, False se errore.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    try:
        # ── Parametri globali ──────────────────────────────────────────
        n_blocks = py_rng.randint(6, 29)            # 6-28 blocchi sull'asse X
        dpi = py_rng.choice([72, 96, 120, 150, 200, 300])
        fig_w = py_rng.uniform(12, 20)
        fig_h = py_rng.uniform(4, 8)

        # Background: 80% white/light, 20% random color
        use_random_bg = py_rng.random() < 0.20
        if use_random_bg:
            bg_r = py_rng.randint(180, 241) / 255
            bg_g = py_rng.randint(180, 241) / 255
            bg_b = py_rng.randint(180, 241) / 255
            bg_color = (bg_r, bg_g, bg_b)
        else:
            v = py_rng.randint(240, 256) / 256   # max 255/256 < 1.0
            bg_color = (v, v, v)

        # ── Colori blocchi ─────────────────────────────────────────────
        c1, c2 = py_rng.choice(COLOR_SCHEMES)
        # Opzionalmente terzo colore
        use_third_color = py_rng.random() < 0.30
        c3 = None
        if use_third_color:
            c3 = py_rng.choice([s[0] for s in COLOR_SCHEMES if s[0] not in (c1, c2)])
        block_colors = [c1, c2] if not use_third_color else [c1, c2, c3]

        # ── Strategia globale del rumore di fondo ─────────────────────
        # Scelta di un "regime" dominante per l'intero grafico, poi ogni
        # blocco può deviare localmente per aumentare la variabilità.
        global_noise_regime = py_rng.choice([
            "flat",          # rumore compresso in basso, plateau quasi piatto
            "exponential",   # decadimento esponenziale, pochi valori alti
            "bimodal",       # due fasce di densità separate
            "gradient",      # il rumore sale/scende lungo il cromosoma
            "banded",        # bande orizzontali di densità alternata
            "heavy_tail",    # distribuzione con coda pesante (Pareto/Cauchy)
            "spiky_noise",   # molti mini-picchi nel rumore (non picchi GWAS)
            "uniform_low",   # quasi tutto compresso sotto una soglia bassa
            "chi2",          # distribuzione chi-quadro (realistica per p-values)
            "layered",       # più strati sovrapposti a densità diverse
        ])

        # Parametri globali del rumore condivisi tra i blocchi
        global_noise_scale = py_rng.uniform(1.5, 7.0)   # altezza massima rumore
        global_noise_floor = py_rng.uniform(0.0, 1.5)   # pavimento minimo

        # ── Generazione dati per blocco ────────────────────────────────
        all_x, all_y, all_colors = [], [], []
        block_centers = []
        current_x = 0.0
        x_gap = py_rng.uniform(0.3, 0.8)

        y_max_data = 0.0

        for b in range(n_blocks):
            block_color = block_colors[b % len(block_colors)]
            block_width = py_rng.uniform(0.8, 2.5)
            n_points = py_rng.randint(800, 4000)
            x_pos = rng.uniform(current_x, current_x + block_width, n_points)

            # ── Rumore di fondo: regime variabile per blocco ───────────
            # Ogni blocco ha 25% di probabilità di usare un regime locale
            # diverso da quello globale, per simulare eterogeneità cromosomica.
            if py_rng.random() < 0.25:
                local_regime = py_rng.choice([
                    "flat", "exponential", "bimodal", "gradient",
                    "heavy_tail", "spiky_noise", "chi2", "uniform_low",
                ])
            else:
                local_regime = global_noise_regime

            ns = global_noise_scale
            nf = global_noise_floor

            if local_regime == "flat":
                # Schiacciato in basso con piccola varianza
                base_noise = rng.beta(0.3, 8, n_points) * ns * 0.5

            elif local_regime == "exponential":
                # Tipico del -log10(p) reale: esponenziale con lambda alto
                lam = py_rng.uniform(0.5, 2.0)
                base_noise = rng.exponential(1.0 / lam, n_points)
                base_noise = np.clip(base_noise, 0, ns)

            elif local_regime == "bimodal":
                # Due gaussiane: una bassa (80%) e una media (20%)
                n_low  = int(n_points * py_rng.uniform(0.65, 0.85))
                n_mid  = n_points - n_low
                low_cluster  = rng.normal(nf + ns * 0.15, ns * 0.08, n_low)
                mid_cluster  = rng.normal(nf + ns * 0.55, ns * 0.12, n_mid)
                base_noise = np.concatenate([low_cluster, mid_cluster])
                base_noise = np.clip(base_noise, 0, ns)

            elif local_regime == "gradient":
                # Il livello medio del rumore sale o scende lungo il blocco
                direction = py_rng.choice([-1, 1])
                t = np.linspace(0, 1, n_points)
                trend = direction * t * ns * py_rng.uniform(0.3, 0.8)
                base_noise = rng.beta(0.4, 5, n_points) * ns * 0.5 + trend
                base_noise = np.abs(base_noise)
                base_noise = np.clip(base_noise, 0, ns * 1.2)

            elif local_regime == "banded":
                # Bande orizzontali: i punti si raggruppano in fasce di y
                n_bands = py_rng.randint(2, 5)
                band_centers = rng.uniform(nf, ns, n_bands)
                chosen_bands = rng.integers(0, n_bands, n_points)
                base_noise = band_centers[chosen_bands] + rng.normal(0, ns * 0.05, n_points)
                base_noise = np.clip(base_noise, 0, ns)

            elif local_regime == "heavy_tail":
                # Pareto: molti punti bassi, pochi molto alti (non picchi GWAS)
                alpha = py_rng.uniform(1.5, 4.0)
                base_noise = (rng.pareto(alpha, n_points) + 1) * py_rng.uniform(0.3, 1.5)
                base_noise = np.clip(base_noise, 0, ns * 1.5)

            elif local_regime == "spiky_noise":
                # Rumore con tanti mini-picchi casuali distribuiti sul blocco
                base_noise = rng.beta(0.5, 6, n_points) * ns * 0.4
                n_mini = py_rng.randint(10, 40)
                for _ in range(n_mini):
                    cx = py_rng.uniform(current_x, current_x + block_width)
                    mask = np.abs(x_pos - cx) < py_rng.uniform(0.005, 0.04)
                    base_noise[mask] += rng.uniform(ns * 0.3, ns * 0.9, mask.sum())
                base_noise = np.clip(base_noise, 0, ns)

            elif local_regime == "chi2":
                # Chi-quadro con df bassi: realistico per test statistici
                df = py_rng.uniform(1.0, 4.0)
                base_noise = rng.chisquare(df, n_points) * py_rng.uniform(0.3, 1.2)
                base_noise = np.clip(base_noise, 0, ns * 1.5)

            elif local_regime == "uniform_low":
                # Quasi tutto compressa in una fascia molto bassa
                low_b = max(0.0, nf)
                high_b = low_b + max(0.1, ns * py_rng.uniform(0.10, 0.35))
                base_noise = rng.uniform(low_b, high_b, n_points)

            elif local_regime == "layered":
                # Tre strati sovrapposti a diversa densità
                w1 = py_rng.uniform(0.50, 0.68)
                w2 = py_rng.uniform(0.15, min(0.30, 0.95 - w1))
                n1 = int(n_points * w1)
                n2 = int(n_points * w2)
                n3 = max(0, n_points - n1 - n2)
                l1 = rng.beta(0.3, 8, n1) * ns * 0.3 + nf
                l2 = rng.beta(1.0, 3, n2) * ns * 0.5 + nf + ns * 0.1
                lo3 = nf + ns * 0.4
                hi3 = max(lo3 + 0.5, nf + ns * 0.8)
                l3 = rng.uniform(lo3, hi3, n3)
                base_noise = np.concatenate([l1, l2, l3])
                # Assicura lunghezza corretta
                if len(base_noise) != n_points:
                    base_noise = np.resize(base_noise, n_points)
                base_noise = np.clip(base_noise, 0, ns)

            else:
                # Fallback sicuro
                base_noise = rng.beta(0.5, 5, n_points) * ns

            # Aggiungi il pavimento e un piccolo jitter uniforme
            base_noise = base_noise + nf + rng.uniform(0, 0.2, len(base_noise))
            base_noise = np.clip(base_noise, 0, None)

            # Permuta casualmente per non avere gradiente artificiale sull'asse x
            rng.shuffle(base_noise)

            block_y = base_noise

            # Picchi isolati: 1-6 per blocco
            n_peaks = py_rng.randint(1, 7)
            for _ in range(n_peaks):
                peak_x_center = py_rng.uniform(current_x + 0.1, current_x + block_width - 0.1)
                peak_height = py_rng.uniform(6, 30)
                peak_spread = py_rng.uniform(0.01, 0.06)
                n_peak_pts = py_rng.randint(3, 25)
                px = rng.normal(peak_x_center, peak_spread, n_peak_pts)
                py_vals = peak_height + rng.exponential(2, n_peak_pts)
                # Clamp
                px = np.clip(px, current_x, current_x + block_width)
                all_x.extend(px.tolist())
                all_y.extend(py_vals.tolist())
                all_colors.extend([block_color] * n_peak_pts)
                y_max_data = max(y_max_data, py_vals.max())

            all_x.extend(x_pos.tolist())
            all_y.extend(block_y.tolist())
            all_colors.extend([block_color] * n_points)

            block_centers.append(current_x + block_width / 2)
            current_x += block_width + x_gap

        # ── Etichette asse X: completamente random ────────────────────
        # Ogni etichetta è generata indipendentemente, lunghezza e caratteri liberi.
        # Opzionalmente si può usare un "tema" condiviso per il grafico (coerenza visiva)
        # ma con variazioni di lunghezza e contenuto libere.
        tick_theme = py_rng.choice([
            "free",       # ogni etichetta completamente indipendente
            "prefixed",   # stesso prefisso random + suffisso random
            "numeric",    # numeri random (non necessariamente sequenziali)
            "coded",      # codici tipo "AB-123" di lunghezza variabile
        ])

        if tick_theme == "free":
            x_labels = [_rand_xlabel_tick(py_rng) for _ in range(n_blocks)]

        elif tick_theme == "prefixed":
            prefix = _rand_word(py_rng, 1, 5, _UPPER + _LOWER)
            sep = py_rng.choice(["-", "_", ".", "", "/"])
            x_labels = [
                f"{prefix}{sep}{_rand_word(py_rng, 1, 6, _DIGITS + _UPPER + _LOWER)}"
                for _ in range(n_blocks)
            ]

        elif tick_theme == "numeric":
            # Numeri random non ordinati, eventualmente con decimali
            if py_rng.random() < 0.4:
                x_labels = [f"{py_rng.uniform(0, 9999):.{py_rng.randint(0,2)}f}"
                             for _ in range(n_blocks)]
            else:
                pool = py_rng.sample(range(1, n_blocks * 5), n_blocks)
                x_labels = [str(v) for v in pool]

        else:  # coded
            sep = py_rng.choice(["-", ".", "_", ""])
            x_labels = [
                f"{_rand_word(py_rng, 1, 4, _UPPER)}{sep}{_rand_word(py_rng, 1, 5, _DIGITS)}"
                for _ in range(n_blocks)
            ]

        # ── Marker size ────────────────────────────────────────────────
        marker_size = py_rng.choice([1, 1, 1, 2, 3, 5])

        # ── Soglie ────────────────────────────────────────────────────
        n_thresholds = py_rng.randint(1, 3)
        threshold_vals = sorted(
            [py_rng.uniform(5, 9), py_rng.uniform(3, 5)][:n_thresholds],
            reverse=True
        )
        thresh_colors = py_rng.sample(THRESHOLD_COLORS, min(n_thresholds, len(THRESHOLD_COLORS)))
        thresh_styles = py_rng.choices(["--", "-.", ":"], k=n_thresholds)
        thresh_widths = [py_rng.uniform(0.8, 2.0) for _ in range(n_thresholds)]

        # ── Titolo e label: completamente random ─────────────────────
        title  = _rand_title(py_rng)
        xlabel = _rand_axis_label(py_rng)
        ylabel = _rand_ylabel(py_rng)

        # ── Font sizes variabili ──────────────────────────────────────
        title_fs = py_rng.uniform(10, 16)
        label_fs = py_rng.uniform(8, 13)
        tick_fs = py_rng.uniform(6, 10)

        # ── PLOT ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Scatter unico (molto più veloce di loop)
        ax.scatter(
            all_x, all_y,
            c=all_colors,
            s=marker_size,
            marker='o',
            linewidths=0,
            alpha=py_rng.uniform(0.6, 1.0),
            zorder=2,
            rasterized=True    # rasterize per velocità/memoria su DPI alto
        )

        # Linee soglia
        for tv, tc, ts, tw in zip(threshold_vals, thresh_colors, thresh_styles, thresh_widths):
            ax.axhline(
                y=tv,
                color=tc,
                linestyle=ts,
                linewidth=tw,
                zorder=3,
                label=_rand_word(py_rng, 2, 8, _UPPER + _LOWER) + ": " + _rand_word(py_rng, 1, 5, _DIGITS + ".")
            )

        # Asse X: etichette centrali per ogni blocco
        ax.set_xticks(block_centers)
        ax.set_xticklabels(x_labels, fontsize=tick_fs, rotation=py_rng.choice([0, 45, 90]))

        # Asse Y
        y_upper = max(y_max_data * 1.1, max(threshold_vals) * 1.3, 10)
        ax.set_ylim(0, y_upper)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.tick_params(axis='y', labelsize=tick_fs)

        # Asse X range
        ax.set_xlim(-x_gap, current_x)

        # Titolo e label
        ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, fontsize=label_fs, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=6)

        # Legenda soglie (opzionale, 50% prob)
        if py_rng.random() < 0.5 and n_thresholds > 0:
            ax.legend(
                fontsize=max(6, label_fs - 2),
                loc=py_rng.choice(["upper right", "upper left", "lower right"]),
                framealpha=0.7,
                edgecolor='gray'
            )

        # Griglia leggera (30% prob)
        if py_rng.random() < 0.30:
            ax.yaxis.grid(True, linestyle=':', linewidth=0.4, color='gray', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

        # Spine
        for spine in ax.spines.values():
            spine.set_linewidth(py_rng.uniform(0.5, 1.5))

        plt.tight_layout(pad=py_rng.uniform(0.3, 1.2))
        fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight', facecolor=bg_color)

        return True

    except Exception:
        log.error(f"Errore generando {output_path}:\n{traceback.format_exc()}")
        return False
    finally:
        # CRITICO: chiude la figura e libera memoria
        plt.close('all')
        gc.collect()


# ─────────────────────────────────────────────
#  BATCH GENERATOR
# ─────────────────────────────────────────────
def generate_batch(
    num_plots: int,
    output_dir: Path,
    prefix: str = "manhattan",
    fmt: str = "png",
    log_every: int = 50,
    workers: int = 1,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    base_seed = random.randint(0, 10**9)
    tasks = [
        (output_dir / f"{prefix}_{i:05d}.{fmt}", base_seed + i)
        for i in range(num_plots)
    ]

    success = 0
    failed = 0

    if workers > 1:
        # Multiprocessing per batch grandi
        from multiprocessing import Pool
        log.info(f"Avvio generazione con {workers} worker paralleli...")

        def _worker(args):
            path, seed = args
            return generate_manhattan_plot(path, seed)

        with Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
                if result:
                    success += 1
                else:
                    failed += 1
                if (i + 1) % log_every == 0 or (i + 1) == num_plots:
                    log.info(f"  Completati: {i+1}/{num_plots} | OK: {success} | Errori: {failed}")
    else:
        log.info(f"Avvio generazione sequenziale di {num_plots} grafici...")
        for i, (path, seed) in enumerate(tasks):
            result = generate_manhattan_plot(path, seed)
            if result:
                success += 1
            else:
                failed += 1
            if (i + 1) % log_every == 0 or (i + 1) == num_plots:
                log.info(f"  Completati: {i+1}/{num_plots} | OK: {success} | Errori: {failed}")

    log.info(f"\n{'='*50}")
    log.info(f"BATCH COMPLETATO")
    log.info(f"  Totale richiesti : {num_plots}")
    log.info(f"  Salvati con successo: {success}")
    log.info(f"  Falliti           : {failed}")
    log.info(f"  Directory output  : {output_dir.resolve()}")
    log.info(f"{'='*50}")

    return success, failed


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generatore batch di Manhattan Plot sintetici.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Genera 100 plot nella cartella corrente
  python manhattan_plot_generator.py --num 100

  # Genera 4000 plot in alta qualità con 4 CPU
  python manhattan_plot_generator.py --num 4000 --output ./plots --workers 4

  # Genera con prefisso e formato JPEG
  python manhattan_plot_generator.py --num 500 --prefix gwas --fmt jpg

  # Singolo plot per test
  python manhattan_plot_generator.py --num 1 --output ./test
        """
    )
    parser.add_argument("--num", type=int, default=10,
                        help="Numero di grafici da generare (default: 10)")
    parser.add_argument("--output", type=str, default="./manhattan_plots",
                        help="Directory di output (default: ./manhattan_plots)")
    parser.add_argument("--prefix", type=str, default="manhattan",
                        help="Prefisso per i nomi dei file (default: manhattan)")
    parser.add_argument("--fmt", type=str, default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"],
                        help="Formato immagine (default: png)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Numero di processi paralleli (default: 1, usa 4-8 per batch grandi)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log progress ogni N grafici (default: 50)")

    args = parser.parse_args()

    log.info("Manhattan Plot Synthetic Generator")
    log.info(f"  Grafici da generare: {args.num}")
    log.info(f"  Output dir        : {args.output}")
    log.info(f"  Formato           : {args.fmt}")
    log.info(f"  Workers           : {args.workers}")
    log.info("")

    generate_batch(
        num_plots=args.num,
        output_dir=Path(args.output),
        prefix=args.prefix,
        fmt=args.fmt,
        log_every=args.log_every,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()