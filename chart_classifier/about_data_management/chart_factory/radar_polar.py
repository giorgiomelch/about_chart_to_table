"""
Generatore Ottimizzato di Radar e Polar Charts per Dataset AI
Genera migliaia di grafici radar e polari con variazioni casuali
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import string
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Configurazione
OUTPUT_DIR = Path("radar_polar_charts_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Caratteri per testo random (inclusi caratteri accentati italiani)
CHARS = string.ascii_letters + string.digits + 'àèéìòùÀÈÉÌÒÙ'

# Tipi di grafici
CHART_TYPES = [
    {'name': 'radar', 'type': 'radar'},
    {'name': 'radar_filled', 'type': 'radar', 'fill': True},
    {'name': 'polar_scatter', 'type': 'polar_scatter'},
    {'name': 'polar_line', 'type': 'polar_line'},
    {'name': 'polar_bar', 'type': 'polar_bar'},
    {'name': 'polar_area', 'type': 'polar_area'},
    {'name': 'radar_multi', 'type': 'radar_multi'},  # Multiple series
    {'name': 'polar_line_multi', 'type': 'polar_line_multi'},
]


def generate_random_text(min_len=4, max_len=12):
    """Genera testo casuale con lunghezza variabile"""
    length = random.randint(min_len, max_len)
    return ''.join(random.choice(CHARS) for _ in range(length))


def generate_random_colors(n):
    """Genera n colori casuali"""
    return [
        (random.random(), random.random(), random.random())
        for _ in range(n)
    ]


def generate_background_color():
    """Genera un colore di sfondo con alta probabilità di bianco/chiaro"""
    rand = random.random()
    
    # 60% bianco puro
    if rand < 0.6:
        return 'white'
    
    # 20% colori molto chiari (quasi bianchi)
    elif rand < 0.8:
        r = random.randint(240, 255)
        g = random.randint(240, 255)
        b = random.randint(240, 255)
        return (r/255, g/255, b/255)
    
    # 15% colori chiari pastello
    elif rand < 0.95:
        r = random.randint(200, 240)
        g = random.randint(200, 240)
        b = random.randint(200, 240)
        return (r/255, g/255, b/255)
    
    # 5% colori casuali
    else:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r/255, g/255, b/255)


def create_radar_chart(ax, labels, values, color, fill=False, alpha=0.25, linewidth=2):
    """Crea un grafico radar"""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Chiudi il cerchio
    values = values + values[:1]
    angles = angles + angles[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=linewidth, color=color, label=generate_random_text(5, 10))
    if fill:
        ax.fill(angles, values, alpha=alpha, color=color)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=random.randint(8, 12))
    
    # Grid
    ax.grid(True, linestyle='--', alpha=random.uniform(0.3, 0.7))
    
    return ax


def generate_single_chart(args):
    """Genera un singolo grafico - funzione per multiprocessing"""
    chart_id, total = args
    
    try:
        # Parametri casuali
        num_points = random.randint(3, 12)
        chart_type = random.choice(CHART_TYPES)
        
        # Genera dati
        labels = [generate_random_text() for _ in range(num_points)]
        
        # Background casuale
        bg_color = generate_background_color()
        fig = plt.figure(figsize=(8, 8), dpi=100, facecolor=bg_color)
        
        if chart_type['type'] in ['radar', 'radar_filled', 'radar_multi']:
            # RADAR CHART
            ax = fig.add_subplot(111, projection='polar')
            ax.set_facecolor(bg_color)
            
            if chart_type['type'] == 'radar_multi':
                # Multiple series
                num_series = random.randint(2, 4)
                for i in range(num_series):
                    values = [random.uniform(0, 100) for _ in range(num_points)]
                    color = generate_random_colors(1)[0]
                    fill = random.random() > 0.5
                    create_radar_chart(ax, labels, values, color, fill, 
                                     alpha=random.uniform(0.1, 0.3),
                                     linewidth=random.randint(1, 3))
            else:
                values = [random.uniform(0, 100) for _ in range(num_points)]
                color = generate_random_colors(1)[0]
                fill = chart_type.get('fill', False) or random.random() > 0.5
                create_radar_chart(ax, labels, values, color, fill,
                                 alpha=random.uniform(0.15, 0.4),
                                 linewidth=random.randint(2, 4))
            
            # Y-axis
            max_val = random.choice([50, 100, 150, 200])
            ax.set_ylim(0, max_val)
            if random.random() > 0.3:
                ax.set_yticks(np.linspace(0, max_val, random.randint(4, 7)))
                ax.set_yticklabels([str(int(x)) for x in np.linspace(0, max_val, random.randint(4, 7))], 
                                   size=random.randint(8, 11))
            
        elif chart_type['type'] in ['polar_scatter', 'polar_line', 'polar_line_multi']:
            # POLAR LINE/SCATTER
            ax = fig.add_subplot(111, projection='polar')
            ax.set_facecolor(bg_color)
            
            if chart_type['type'] == 'polar_line_multi':
                num_series = random.randint(2, 4)
                for i in range(num_series):
                    theta = np.linspace(0, 2 * np.pi, num_points)
                    r = np.abs(np.random.randn(num_points) * random.uniform(1, 5) + random.uniform(2, 8))
                    color = generate_random_colors(1)[0]
                    ax.plot(theta, r, 'o-', linewidth=random.randint(1, 3), 
                           markersize=random.randint(4, 8), color=color,
                           label=generate_random_text(5, 10))
            else:
                theta = np.linspace(0, 2 * np.pi, num_points)
                r = np.abs(np.random.randn(num_points) * random.uniform(1, 5) + random.uniform(2, 8))
                color = generate_random_colors(1)[0]
                
                if chart_type['type'] == 'polar_scatter':
                    ax.scatter(theta, r, s=random.randint(50, 200), alpha=random.uniform(0.5, 0.9),
                             color=color)
                else:  # polar_line
                    ax.plot(theta, r, 'o-', linewidth=random.randint(2, 4), 
                           markersize=random.randint(5, 10), color=color)
            
            # Grid e labels
            ax.grid(True, linestyle='--', alpha=random.uniform(0.4, 0.8))
            if random.random() > 0.3:
                ax.set_theta_zero_location(random.choice(['N', 'E', 'S', 'W']))
            
        elif chart_type['type'] == 'polar_bar':
            # POLAR BAR
            ax = fig.add_subplot(111, projection='polar')
            ax.set_facecolor(bg_color)
            
            theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            radii = [random.uniform(1, 10) for _ in range(num_points)]
            width = (2 * np.pi) / num_points * random.uniform(0.6, 0.95)
            colors = generate_random_colors(num_points)
            
            bars = ax.bar(theta, radii, width=width, bottom=0.0, alpha=random.uniform(0.7, 0.95))
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
            
            # Labels
            if random.random() > 0.3:
                ax.set_xticks(theta)
                ax.set_xticklabels(labels, size=random.randint(8, 12))
            
            ax.grid(True, linestyle='--', alpha=random.uniform(0.3, 0.7))
            
        elif chart_type['type'] == 'polar_area':
            # POLAR AREA (come rose chart)
            ax = fig.add_subplot(111, projection='polar')
            ax.set_facecolor(bg_color)
            
            theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            radii = [random.uniform(2, 10) for _ in range(num_points)]
            width = (2 * np.pi) / num_points
            colors = generate_random_colors(num_points)
            
            bars = ax.bar(theta, radii, width=width, bottom=0.0, alpha=random.uniform(0.6, 0.9))
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
                bar.set_edgecolor('white')
                bar.set_linewidth(random.randint(1, 3))
            
            # Labels
            if random.random() > 0.2:
                ax.set_xticks(theta)
                ax.set_xticklabels(labels, size=random.randint(8, 12))
            
            ax.grid(True, linestyle=random.choice(['-', '--', '-.', ':']), 
                   alpha=random.uniform(0.3, 0.7))
        
        # Titolo casuale
        if random.random() > 0.3:
            title = generate_random_text(min_len=8, max_len=20)
            ax.set_title(title, fontsize=random.randint(14, 20), 
                        fontweight=random.choice(['normal', 'bold']), 
                        pad=20)
        
        # Legenda casuale
        if random.random() > 0.6 and chart_type['type'] in ['radar_multi', 'polar_line_multi']:
            ax.legend(loc=random.choice(['upper right', 'upper left', 'lower right', 'lower left']),
                     fontsize=random.randint(8, 12),
                     frameon=random.random() > 0.5)
        
        # Salva
        filename = OUTPUT_DIR / f"{chart_type['name']}_{chart_id:06d}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=100, facecolor=bg_color)
        plt.close(fig)
        
        # Progress update
        if chart_id % 100 == 0:
            print(f"Generati: {chart_id}/{total}")
        
        return True
        
    except Exception as e:
        print(f"Errore nel grafico {chart_id}: {e}")
        return False


def generate_charts_batch(num_charts, num_processes=None):
    """
    Genera batch di grafici usando multiprocessing
    
    Args:
        num_charts: Numero totale di grafici da generare
        num_processes: Numero di processi paralleli (default: CPU count - 1)
    """
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    print(f"🚀 Avvio generazione di {num_charts} grafici radar/polar...")
    print(f"💻 Utilizzo {num_processes} processi paralleli")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    start_time = time.time()
    
    # Crea argomenti per ogni grafico
    args_list = [(i, num_charts) for i in range(num_charts)]
    
    # Genera grafici in parallelo
    with Pool(processes=num_processes) as pool:
        results = pool.map(generate_single_chart, args_list)
    
    # Statistiche
    successful = sum(results)
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Completato!")
    print(f"📊 Grafici generati con successo: {successful}/{num_charts}")
    print(f"⏱️  Tempo totale: {elapsed_time:.2f} secondi")
    print(f"⚡ Velocità media: {successful/elapsed_time:.2f} grafici/secondo")
    print(f"📁 File salvati in: {OUTPUT_DIR.absolute()}")


def generate_charts_sequential(num_charts):
    """
    Genera grafici sequenzialmente (utile per debug o sistemi con poca RAM)
    
    Args:
        num_charts: Numero totale di grafici da generare
    """
    print(f"🚀 Avvio generazione sequenziale di {num_charts} grafici radar/polar...")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    start_time = time.time()
    successful = 0
    
    for i in range(num_charts):
        if generate_single_chart((i, num_charts)):
            successful += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Completato!")
    print(f"📊 Grafici generati con successo: {successful}/{num_charts}")
    print(f"⏱️  Tempo totale: {elapsed_time:.2f} secondi")
    print(f"⚡ Velocità media: {successful/elapsed_time:.2f} grafici/secondo")


if __name__ == "__main__":
    # Configurazione della generazione
    NUM_CHARTS = 2774  # Modifica questo valore per generare più o meno grafici
    USE_MULTIPROCESSING = True  # Imposta False per generazione sequenziale
    NUM_PROCESSES = None  # None = auto (CPU count - 1), oppure specifica un numero
    
    print("=" * 60)
    print("GENERATORE RADAR & POLAR CHARTS PER DATASET AI")
    print("=" * 60)
    
    if USE_MULTIPROCESSING:
        generate_charts_batch(NUM_CHARTS, NUM_PROCESSES)
    else:
        generate_charts_sequential(NUM_CHARTS)
    
    print(f"\n🎉 Dataset pronto per l'addestramento!")