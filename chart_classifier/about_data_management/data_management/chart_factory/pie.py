"""
Generatore Ottimizzato di Pie Charts per Dataset AI
Genera migliaia di grafici a torta con variazioni casuali
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import string
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Configurazione
OUTPUT_DIR = Path("pie_charts_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Caratteri per testo random (inclusi caratteri accentati italiani)
CHARS = string.ascii_letters + string.digits + 'àèéìòùÀÈÉÌÒÙ'

# Tipi di pie chart con configurazioni
CHART_TYPES = [
    {'name': 'pie', 'startangle': 0, 'wedgeprops': None, 'pctdistance': 0.85},
    {'name': 'pie', 'startangle': 90, 'wedgeprops': None, 'pctdistance': 0.85},
    {'name': 'pie', 'startangle': 45, 'wedgeprops': None, 'pctdistance': 0.85},
    {'name': 'pie', 'startangle': 135, 'wedgeprops': None, 'pctdistance': 0.85},
    {'name': 'donut', 'startangle': 0, 'wedgeprops': {'width': 0.5}, 'pctdistance': 0.75},
    {'name': 'donut', 'startangle': 90, 'wedgeprops': {'width': 0.5}, 'pctdistance': 0.75},
    {'name': 'donut_thin', 'startangle': 0, 'wedgeprops': {'width': 0.3}, 'pctdistance': 0.65},
    {'name': 'donut_thick', 'startangle': 0, 'wedgeprops': {'width': 0.7}, 'pctdistance': 0.8},
]


def generate_background_color():
    """Genera un colore di sfondo con alta probabilità di bianco/chiaro"""
    rand = random.random()
    
    # 60% bianco puro
    if rand < 0.6:
        return 'white'
    
    # 20% colori molto chiari (quasi bianchi)
    elif rand < 0.8:
        # Colori con RGB tra 240-255
        r = random.randint(240, 255)
        g = random.randint(240, 255)
        b = random.randint(240, 255)
        return (r/255, g/255, b/255)
    
    # 15% colori chiari pastello
    elif rand < 0.95:
        # Colori con RGB tra 200-240
        r = random.randint(200, 240)
        g = random.randint(200, 240)
        b = random.randint(200, 240)
        return (r/255, g/255, b/255)
    
    # 5% colori casuali (qualsiasi colore)
    else:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r/255, g/255, b/255)


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


def generate_single_chart(args):
    """Genera un singolo grafico - funzione per multiprocessing"""
    chart_id, total = args
    
    try:
        # Parametri casuali
        num_slices = random.randint(3, 8)
        chart_type = random.choice(CHART_TYPES)
        
        # Genera dati
        labels = [generate_random_text() for _ in range(num_slices)]
        sizes = [random.randint(10, 100) for _ in range(num_slices)]
        colors = generate_random_colors(num_slices)
        
        # Opzioni casuali
        explode = tuple(random.uniform(0, 0.1) if random.random() > 0.7 else 0 for _ in range(num_slices))
        show_labels = random.random() > 0.2
        show_percentages = random.random() > 0.3
        show_legend = random.random() > 0.3
        shadow = random.random() > 0.5
        
        # Crea figura con background casuale
        bg_color = generate_background_color()
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # Autopct personalizzato
        autopct = '%1.1f%%' if show_percentages else None
        
        # Crea pie chart
        if autopct:
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels if show_labels else None,
                colors=colors,
                autopct=autopct,
                startangle=chart_type['startangle'],
                explode=explode,
                shadow=shadow,
                wedgeprops=chart_type['wedgeprops'],
                pctdistance=chart_type['pctdistance'],
                textprops={'fontsize': random.randint(8, 14)}
            )
            # Formattazione testo percentuali
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(random.randint(8, 12))
        else:
            wedges, texts = ax.pie(
                sizes,
                labels=labels if show_labels else None,
                colors=colors,
                autopct=autopct,
                startangle=chart_type['startangle'],
                explode=explode,
                shadow=shadow,
                wedgeprops=chart_type['wedgeprops'],
                textprops={'fontsize': random.randint(8, 14)}
            )
        
        # Titolo casuale
        if random.random() > 0.4:
            title = generate_random_text(min_len=8, max_len=20)
            ax.set_title(title, fontsize=random.randint(14, 20), fontweight='bold', pad=20)
        
        # Legenda casuale
        if show_legend:
            legend_pos = random.choice(['upper left', 'upper right', 'lower left', 'lower right', 'right', 'center left', 'center right', 'best'])
            ax.legend(
                wedges, 
                labels,
                loc=legend_pos,
                fontsize=random.randint(8, 12),
                frameon=random.random() > 0.5
            )
        
        ax.axis('equal')
        
        # Salva
        filename = OUTPUT_DIR / f"pie_chart_{chart_id:07d}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=100)
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
    
    print(f"🚀 Avvio generazione di {num_charts} grafici...")
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
    print(f"🚀 Avvio generazione sequenziale di {num_charts} grafici...")
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
    NUM_CHARTS = 19  # Modifica questo valore per generare più o meno grafici
    USE_MULTIPROCESSING = True  # Imposta False per generazione sequenziale
    NUM_PROCESSES = None  # None = auto (CPU count - 1), oppure specifica un numero
    
    print("=" * 60)
    print("GENERATORE PIE CHARTS PER DATASET AI")
    print("=" * 60)
    
    if USE_MULTIPROCESSING:
        generate_charts_batch(NUM_CHARTS, NUM_PROCESSES)
    else:
        generate_charts_sequential(NUM_CHARTS)
    
    print(f"\n🎉 Dataset pronto per l'addestramento!")