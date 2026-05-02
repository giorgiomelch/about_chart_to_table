import matplotlib.pyplot as plt
import squarify
import random
import string
import os
import numpy as np
import matplotlib.cm as cm

def get_random_string(length):
    """Genera una stringa casuale di caratteri alfanumerici."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def generate_random_data():
    """
    Genera valori e etichette con distribuzioni statistiche diverse
    per garantire variabilità nella struttura del Treemap.
    """
    # Numero casuale di blocchi (da pochi a molti per variare la densità)
    num_items = random.randint(5, 50)
    
    # Scelta casuale della distribuzione dei dati per variare le proporzioni dei rettangoli
    dist_type = random.choice(['uniform', 'exponential', 'pareto'])
    
    if dist_type == 'uniform':
        values = np.random.uniform(1, 100, num_items)
    elif dist_type == 'exponential':
        values = np.random.exponential(10, num_items)
    elif dist_type == 'pareto':
        values = np.random.pareto(a=2.0, size=num_items) * 10
    
    values = [max(v, 1) for v in values] # Assicura che non ci siano valori 0 o negativi
    values.sort(reverse=True) # I treemap funzionano meglio ordinati
    
    # Generazione etichette casuali di lunghezza variabile
    labels = [get_random_string(random.randint(3, 10)) for _ in range(num_items)]
    
    return values, labels

def generate_treemap_dataset():
    output_dir = "treemap_dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        num_charts = int(input("Quanti grafici vuoi generare? "))
    except ValueError:
        print("Inserisci un numero intero valido.")
        return

    print(f"Inizio generazione di {num_charts} immagini...")

    # Lista di colormap per variare drasticamente i colori
    colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
        'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ]

    for i in range(3000, num_charts):
        values, labels = generate_random_data()
        
        # 1. Variazione Aspect Ratio (Dimensioni immagine)
        # Larghezza e altezza variano tra 6 e 12 pollici
        fig_width = random.uniform(6, 12)
        fig_height = random.uniform(6, 12)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)

        # 2. Variazione Colori
        cmap_name = random.choice(colormaps)
        cmap = cm.get_cmap(cmap_name)
        # Genera colori campionando la colormap
        norm = plt.Normalize(min(values), max(values))
        colors = [cmap(random.random()) for _ in range(len(values))]

        # 3. Variazione Bordi e Padding
        # Padding tra i blocchi (a volte nullo, a volte evidente)
        pad = random.choice([0, 0, 0.5, 1, 2]) 
        
        # Spessore del bordo (a volte nullo, a volte spesso)
        edge_width = random.choice([0, 1, 2, 3])
        edge_color = random.choice(['#FFFFFF', '#000000', '#333333'])
        
        # Trasparenza
        alpha = random.uniform(0.7, 1.0)

        # 4. Generazione Plot
        try:
            squarify.plot(sizes=values, label=labels, color=colors, 
                          alpha=alpha, pad=pad, 
                          ec=edge_color, linewidth=edge_width, ax=ax)
        except Exception as e:
            # Fallback raro se i calcoli geometrici falliscono
            print(f"Errore generazione grafico {i}: {e}")
            plt.close(fig)
            continue

        # 5. Variazione Testo (Font e Colore)
        # Rimuove assi per pulizia
        plt.axis('off')
        
        # Manipolazione casuale delle proprietà del testo generato da squarify
        text_color = random.choice(['black', 'white', 'blue', 'darkred'])
        font_size = random.randint(8, 14)
        
        for text in ax.texts:
            text.set_color(text_color)
            text.set_fontsize(font_size)
            # Rotazione casuale occasionale per aumentare difficoltà dataset
            if random.random() > 0.9: 
                text.set_rotation(90)

        # 6. Variazione Sfondo (dietro il treemap)
        fig.patch.set_facecolor(random.choice(['white', 'black', 'gray', '#f0f0f0']))

        # Salvataggio
        filename = os.path.join(output_dir, f"treemap_{i:05d}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=random.choice([72, 100, 150]))
        plt.close(fig) # Fondamentale per liberare memoria

        if (i + 1) % 100 == 0:
            print(f"Generati {i + 1} grafici...")

    print(f"Completato. Immagini salvate in '{output_dir}'.")

if __name__ == "__main__":
    generate_treemap_dataset()