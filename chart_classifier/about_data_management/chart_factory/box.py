import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per ambienti senza display
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string

def genera_testo_random(lunghezza_min=3, lunghezza_max=15, tipo='misto'):
    """Genera testo casuale con vari formati"""
    lunghezza = random.randint(lunghezza_min, lunghezza_max)
    
    if tipo == 'lettere':
        return ''.join(random.choices(string.ascii_letters, k=lunghezza))
    elif tipo == 'maiuscole':
        return ''.join(random.choices(string.ascii_uppercase, k=lunghezza))
    elif tipo == 'minuscole':
        return ''.join(random.choices(string.ascii_lowercase, k=lunghezza))
    elif tipo == 'numeri':
        return ''.join(random.choices(string.digits, k=lunghezza))
    elif tipo == 'alfanumerico':
        return ''.join(random.choices(string.ascii_letters + string.digits, k=lunghezza))
    elif tipo == 'parola':
        # Genera parole più realistiche alternando consonanti e vocali
        vocali = 'aeiou'
        consonanti = 'bcdfghlmnpqrstvwxyz'
        parola = ''
        for i in range(lunghezza):
            if i % 2 == 0:
                parola += random.choice(consonanti)
            else:
                parola += random.choice(vocali)
        return parola.capitalize() if random.random() > 0.5 else parola
    else:  # misto
        caratteri = string.ascii_letters + string.digits + '_-'
        return ''.join(random.choices(caratteri, k=lunghezza))

def genera_boxplot(idx, output_dir="box_synt"):
    """Genera un singolo box chart con variazioni randomiche estreme"""
    
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Parametri randomici per presenza elementi (soglie personalizzabili)
    prob_xlabel = random.random() > 0.1
    prob_ylabel = random.random() > 0.1
    prob_title = random.random() > 0.15
    prob_legend = random.random() > 0.15
    prob_grid = random.random() > 0.2
    prob_xticks = random.random() > 0.1
    prob_yticks = random.random() > 0.1
    prob_background = random.random() > 0.3
    prob_spines = random.random() > 0.3
    prob_outliers = random.random() > 0.2
    prob_horizontal = random.random() > 0.25
    
    # Numero casuale di box (da 2 a 12)
    n_boxes = random.randint(2, 12)
    
    # Numero casuale di campioni (da 15 a 300)
    n_samples = random.randint(15, 300)
    
    # Genera dati casuali con distribuzioni molto variabili
    data = []
    distribution_types = ['normal', 'uniform', 'exponential', 'bimodal', 'extreme']
    
    for i in range(n_boxes):
        dist_type = random.choice(distribution_types)
        
        if dist_type == 'normal':
            mean = random.uniform(-20, 20)
            std = random.uniform(0.3, 8)
            samples = np.random.normal(mean, std, n_samples)
        elif dist_type == 'uniform':
            low = random.uniform(-15, 10)
            high = low + random.uniform(5, 25)
            samples = np.random.uniform(low, high, n_samples)
        elif dist_type == 'exponential':
            scale = random.uniform(0.5, 5)
            samples = np.random.exponential(scale, n_samples) * random.choice([1, -1])
        elif dist_type == 'bimodal':
            mean1 = random.uniform(-10, 0)
            mean2 = random.uniform(5, 15)
            std = random.uniform(1, 3)
            half = n_samples // 2
            samples = np.concatenate([
                np.random.normal(mean1, std, half),
                np.random.normal(mean2, std, n_samples - half)
            ])
        else:  # extreme
            samples = np.random.standard_t(df=3, size=n_samples) * random.uniform(2, 8)
        
        # Aggiungi outliers casuali
        if random.random() > 0.7:
            n_outliers = random.randint(1, 5)
            outlier_indices = random.sample(range(len(samples)), n_outliers)
            for idx_out in outlier_indices:
                samples[idx_out] *= random.uniform(2, 5) * random.choice([1, -1])
        
        data.append(samples)
    
    # Crea figura con dimensioni molto variabili
    fig_width = random.uniform(5, 16)
    fig_height = random.uniform(3, 12)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Colore di sfondo molto variabile
    if prob_background:
        bg_colors = [
            '#f0f0f0', '#e8f4f8', '#fff9e6', '#f5f5dc', '#e6f3ff', 'white', 
            '#fafafa', '#ffe6e6', '#e6ffe6', '#f0e6ff', '#fff0e6', '#e6f7ff',
            '#2c3e50', '#34495e', '#1a1a1a', '#0d0d0d'  # Sfondi scuri
        ]
        bg_color = random.choice(bg_colors)
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)
        
        # Colori testo adattivi per sfondi scuri
        is_dark = bg_color in ['#2c3e50', '#34495e', '#1a1a1a', '#0d0d0d']
        text_color = 'white' if is_dark else 'black'
    else:
        bg_color = 'white'
        text_color = 'black'
    
    # Stili di box molto randomici
    box_widths = random.uniform(0.2, 0.95)
    patch_artist = True  # SEMPRE True per avere controllo sui contorni
    showmeans = random.choice([True, False])
    showfliers = showfliers_val = random.choice([True, False]) if not prob_outliers else False
    notch = random.choice([True, False])
    showcaps = True  # SEMPRE visibili per mostrare min/max
    showbox = True  # SEMPRE visibile
    
    # Stili whiskers variabili (solo float o tuple di percentili)
    whis_options = [1.5, 2.0, 2.5, 1.0, 3.0, (5, 95), (10, 90), (1, 99), (25, 75)]
    whis = random.choice(whis_options)
    
    # Colori molto variabili
    color_schemes = [
        ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', '#6BCB77'],
        ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#96c93d', '#fa709a'],
        ['#fa709a', '#fee140', '#30cfd0', '#330867', '#96c93d', '#ff6a88', '#fbc687'],
        ['#ff9a56', '#ff6a88', '#fbc687', '#a8edea', '#fed6e3', '#ff9a9e', '#fad0c4'],
        ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#ecf0f1', '#3498db'],
        ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#1abc9c', '#3498db'],
        ['#8e44ad', '#9b59b6', '#c0392b', '#d35400', '#27ae60', '#16a085', '#2980b9'],
        ['#000000', '#2d2d2d', '#5a5a5a', '#878787', '#b4b4b4', '#e1e1e1', '#555555'],
    ]
    selected_scheme = random.choice(color_schemes)
    colors = random.choices(selected_scheme, k=n_boxes)
    
    # Se sfondo bianco/chiaro, evita colori chiari per i box
    if prob_background and bg_color in ['white', '#fafafa', '#f0f0f0', '#e8f4f8', '#fff9e6', '#f5f5dc', '#e6f3ff', '#ffe6e6', '#e6ffe6', '#f0e6ff', '#fff0e6', '#e6f7ff']:
        dark_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#667eea', '#764ba2', '#fa709a', 
                      '#2c3e50', '#34495e', '#e74c3c', '#8e44ad', '#c0392b', '#27ae60']
        colors = random.choices(dark_colors, k=n_boxes)
    # Se sfondo scuro, evita colori scuri per i box
    elif prob_background and bg_color in ['#2c3e50', '#34495e', '#1a1a1a', '#0d0d0d']:
        light_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', 
                       '#6BCB77', '#f093fb', '#4facfe', '#00f2fe', '#fee140', '#fbc687', 
                       '#a8edea', '#fed6e3', '#ff9a9e', '#ecf0f1', '#f1c40f', '#2ecc71']
        colors = random.choices(light_colors, k=n_boxes)
    
    # Orientamento orizzontale o verticale
    vert = not prob_horizontal
    
    # Crea boxplot
    bp = ax.boxplot(data, 
                    widths=box_widths,
                    patch_artist=patch_artist,
                    showmeans=showmeans,
                    showfliers=showfliers_val,
                    notch=notch,
                    vert=vert,
                    whis=whis,
                    showcaps=showcaps,
                    showbox=showbox,
                    meanprops=dict(marker=random.choice(['o', 'D', '^', 's', '*']),
                                  markerfacecolor=random.choice(['red', 'green', 'blue', 'orange', 'purple']),
                                  markersize=random.uniform(4, 10)),
                    flierprops=dict(marker=random.choice(['o', '+', 'x', 'd', '*', '.', ',']),
                                   markerfacecolor=random.choice(colors),
                                   markersize=random.uniform(2, 8),
                                   alpha=random.uniform(0.3, 0.8)))
    
    # COLORA LE BOX CON CONTORNI SEMPRE DEFINITI
    # Definisci colori di contorno scuri che contrastino sempre
    edge_colors_dark = ['#000000', '#1a1a1a', '#2c2c2c', '#0d0d0d', '#333333']
    edge_colors_varied = ['#000000', '#2c3e50', '#34495e', '#1a1a1a', '#8B0000', '#006400', '#00008B']
    
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        # Imposta il colore di riempimento
        patch.set_facecolor(color)
        patch.set_alpha(random.uniform(0.6, 0.95))
        
        # CONTORNO SEMPRE DEFINITO E VISIBILE
        # Linewidth sempre sostanziale (mai troppo sottile)
        edge_linewidth = random.uniform(1.5, 3.5)
        patch.set_linewidth(edge_linewidth)
        
        # Colore del contorno: scegli tra nero, grigio scuro o colori molto scuri
        edge_color = random.choice(edge_colors_varied)
        patch.set_edgecolor(edge_color)
        
        # BORDI DELLE BOX SEMPRE SOLIDI (nessun tratteggio)
        patch.set_linestyle('-')
        
        # Pattern di riempimento casuali (opzionale)
        if random.random() > 0.8:
            patch.set_hatch(random.choice(['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']))
    
    # Personalizza linee del boxplot con variazioni estreme
    for element in ['whiskers', 'caps', 'medians', 'boxes']:
        if element in bp:
            for line in bp[element]:
                # Assicura linewidth visibile per tutti gli elementi chiave
                if element in ['boxes', 'medians']:
                    line.set_linewidth(random.uniform(1.5, 3.5))
                elif element in ['whiskers', 'caps']:
                    # Whiskers e caps sempre visibili (min/max)
                    line.set_linewidth(random.uniform(1.2, 3.0))
                else:
                    line.set_linewidth(random.uniform(1.0, 3))
                
                line.set_linestyle(random.choice(['-', '--', '-.', ':']))
                
                if element == 'medians':
                    line.set_color(random.choice(['red', 'black', 'blue', 'green', 'orange', 'purple', 'cyan']))
                    line.set_linewidth(random.uniform(1.8, 4))
                elif element == 'whiskers':
                    # Whiskers sempre visibili con colori scuri
                    line.set_color(random.choice(['black', 'darkgray', 'darkblue', 'darkred', 'darkgreen']))
                    line.set_alpha(1.0)  # Alpha pieno per visibilità
                elif element == 'caps':
                    # Caps sempre visibili
                    line.set_color(random.choice(['black', 'darkgray', 'darkblue']))
                    line.set_alpha(1.0)
    
    # Font molto variabili
    available_fonts = [
        'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif', 'serif', 
        'monospace', 'DejaVu Serif', 'Courier New', 'Times New Roman'
    ]
    font_family = random.choice(available_fonts)
    font_size_title = random.randint(8, 22)
    font_size_labels = random.randint(7, 16)
    font_size_ticks = random.randint(5, 13)
    font_weight = random.choice(['normal', 'bold', 'light'])
    font_style = random.choice(['normal', 'italic'])
    
    # Etichette assi con testo completamente random
    tipo_testo_label = random.choice(['lettere', 'maiuscole', 'minuscole', 'alfanumerico', 'parola', 'misto'])
    
    if vert:
        if prob_xlabel:
            xlabel_text = genera_testo_random(4, 12, tipo_testo_label)
            ax.set_xlabel(xlabel_text, fontsize=font_size_labels, 
                         family=font_family, weight=font_weight, style=font_style, color=text_color)
        if prob_ylabel:
            ylabel_text = genera_testo_random(4, 12, tipo_testo_label)
            ax.set_ylabel(ylabel_text, fontsize=font_size_labels, 
                         family=font_family, weight=font_weight, style=font_style, color=text_color)
    else:
        if prob_xlabel:
            xlabel_text = genera_testo_random(4, 12, tipo_testo_label)
            ax.set_xlabel(xlabel_text, fontsize=font_size_labels, 
                         family=font_family, weight=font_weight, style=font_style, color=text_color)
        if prob_ylabel:
            ylabel_text = genera_testo_random(4, 12, tipo_testo_label)
            ax.set_ylabel(ylabel_text, fontsize=font_size_labels, 
                         family=font_family, weight=font_weight, style=font_style, color=text_color)
    
    # Tick labels completamente random
    tipo_testo_tick = random.choice(['lettere', 'maiuscole', 'minuscole', 'alfanumerico', 'numeri', 'parola', 'misto'])
    
    if prob_xticks:
        # Genera etichette random per ogni box
        random_labels = [genera_testo_random(2, 8, tipo_testo_tick) for _ in range(n_boxes)]
        
        rotation = random.choice([0, 15, 30, 45, 60, 90, -15, -30, -45])
        ha = 'right' if rotation > 30 or rotation < -30 else 'center'
        
        if vert:
            ax.set_xticklabels(random_labels, fontsize=font_size_ticks, 
                              rotation=rotation, ha=ha, color=text_color)
        else:
            ax.set_yticklabels(random_labels, fontsize=font_size_ticks, color=text_color)
    else:
        if vert:
            ax.set_xticklabels([])
        else:
            ax.set_yticklabels([])
    
    if not prob_yticks:
        if vert:
            ax.set_yticklabels([])
        else:
            ax.set_xticklabels([])
    else:
        if vert:
            ax.tick_params(axis='y', labelsize=font_size_ticks, colors=text_color)
        else:
            ax.tick_params(axis='x', labelsize=font_size_ticks, colors=text_color)
    
    # Grid molto variabile
    if prob_grid:
        grid_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 5)), (0, (3, 1, 1, 1))]
        grid_alpha = random.uniform(0.15, 0.7)
        grid_color = random.choice(['gray', 'black', 'lightgray', '#cccccc', '#333333'])
        grid_linewidth = random.uniform(0.3, 2)
        grid_axis = random.choice(['both', 'y', 'x'])
        ax.grid(True, alpha=grid_alpha, linestyle=random.choice(grid_styles), 
               axis=grid_axis, color=grid_color, linewidth=grid_linewidth)
        # Layer della grid
        ax.set_axisbelow(random.choice([True, False]))
    
    # Legenda con testo completamente random
    if prob_legend:
        n_legend_items = min(n_boxes, random.randint(3, 8))
        tipo_testo_legend = random.choice(['lettere', 'maiuscole', 'minuscole', 'alfanumerico', 'parola', 'misto'])
        legend_labels = [genera_testo_random(3, 10, tipo_testo_legend) for _ in range(n_legend_items)]
        
        ax.legend(bp['boxes'][:n_legend_items], legend_labels, 
                 loc=random.choice(['upper right', 'upper left', 'lower right', 'lower left', 
                                   'center right', 'center left', 'upper center', 'lower center', 'best']),
                 fontsize=random.randint(5, 12),
                 frameon=random.choice([True, False]),
                 shadow=random.choice([True, False]),
                 fancybox=random.choice([True, False]),
                 framealpha=random.uniform(0.5, 1.0),
                 ncol=random.choice([1, 1, 1, 2]),
                 facecolor=random.choice(['white', 'lightgray', 'lightyellow', bg_color]) if prob_background else 'white')
    
    # Titolo completamente random
    if prob_title:
        tipo_testo_title = random.choice(['lettere', 'maiuscole', 'minuscole', 'alfanumerico', 'parola', 'misto'])
        title_text = genera_testo_random(5, 20, tipo_testo_title)
        title_pad = random.uniform(5, 30)
        ax.set_title(title_text, fontsize=font_size_title, 
                    family=font_family, pad=title_pad, weight=font_weight, 
                    style=font_style, color=text_color,
                    loc=random.choice(['center', 'left', 'right']))
    
    # Bordi del grafico molto variabili
    if prob_spines:
        # Rimuovi alcuni bordi casualmente
        spines_to_remove = random.sample(['top', 'right', 'bottom', 'left'], 
                                        k=random.randint(0, 3))
        for spine_name in spines_to_remove:
            ax.spines[spine_name].set_visible(False)
        
        # Personalizza bordi rimanenti
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(random.uniform(0.5, 3))
                spine.set_color(random.choice(['black', 'gray', text_color]))
                if random.random() > 0.8:
                    spine.set_linestyle(random.choice(['--', '-.', ':']))
    
    # Aggiungi margini casuali
    if vert:
        ax.margins(x=random.uniform(0.01, 0.15), y=random.uniform(0.05, 0.2))
    else:
        ax.margins(y=random.uniform(0.01, 0.15), x=random.uniform(0.05, 0.2))
    
    # Tick marks personalizzati
    tick_direction = random.choice(['in', 'out', 'inout'])
    tick_length = random.uniform(2, 8)
    tick_width = random.uniform(0.5, 2.5)
    ax.tick_params(direction=tick_direction, length=tick_length, width=tick_width, colors=text_color)
    
    # Aspetto generale
    plt.tight_layout(pad=random.uniform(0.5, 3))
    
    # Salva figura con parametri variabili
    filename = os.path.join(output_dir, f'boxplot_{idx:05d}.png')
    dpi = random.choice([72, 96, 100, 120, 150, 200, 300])
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
               facecolor=fig.get_facecolor(),
               edgecolor=random.choice(['none', 'black', 'white']),
               transparent=random.choice([False, False, False, True]))  # Bias verso False
    plt.close()
    
    return filename

def genera_dataset(n_grafici):
    """Genera un dataset di n box chart"""
    print(f"Generazione di {n_grafici} box chart in corso...")
    print(f"Cartella di output: box_synt/")
    
    for i in range(n_grafici):
        filename = genera_boxplot(i)
        if (i + 1) % 10 == 0 or (i + 1) == n_grafici:
            print(f"Generati {i + 1}/{n_grafici} grafici")
    
    print(f"\n✓ Dataset completato! {n_grafici} grafici salvati nella cartella 'box_synt'")

# Esecuzione
if __name__ == "__main__":
    n = int(input("Inserisci il numero di box chart da generare: "))
    genera_dataset(n)