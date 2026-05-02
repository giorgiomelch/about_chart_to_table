import matplotlib.pyplot as plt
import numpy as np
import random
import string
from matplotlib import font_manager
import os

def generate_random_string(min_length=3, max_length=12):
    """Genera una stringa random"""
    length = random.randint(min_length, max_length)
    if random.random() < 0.3:
        # Parole realistiche
        words = ['Vendite', 'Profitti', 'Utenti', 'Visite', 'Costi', 'Revenue', 'Budget', 
                 'Margine', 'Quota', 'Target', 'Sales', 'Growth', 'Market', 'Share',
                 'Performance', 'Traffic', 'Conversioni', 'Engagement', 'ROI', 'EBITDA']
        return random.choice(words)
    else:
        # Stringhe casuali
        return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=length))

def generate_random_data(num_points):
    """Genera dati variabili con pattern diversi"""
    pattern = random.choice(['linear', 'exponential', 'sine', 'random', 'stepped', 'polynomial'])
    
    if pattern == 'linear':
        slope = random.uniform(-5, 5)
        intercept = random.uniform(0, 100)
        data = [slope * i + intercept + random.uniform(-10, 10) for i in range(num_points)]
    elif pattern == 'exponential':
        base = random.uniform(1.01, 1.2)
        data = [random.uniform(10, 30) * (base ** i) + random.uniform(-5, 5) for i in range(num_points)]
    elif pattern == 'sine':
        amplitude = random.uniform(20, 80)
        frequency = random.uniform(0.1, 0.5)
        offset = random.uniform(30, 70)
        data = [amplitude * np.sin(frequency * i) + offset + random.uniform(-5, 5) for i in range(num_points)]
    elif pattern == 'stepped':
        data = []
        current = random.uniform(20, 80)
        for i in range(num_points):
            if i % random.randint(2, 5) == 0:
                current += random.uniform(-20, 20)
            data.append(current + random.uniform(-3, 3))
    elif pattern == 'polynomial':
        coeffs = [random.uniform(-0.1, 0.1) for _ in range(3)]
        data = [coeffs[0]*i**2 + coeffs[1]*i + random.uniform(20, 60) + random.uniform(-5, 5) for i in range(num_points)]
    else:
        data = [random.uniform(0, 100) for _ in range(num_points)]
    
    return [max(0, x) for x in data]  # Evita valori negativi

def get_random_color():
    """Genera un colore random in vari formati"""
    color_type = random.choice(['hex', 'rgb', 'named'])
    
    if color_type == 'hex':
        return '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
    elif color_type == 'rgb':
        return (random.random(), random.random(), random.random())
    else:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 
                  'yellow', 'brown', 'pink', 'lime', 'navy', 'teal', 'coral', 'indigo']
        return random.choice(colors)

def create_area_chart(filename):
    """Crea un area chart con stile variabile"""
    # Parametri variabili
    num_series = random.randint(1, 5)
    num_points = random.randint(5, 20)
    
    # Dimensione figura variabile
    fig_width = random.uniform(6, 14)
    fig_height = random.uniform(4, 10)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Genera categorie
    if random.random() < 0.5:
        categories = [generate_random_string(1, 8) for _ in range(num_points)]
    else:
        categories = [f'Q{i+1}' if random.random() < 0.3 else str(2020+i) if random.random() < 0.5 
                     else f'W{i+1}' for i in range(num_points)]
    
    x = np.arange(len(categories))
    
    # Stile di riempimento
    alpha = random.uniform(0.3, 0.9)
    
    # Genera e plotta le serie
    for i in range(num_series):
        data = generate_random_data(num_points)
        color = get_random_color()
        label = generate_random_string()
        
        # Vari stili di area
        if random.random() < 0.5:
            ax.fill_between(x, data, alpha=alpha, color=color, label=label)
        else:
            ax.fill_between(x, data, alpha=alpha, color=color, label=label, 
                          edgecolor=get_random_color(), linewidth=random.uniform(0.5, 3))
    
    # Titolo e labels con font variabili
    title_size = random.randint(10, 20)
    label_size = random.randint(8, 14)
    
    if random.random() < 0.7:
        ax.set_title(generate_random_string(5, 20), fontsize=title_size, 
                    fontweight=random.choice(['normal', 'bold']))
    
    if random.random() < 0.6:
        ax.set_xlabel(generate_random_string(4, 12), fontsize=label_size)
    
    if random.random() < 0.6:
        ax.set_ylabel(generate_random_string(4, 12), fontsize=label_size)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=random.choice([0, 45, 90]), 
                       fontsize=random.randint(7, 12))
    
    # Grid variabile
    if random.random() < 0.5:
        ax.grid(random.random() < 0.5, alpha=random.uniform(0.2, 0.6), 
               linestyle=random.choice(['-', '--', '-.', ':']),
               axis=random.choice(['both', 'x', 'y']))
    
    # Legenda variabile
    if num_series > 1 and random.random() < 0.7:
        ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 
                                     'lower right', 'center', 'best']),
                 fontsize=random.randint(7, 11))
    
    # Background variabile
    if random.random() < 0.4:
        ax.set_facecolor(get_random_color() if random.random() < 0.3 else 
                        (random.random()*0.1, random.random()*0.1, random.random()*0.1))
    
    if random.random() < 0.3:
        fig.patch.set_facecolor(get_random_color() if random.random() < 0.2 else 
                               (random.random()*0.15, random.random()*0.15, random.random()*0.15))
    
    # Spine visibility
    if random.random() < 0.3:
        for spine in ax.spines.values():
            spine.set_visible(random.random() < 0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=random.randint(80, 150), bbox_inches='tight')
    plt.close()

def create_horizontal_bar_chart(filename):
    """Crea un grafico a barre orizzontali con stile variabile"""
    # Parametri variabili
    num_bars = random.randint(3, 15)
    
    # Dimensione figura variabile
    fig_width = random.uniform(6, 12)
    fig_height = random.uniform(4, 10)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Genera categorie
    categories = [generate_random_string(3, 15) for _ in range(num_bars)]
    values = generate_random_data(num_bars)
    
    y_pos = np.arange(len(categories))
    
    # Colori variabili
    if random.random() < 0.4:
        colors = [get_random_color() for _ in range(num_bars)]
    else:
        base_color = get_random_color()
        colors = [base_color] * num_bars
    
    # Stili di barre variabili
    bar_height = random.uniform(0.4, 0.9)
    
    if random.random() < 0.3:
        # Con bordi
        bars = ax.barh(y_pos, values, bar_height, color=colors, 
                      edgecolor=get_random_color(), linewidth=random.uniform(0.5, 2.5))
    else:
        bars = ax.barh(y_pos, values, bar_height, color=colors, alpha=random.uniform(0.6, 1.0))
    
    # Aggiungi valori sulle barre (a volte)
    if random.random() < 0.4:
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            label_position = random.choice(['inside', 'outside'])
            if label_position == 'inside' and width > max(values) * 0.1:
                ax.text(width * random.uniform(0.5, 0.9), bar.get_y() + bar.get_height()/2, 
                       f'{value:.1f}', ha='center', va='center', 
                       fontsize=random.randint(7, 11), fontweight=random.choice(['normal', 'bold']))
            elif label_position == 'outside':
                ax.text(width + max(values)*0.02, bar.get_y() + bar.get_height()/2, 
                       f'{value:.1f}', ha='left', va='center', 
                       fontsize=random.randint(7, 11))
    
    # Titolo e labels
    title_size = random.randint(10, 20)
    label_size = random.randint(8, 14)
    
    if random.random() < 0.7:
        ax.set_title(generate_random_string(5, 20), fontsize=title_size,
                    fontweight=random.choice(['normal', 'bold']),
                    pad=random.randint(10, 25))
    
    if random.random() < 0.6:
        ax.set_xlabel(generate_random_string(4, 12), fontsize=label_size)
    
    if random.random() < 0.6:
        ax.set_ylabel(generate_random_string(4, 12), fontsize=label_size)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=random.randint(7, 12))
    ax.tick_params(axis='x', labelsize=random.randint(7, 11))
    
    # Grid variabile
    if random.random() < 0.5:
        ax.grid(axis=random.choice(['x', 'both']), alpha=random.uniform(0.2, 0.6),
               linestyle=random.choice(['-', '--', '-.', ':']),
               color=get_random_color() if random.random() < 0.3 else 'gray')
    
    # Inverti asse Y a volte
    if random.random() < 0.3:
        ax.invert_yaxis()
    
    # Background variabile
    if random.random() < 0.4:
        ax.set_facecolor(get_random_color() if random.random() < 0.3 else 
                        (random.random()*0.1, random.random()*0.1, random.random()*0.1))
    
    if random.random() < 0.3:
        fig.patch.set_facecolor(get_random_color() if random.random() < 0.2 else 
                               (random.random()*0.15, random.random()*0.15, random.random()*0.15))
    
    # Spine style
    if random.random() < 0.3:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=random.randint(80, 150), bbox_inches='tight')
    plt.close()
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from matplotlib import font_manager
import os

def generate_random_string(min_length=3, max_length=12):
    """Genera una stringa casuale"""
    length = random.randint(min_length, max_length)
    if random.random() < 0.3:
        # Genera parole simili a quelle reali
        consonants = 'bcdfghlmnpqrstvz'
        vowels = 'aeiou'
        word = ''
        for i in range(length):
            if i % 2 == 0:
                word += random.choice(consonants)
            else:
                word += random.choice(vowels)
        return word.capitalize()
    else:
        # Genera stringhe alfanumeriche
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))

def generate_data_points(num_points):
    """Genera dati con pattern variabili"""
    pattern_type = random.choice([
        'linear', 'quadratic', 'sine', 'exponential', 
        'random_walk', 'plateau', 'spikes', 'cyclical'
    ])
    
    x = np.linspace(0, 10, num_points)
    
    if pattern_type == 'linear':
        slope = random.uniform(-5, 5)
        intercept = random.uniform(0, 100)
        y = slope * x + intercept
    elif pattern_type == 'quadratic':
        a = random.uniform(-2, 2)
        b = random.uniform(-10, 10)
        c = random.uniform(0, 100)
        y = a * x**2 + b * x + c
    elif pattern_type == 'sine':
        amplitude = random.uniform(10, 50)
        frequency = random.uniform(0.5, 3)
        phase = random.uniform(0, 2*np.pi)
        offset = random.uniform(20, 80)
        y = amplitude * np.sin(frequency * x + phase) + offset
    elif pattern_type == 'exponential':
        base = random.uniform(0.8, 1.3)
        scale = random.uniform(1, 20)
        y = scale * base**x
    elif pattern_type == 'random_walk':
        y = np.cumsum(np.random.randn(num_points)) + random.uniform(20, 80)
    elif pattern_type == 'plateau':
        y = np.concatenate([
            np.linspace(random.uniform(10, 30), random.uniform(60, 90), num_points//3),
            np.ones(num_points//3) * random.uniform(60, 90),
            np.linspace(random.uniform(60, 90), random.uniform(10, 30), num_points - 2*(num_points//3))
        ])
    elif pattern_type == 'spikes':
        y = np.random.uniform(10, 30, num_points)
        spike_indices = random.sample(range(num_points), random.randint(2, 5))
        for idx in spike_indices:
            y[idx] = random.uniform(70, 100)
    else:  # cyclical
        y = 50 + 30*np.sin(2*np.pi*x/5) + 10*np.cos(2*np.pi*x/2)
    
    # Aggiungi rumore
    noise_level = random.uniform(0, 5)
    y += np.random.randn(num_points) * noise_level
    
    # Assicurati che i valori siano positivi
    y = np.maximum(y, 0)
    
    return x, y

def get_random_color():
    """Genera un colore casuale in vari formati"""
    color_format = random.choice(['hex', 'rgb', 'named', 'hsv'])
    
    if color_format == 'hex':
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    elif color_format == 'rgb':
        return (random.random(), random.random(), random.random())
    elif color_format == 'named':
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 
                 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'black',
                 'navy', 'teal', 'lime', 'coral', 'indigo', 'gold']
        return random.choice(colors)
    else:  # hsv
        h = random.random()
        s = random.uniform(0.3, 1.0)
        v = random.uniform(0.3, 1.0)
        from matplotlib.colors import hsv_to_rgb
        return hsv_to_rgb([h, s, v])

def create_area_chart(output_path, chart_index):
    """Crea un area chart con stile completamente randomizzato"""
    
    # Dimensioni variabili
    width = random.uniform(6, 16)
    height = random.uniform(4, 12)
    dpi = random.choice([80, 100, 120, 150, 200, 300])
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    
    # Numero di serie di dati
    num_series = random.randint(1, 5)
    num_points = random.randint(10, 100)
    
    # Genera dati per ogni serie
    series_data = []
    for i in range(num_series):
        x, y = generate_data_points(num_points)
        series_data.append((x, y))
    
    # Stili di riempimento
    fill_style = random.choice(['solid', 'stacked', 'overlapping', 'baseline'])
    
    if fill_style == 'stacked':
        # Area chart impilato
        y_stack = np.zeros(num_points)
        for i, (x, y) in enumerate(series_data):
            color = get_random_color()
            alpha = random.uniform(0.4, 0.9)
            ax.fill_between(x, y_stack, y_stack + y, 
                           color=color, alpha=alpha, 
                           label=generate_random_string())
            y_stack += y
    else:
        # Area chart non impilato
        for i, (x, y) in enumerate(series_data):
            color = get_random_color()
            alpha = random.uniform(0.3, 0.8)
            
            # Varia lo stile della linea
            if random.random() < 0.3:
                linestyle = random.choice(['-', '--', '-.', ':'])
                linewidth = random.uniform(0.5, 3)
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
            
            # Varia il riempimento
            if random.random() < 0.7:
                ax.fill_between(x, y, alpha=alpha, color=color, 
                               label=generate_random_string())
            else:
                baseline = random.uniform(0, min(y))
                ax.fill_between(x, baseline, y, alpha=alpha, color=color,
                               label=generate_random_string())
    
    # Titolo e etichette con font casuali
    available_fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    title_font = random.choice(available_fonts)
    label_font = random.choice(available_fonts)
    
    title_size = random.randint(10, 24)
    label_size = random.randint(8, 16)
    tick_size = random.randint(6, 14)
    
    if random.random() < 0.8:
        ax.set_title(generate_random_string(5, 20), 
                    fontsize=title_size, 
                    family=title_font,
                    weight=random.choice(['normal', 'bold', 'light']),
                    style=random.choice(['normal', 'italic']))
    
    if random.random() < 0.7:
        ax.set_xlabel(generate_random_string(4, 15), 
                     fontsize=label_size, 
                     family=label_font)
    
    if random.random() < 0.7:
        ax.set_ylabel(generate_random_string(4, 15), 
                     fontsize=label_size, 
                     family=label_font)
    
    # Grid casuale
    if random.random() < 0.5:
        grid_style = random.choice(['-', '--', '-.', ':'])
        grid_alpha = random.uniform(0.1, 0.5)
        grid_which = random.choice(['major', 'minor', 'both'])
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha, which=grid_which)
    
    # Legenda
    if random.random() < 0.6 and num_series > 1:
        legend_pos = random.choice(['best', 'upper right', 'upper left', 
                                   'lower right', 'lower left', 'center'])
        legend_cols = random.randint(1, min(3, num_series))
        ax.legend(loc=legend_pos, ncol=legend_cols, 
                 fontsize=random.randint(6, 12),
                 framealpha=random.uniform(0.3, 1.0))
    
    # Colore di sfondo
    if random.random() < 0.4:
        bg_color = get_random_color()
        ax.set_facecolor(bg_color)
        if random.random() < 0.3:
            fig.patch.set_facecolor(get_random_color())
    
    # Bordi degli assi
    if random.random() < 0.3:
        for spine in ax.spines.values():
            spine.set_linewidth(random.uniform(0.5, 3))
            spine.set_color(get_random_color())
    
    # Nascondi assi casualmente
    if random.random() < 0.2:
        ax.spines[random.choice(['top', 'right', 'bottom', 'left'])].set_visible(False)
    
    # Tick params
    ax.tick_params(labelsize=tick_size, 
                   colors=get_random_color() if random.random() < 0.3 else 'black')
    
    # Rotazione tick labels
    if random.random() < 0.4:
        plt.xticks(rotation=random.randint(0, 90))
    
    # Layout
    plt.tight_layout()
    
    # Salva
    plt.savefig(output_path, 
                bbox_inches='tight', 
                facecolor=fig.get_facecolor(),
                edgecolor='none')
    plt.close()
    
    print(f"✓ Grafico {chart_index} creato: {output_path}")

def main():
    # Richiedi il numero di grafici
    try:
        num_charts = int(input("Inserisci il numero di grafici da generare: "))
        if num_charts <= 0:
            print("Il numero deve essere positivo!")
            return
    except ValueError:
        print("Input non valido!")
        return
    
    # Crea cartella output
    output_dir = "area_charts_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 Generazione di {num_charts} area charts...")
    print("=" * 50)
    
    # Genera i grafici
    for i in range(3766, num_charts):
        output_path = os.path.join(output_dir, f"area_chart_{i+1:05d}.png")
        create_area_chart(output_path, i+1)
    
    print("=" * 50)
    print(f"✅ Completato! {num_charts} grafici salvati in '{output_dir}/'")

if __name__ == "__main__":
    main()