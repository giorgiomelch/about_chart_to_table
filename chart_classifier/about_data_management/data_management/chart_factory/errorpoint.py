import matplotlib.pyplot as plt
import numpy as np
import random
import string
from matplotlib import font_manager
import os

def random_string(length_range=(3, 12)):
    """Genera una stringa random"""
    length = random.randint(*length_range)
    chars = string.ascii_letters + string.digits + ' '
    return ''.join(random.choice(chars) for _ in range(length)).strip()

def random_color():
    """Genera un colore random in vari formati"""
    formats = [
        lambda: f'#{random.randint(0, 0xFFFFFF):06x}',  # hex
        lambda: (random.random(), random.random(), random.random()),  # RGB
        lambda: random.choice(['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown'])
    ]
    return random.choice(formats)()

def generate_data_pattern():
    """Genera pattern di dati diversi"""
    n_points = random.randint(5, 30)
    patterns = [
        lambda: np.random.randn(n_points) * random.uniform(5, 50) + random.uniform(-100, 100),  # random normale
        lambda: np.linspace(random.uniform(-50, 0), random.uniform(50, 200), n_points),  # lineare
        lambda: np.sin(np.linspace(0, random.uniform(2, 10), n_points)) * random.uniform(10, 100),  # sinusoidale
        lambda: np.exp(np.linspace(random.uniform(-2, 0), random.uniform(0.5, 3), n_points)),  # esponenziale
        lambda: np.random.exponential(random.uniform(5, 50), n_points),  # esponenziale random
        lambda: np.random.gamma(random.uniform(1, 5), random.uniform(1, 10), n_points),  # gamma
        lambda: np.cumsum(np.random.randn(n_points)) * random.uniform(1, 10),  # random walk
        lambda: np.logspace(random.uniform(-1, 0), random.uniform(1, 3), n_points),  # logaritmico
    ]
    return random.choice(patterns)(), n_points

def generate_error_bars(n_points):
    """Genera error bars con pattern diversi"""
    error_types = [
        lambda: np.random.uniform(0.5, 5, n_points),  # uniforme
        lambda: np.abs(np.random.randn(n_points)) * random.uniform(1, 10),  # normale assoluto
        lambda: np.linspace(random.uniform(0.1, 2), random.uniform(3, 10), n_points),  # crescente
        lambda: np.ones(n_points) * random.uniform(1, 8),  # costante
        lambda: np.random.exponential(random.uniform(1, 5), n_points),  # esponenziale
    ]
    return random.choice(error_types)()

def create_error_bar_plot(idx, output_dir='dataset_grafici'):
    """Crea un singolo grafico con error bars"""
    
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Genera dati
    y_data, n_points = generate_data_pattern()
    x_data = np.arange(n_points) if random.random() > 0.3 else np.sort(np.random.uniform(0, 100, n_points))
    
    # Genera errori
    orientation = random.choice(['vertical', 'horizontal', 'both'])
    
    if orientation == 'vertical':
        yerr = generate_error_bars(n_points)
        xerr = None
    elif orientation == 'horizontal':
        xerr = generate_error_bars(n_points)
        yerr = None
    else:
        yerr = generate_error_bars(n_points)
        xerr = generate_error_bars(n_points)
    
    # Configurazione figura
    fig_size = (random.uniform(6, 14), random.uniform(5, 10))
    fig, ax = plt.subplots(figsize=fig_size, dpi=random.choice([80, 100, 150, 200]))
    
    # Stili di sfondo
    bg_styles = ['white', 'lightgray', '#f0f0f0', '#e8e8e8', 'whitesmoke']
    ax.set_facecolor(random.choice(bg_styles))
    fig.patch.set_facecolor(random.choice(bg_styles))
    
    # Marker styles
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+', '.', 'd', '|', '_']
    marker = random.choice(markers)
    markersize = random.uniform(3, 12)
    
    # Line styles
    line_styles = ['-', '--', '-.', ':', '', None]
    linestyle = random.choice(line_styles)
    linewidth = random.uniform(0.5, 3)
    
    # Colori
    color_main = random_color()
    error_color = random_color() if random.random() > 0.5 else color_main
    
    # Capsize e capthick
    capsize = random.uniform(0, 8)
    capthick = random.uniform(0.5, 3)
    elinewidth = random.uniform(0.5, 3)
    
    # Alpha
    alpha = random.uniform(0.4, 1.0)
    
    # Plot error bars
    ax.errorbar(x_data, y_data, yerr=yerr, xerr=xerr,
                fmt=marker if linestyle in ['', None] else linestyle,
                color=color_main,
                ecolor=error_color,
                elinewidth=elinewidth,
                capsize=capsize,
                capthick=capthick,
                markersize=markersize,
                linewidth=linewidth,
                alpha=alpha,
                label=random_string() if random.random() > 0.3 else None)
    
    # Titoli e labels
    fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    title_font = random.choice(fonts)
    label_font = random.choice(fonts)
    
    font_sizes = {
        'title': random.randint(10, 20),
        'label': random.randint(8, 16),
        'tick': random.randint(7, 14)
    }
    
    if random.random() > 0.2:
        ax.set_title(random_string((5, 25)), fontsize=font_sizes['title'], 
                     family=title_font, weight=random.choice(['normal', 'bold']))
    
    if random.random() > 0.2:
        ax.set_xlabel(random_string((3, 15)), fontsize=font_sizes['label'], 
                      family=label_font, style=random.choice(['normal', 'italic']))
    
    if random.random() > 0.2:
        ax.set_ylabel(random_string((3, 15)), fontsize=font_sizes['label'], 
                      family=label_font, style=random.choice(['normal', 'italic']))
    
    # Griglia
    if random.random() > 0.4:
        grid_alpha = random.uniform(0.2, 0.7)
        grid_style = random.choice(['-', '--', '-.', ':'])
        grid_which = random.choice(['major', 'minor', 'both'])
        ax.grid(True, alpha=grid_alpha, linestyle=grid_style, which=grid_which,
                color=random_color())
    
    # Legenda
    if random.random() > 0.5 and ax.get_legend_handles_labels()[1]:
        legend_loc = random.choice(['best', 'upper right', 'upper left', 'lower right', 
                                   'lower left', 'center', 'upper center', 'lower center'])
        ax.legend(loc=legend_loc, fontsize=random.randint(7, 12),
                 framealpha=random.uniform(0.5, 1))
    
    # Spine visibility
    if random.random() > 0.7:
        for spine in random.sample(['top', 'right', 'bottom', 'left'], k=random.randint(1, 2)):
            ax.spines[spine].set_visible(False)
    
    # Tick params
    ax.tick_params(labelsize=font_sizes['tick'], 
                   length=random.uniform(2, 8),
                   width=random.uniform(0.5, 2))
    
    # Limiti assi (a volte)
    if random.random() > 0.6:
        margin = random.uniform(0.05, 0.2)
        ax.margins(margin)
    
    # Layout
    plt.tight_layout()
    
    # Salva
    filename = f'{output_dir}/error_plot_{idx:05d}.png'
    plt.savefig(filename, dpi=random.choice([100, 150, 200]), 
                bbox_inches='tight', facecolor=fig.patch.get_facecolor())
    plt.close()
    
    return filename

def generate_dataset(num_plots):
    """Genera il dataset completo"""
    print(f"Generazione di {num_plots} grafici in corso...")
    
    for i in range(2643, num_plots):
        filename = create_error_bar_plot(i)
        if (i + 1) % 10 == 0 or i == num_plots - 1:
            print(f"Generati {i + 1}/{num_plots} grafici")
    
    print(f"\nDataset completato! {num_plots} grafici salvati nella cartella 'dataset_grafici'")

if __name__ == "__main__":
    try:
        num_grafici = int(input("Inserisci il numero di grafici da generare: "))
        if num_grafici <= 0:
            print("Errore: inserisci un numero positivo")
        else:
            generate_dataset(num_grafici)
    except ValueError:
        print("Errore: inserisci un numero valido")
    except KeyboardInterrupt:
        print("\nGenerazione interrotta dall'utente")