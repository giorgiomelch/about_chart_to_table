import matplotlib.pyplot as plt
import numpy as np
import random
import string
from matplotlib import patches
import os

def random_color():
    """Genera un colore RGB casuale"""
    return (random.random(), random.random(), random.random())

def random_font_properties():
    """Genera proprietà font casuali"""
    fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    sizes = range(8, 16)
    weights = ['normal', 'bold', 'light']
    return {
        'family': random.choice(fonts),
        'size': random.choice(sizes),
        'weight': random.choice(weights)
    }

def generate_data(n_categories, n_samples):
    """Genera dati casuali per violin plot"""
    data = []
    for _ in range(n_categories):
        # Vari tipi di distribuzioni
        dist_type = random.choice(['normal', 'bimodal', 'uniform', 'exponential', 'skewed'])
        
        if dist_type == 'normal':
            d = np.random.normal(random.uniform(-10, 10), random.uniform(0.5, 5), n_samples)
        elif dist_type == 'bimodal':
            d1 = np.random.normal(random.uniform(-5, 0), random.uniform(0.5, 2), n_samples//2)
            d2 = np.random.normal(random.uniform(0, 5), random.uniform(0.5, 2), n_samples//2)
            d = np.concatenate([d1, d2])
        elif dist_type == 'uniform':
            d = np.random.uniform(random.uniform(-10, 0), random.uniform(0, 10), n_samples)
        elif dist_type == 'exponential':
            d = np.random.exponential(random.uniform(1, 3), n_samples)
        else:  # skewed
            d = np.random.gamma(random.uniform(1, 3), random.uniform(1, 2), n_samples)
        
        data.append(d)
    
    return data

def random_label(length=None):
    """Genera etichetta casuale"""
    if length is None:
        length = random.randint(3, 10)
    
    label_type = random.choice(['letters', 'words', 'numbers', 'mixed'])
    
    if label_type == 'letters':
        return ''.join(random.choices(string.ascii_uppercase, k=length))
    elif label_type == 'numbers':
        return f"{random.randint(0, 999)}"
    elif label_type == 'words':
        words = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
                'Group', 'Set', 'Class', 'Type', 'Cat', 'Sample', 'Test', 'Data']
        return random.choice(words) + str(random.randint(1, 20))
    else:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_violin_chart(idx, output_dir='violin_dataset'):
    """Crea un singolo violin chart con variazioni casuali"""
    
    # Parametri casuali
    n_categories = random.randint(1, 8)
    n_samples = random.randint(30, 200)
    horizontal = random.choice([True, False])
    
    # Probabilità di presenza elementi
    has_grid = random.random() > 0.3
    has_legend = random.random() > 0.5 and n_categories > 1
    has_xlabel = random.random() > 0.2
    has_ylabel = random.random() > 0.2
    has_xticks = random.random() > 0.1
    has_yticks = random.random() > 0.1
    has_title = random.random() > 0.4
    
    # Stili violin
    show_means = random.choice([True, False])
    show_medians = random.choice([True, False])
    show_extrema = random.choice([True, False])
    
    # Colori
    bg_color = random_color() if random.random() > 0.7 else 'white'
    
    # Crea figura
    fig_width = random.uniform(6, 14)
    fig_height = random.uniform(4, 10)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(random_color() if random.random() > 0.8 else 'white')
    
    # Genera dati
    data = generate_data(n_categories, n_samples)
    labels = [random_label() for _ in range(n_categories)]
    
    # Colori per violin
    colors = [random_color() for _ in range(n_categories)]
    
    # Crea violin plot
    positions = list(range(1, n_categories + 1))
    
    parts = ax.violinplot(
        data,
        positions=positions,
        vert=not horizontal,
        showmeans=show_means,
        showmedians=show_medians,
        showextrema=show_extrema,
        widths=random.uniform(0.3, 0.9)
    )
    
    # Personalizza colori dei violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(random.uniform(0.4, 0.9))
        pc.set_edgecolor(random_color() if random.random() > 0.5 else 'black')
        pc.set_linewidth(random.uniform(0.5, 2.5))
    
    # Personalizza elementi statistici
    for key in ['cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins']:
        if key in parts:
            parts[key].set_edgecolor(random_color())
            parts[key].set_linewidth(random.uniform(0.5, 2))
    
    # Font properties
    title_font = random_font_properties()
    label_font = random_font_properties()
    tick_font = random_font_properties()
    
    # Titolo
    if has_title:
        ax.set_title(random_label(random.randint(10, 25)), 
                    fontdict=title_font, 
                    color=random_color(),
                    pad=random.randint(10, 30))
    
    # Etichette assi
    if has_xlabel:
        xlabel = random_label(random.randint(5, 15))
        if horizontal:
            ax.set_xlabel(xlabel, fontdict=label_font, color=random_color())
        else:
            ax.set_xlabel(xlabel, fontdict=label_font, color=random_color())
    
    if has_ylabel:
        ylabel = random_label(random.randint(5, 15))
        if horizontal:
            ax.set_ylabel(ylabel, fontdict=label_font, color=random_color())
        else:
            ax.set_ylabel(ylabel, fontdict=label_font, color=random_color())
    
    # Ticks
    if has_xticks:
        if horizontal:
            ax.tick_params(axis='x', labelsize=tick_font['size'], 
                          labelcolor=random_color(), rotation=random.randint(0, 45))
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontdict=tick_font, 
                              color=random_color(), rotation=random.randint(0, 90))
    else:
        ax.set_xticks([])
    
    if has_yticks:
        if horizontal:
            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontdict=tick_font, 
                              color=random_color(), rotation=random.randint(0, 45))
        else:
            ax.tick_params(axis='y', labelsize=tick_font['size'], 
                          labelcolor=random_color())
    else:
        ax.set_yticks([])
    
    # Griglia
    if has_grid:
        grid_style = random.choice(['-', '--', '-.', ':'])
        grid_alpha = random.uniform(0.2, 0.7)
        grid_color = random_color()
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha, 
               color=grid_color, linewidth=random.uniform(0.3, 1.5),
               axis=random.choice(['both', 'x', 'y']))
    
    # Legenda
    if has_legend and n_categories > 1:
        legend_elements = [patches.Patch(facecolor=colors[i], 
                                        edgecolor='black', 
                                        label=labels[i]) 
                          for i in range(n_categories)]
        legend_position = random.choice(['upper right', 'upper left', 'lower right', 
                                        'lower left', 'center', 'best'])
        legend_font = random_font_properties()
        ax.legend(handles=legend_elements, 
                 loc=legend_position,
                 fontsize=legend_font['size'],
                 framealpha=random.uniform(0.3, 1.0),
                 facecolor=random_color() if random.random() > 0.7 else 'white',
                 edgecolor=random_color())
    
    # Bordi
    spine_colors = [random_color() for _ in range(4)]
    spine_widths = [random.uniform(0.5, 3) for _ in range(4)]
    spines = ['top', 'right', 'bottom', 'left']
    
    for i, spine in enumerate(spines):
        if random.random() > 0.2:
            ax.spines[spine].set_color(spine_colors[i])
            ax.spines[spine].set_linewidth(spine_widths[i])
        else:
            ax.spines[spine].set_visible(False)
    
    # Salva
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.tight_layout()
    filename = f"{output_dir}/violin_{idx:05d}.png"
    plt.savefig(filename, dpi=random.randint(80, 150), 
               facecolor=fig.get_facecolor(),
               bbox_inches='tight')
    plt.close()
    
    return filename

def generate_dataset(n_charts):
    """Genera un dataset di n_charts violin charts"""
    print(f"Generazione di {n_charts} violin charts...")
    
    for i in range(0, n_charts):
        filename = create_violin_chart(i)
        if (i + 1) % 10 == 0:
            print(f"Generati {i + 1}/{n_charts} grafici")
    
    print(f"\nDataset completato! {n_charts} grafici salvati in 'violin_dataset/'")

if __name__ == "__main__":
    n = int(input("Inserisci il numero di violin charts da generare: "))
    generate_dataset(n)