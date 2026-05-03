import matplotlib.pyplot as plt
import numpy as np
import random
import os
from matplotlib import cm
import matplotlib.patches as mpatches

def random_color():
    """Genera un colore casuale"""
    return (random.random(), random.random(), random.random(), random.uniform(0.6, 0.9))

def random_colormap():
    """Seleziona una colormap casuale"""
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 
             'Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrRd', 
             'RdYlBu', 'Spectral', 'coolwarm', 'rainbow', 'jet', 'seismic',
             'spring', 'summer', 'autumn', 'winter', 'copper', 'hot', 'cool']
    return random.choice(cmaps)

def random_word():
    """Genera una parola casuale"""
    consonants = 'bcdfghlmnpqrstvwxz'
    vowels = 'aeiou'
    length = random.randint(3, 10)
    word = ''
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(consonants)
        else:
            word += random.choice(vowels)
    return word.capitalize()

def random_text(min_words=1, max_words=4):
    """Genera testo casuale con più parole"""
    n_words = random.randint(min_words, max_words)
    return ' '.join([random_word() for _ in range(n_words)])

def random_number_text():
    """Genera testo con numeri casuali"""
    patterns = [
        lambda: f"{random_word()} {random.randint(1, 999)}",
        lambda: f"{random.choice(['Type', 'Class', 'Group', 'Set'])} {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
        lambda: f"{random_word()}-{random.randint(10, 99)}",
        lambda: f"{random.choice(['v', 'Ver', 'R'])}{random.randint(1, 20)}.{random.randint(0, 9)}",
        lambda: random_word()
    ]
    return random.choice(patterns)()

def generate_data_distribution(n_points, dist_type):
    """Genera dati secondo diverse distribuzioni"""
    distributions = {
        0: lambda: (np.random.uniform(0, 100, n_points), np.random.uniform(0, 100, n_points)),
        1: lambda: (np.random.normal(50, 20, n_points), np.random.normal(50, 20, n_points)),
        2: lambda: (np.random.exponential(20, n_points), np.random.exponential(20, n_points)),
        3: lambda: (np.random.gamma(2, 15, n_points), np.random.gamma(2, 15, n_points)),
        4: lambda: (np.random.beta(2, 5, n_points) * 100, np.random.beta(2, 5, n_points) * 100),
        5: lambda: (np.random.lognormal(3, 1, n_points), np.random.lognormal(3, 1, n_points)),
        6: lambda: (np.random.poisson(30, n_points).astype(float), np.random.poisson(30, n_points).astype(float)),
        7: lambda: (np.random.triangular(0, 50, 100, n_points), np.random.triangular(0, 50, 100, n_points)),
        8: lambda: (np.random.weibull(1.5, n_points) * 30, np.random.weibull(1.5, n_points) * 30),
        9: lambda: (np.linspace(0, 100, n_points) + np.random.normal(0, 10, n_points), 
                   np.linspace(0, 100, n_points) + np.random.normal(0, 10, n_points)),
        10: lambda: (np.random.uniform(0, 100, n_points), 
                    50 + 30 * np.sin(np.random.uniform(0, 100, n_points) * 0.1) + np.random.normal(0, 5, n_points)),
        11: lambda: (np.random.uniform(0, 100, n_points), 
                    (np.random.uniform(0, 100, n_points) ** 2) / 100 + np.random.normal(0, 5, n_points)),
        12: lambda: (np.random.choice([25, 75], n_points) + np.random.normal(0, 8, n_points),
                    np.random.choice([25, 75], n_points) + np.random.normal(0, 8, n_points)),
        13: lambda: (np.random.choice([20, 50, 80], n_points) + np.random.normal(0, 5, n_points),
                    np.random.choice([20, 50, 80], n_points) + np.random.normal(0, 5, n_points)),
        14: lambda: (np.random.uniform(0, 100, n_points), np.random.uniform(0, 20, n_points) + 
                    np.where(np.random.uniform(0, 100, n_points) > 50, 60, 0)),
        15: lambda: (np.sqrt(np.random.uniform(0, 10000, n_points)), np.sqrt(np.random.uniform(0, 10000, n_points))),
        16: lambda: (np.abs(np.random.normal(50, 25, n_points)), np.abs(np.random.normal(50, 25, n_points))),
        17: lambda: (np.random.chisquare(5, n_points) * 5, np.random.chisquare(5, n_points) * 5),
        18: lambda: (np.random.laplace(50, 15, n_points), np.random.laplace(50, 15, n_points)),
        19: lambda: (np.random.pareto(3, n_points) * 20, np.random.pareto(3, n_points) * 20),
    }
    
    x, y = distributions[dist_type]()
    return np.clip(x, 0, 100), np.clip(y, 0, 100)

def generate_bubble_chart(save_path, idx):
    """Genera un singolo bubble chart con variazioni casuali"""
    
    # Parametri di randomizzazione
    has_legend = random.random() > 0.15
    has_grid = random.random() > 0.25
    has_xlabel = random.random() > 0.1
    has_ylabel = random.random() > 0.1
    has_title = random.random() > 0.15
    has_xticks = random.random() > 0.1
    has_yticks = random.random() > 0.1
    has_box = random.random() > 0.2
    
    # Multi-categoria o singola con gradiente
    is_multi_category = random.random() > 0.5
    n_categories = random.randint(2, 6) if is_multi_category else 1
    
    # Numero di bolle per categoria
    n_bubbles_per_cat = random.randint(5, 30)
    
    # Distribuzione dei dati
    distribution_type = random.randint(0, 19)
    
    # Stile figura
    fig_width = random.uniform(8, 14)
    fig_height = random.uniform(6, 10)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Background color con più varietà
    bg_colors = ['white', 'whitesmoke', 'lightgray', 'floralwhite', 'ivory', 
                 'honeydew', 'azure', 'lavenderblush', 'mistyrose', 'seashell',
                 'mintcream', 'ghostwhite', 'aliceblue', 'snow', 'linen']
    if random.random() > 0.3:
        fig.patch.set_facecolor(random.choice(bg_colors))
    
    ax = fig.add_subplot(111)
    if random.random() > 0.3:
        ax.set_facecolor(random.choice(bg_colors))
    
    # Font settings
    fonts = ['serif', 'sans-serif', 'monospace']
    font_family = random.choice(fonts)
    font_size_title = random.randint(10, 20)
    font_size_labels = random.randint(8, 14)
    font_size_ticks = random.randint(7, 12)
    
    # Forma: sempre cerchio per bubble chart
    marker = 'o'
    
    # Genera dati
    for cat in range(n_categories):
        x, y = generate_data_distribution(n_bubbles_per_cat, distribution_type)
        
        actual_n_points = len(x)
        
        # Dimensioni bolle con MOLTA varietà e varianza
        size_strategy = random.choice(['random_varied', 'random_varied', 'random_varied', 
                                       'gradient_varied', 'exponential', 'bimodal', 'clusters'])
        
        if size_strategy == 'random_varied':
            size_multiplier = random.uniform(30, 200)
            min_size = random.uniform(10, 50)
            max_size = random.uniform(200, 800)
            sizes = np.random.uniform(min_size, max_size, actual_n_points) * (size_multiplier / 100)
            
        elif size_strategy == 'gradient_varied':
            min_size = random.uniform(50, 150)
            max_size = random.uniform(300, 900)
            sizes = np.linspace(min_size, max_size, actual_n_points)
            np.random.shuffle(sizes)
            noise = np.random.normal(0, (max_size - min_size) * 0.15, actual_n_points)
            sizes = np.clip(sizes + noise, min_size, max_size)
            
        elif size_strategy == 'exponential':
            sizes = np.random.exponential(150, actual_n_points) + 50
            sizes = np.clip(sizes, 50, 800)
            
        elif size_strategy == 'bimodal':
            small_size = random.uniform(50, 150)
            large_size = random.uniform(300, 700)
            sizes = np.where(np.random.random(actual_n_points) > 0.5, large_size, small_size)
            sizes += np.random.normal(0, 40, actual_n_points)
            sizes = np.clip(sizes, 30, 800)
            
        elif size_strategy == 'clusters':
            n_clusters = random.randint(3, 4)
            cluster_sizes = [random.uniform(80, 600) for _ in range(n_clusters)]
            sizes = np.array([random.choice(cluster_sizes) for _ in range(actual_n_points)])
            sizes += np.random.normal(0, 30, actual_n_points)
            sizes = np.clip(sizes, 40, 800)
            
        else:
            size_multiplier = random.uniform(30, 200)
            min_size = random.uniform(10, 50)
            max_size = random.uniform(200, 800)
            sizes = np.random.uniform(min_size, max_size, actual_n_points) * (size_multiplier / 100)
        
        # Colori con più strategie
        if is_multi_category:
            color_strategy = random.choice(['solid', 'gradient', 'random'])
            
            if color_strategy == 'solid':
                base_color = random_color()
                colors = [base_color] * actual_n_points
            elif color_strategy == 'gradient':
                cmap = cm.get_cmap(random_colormap())
                norm_values = np.linspace(0, 1, actual_n_points)
                colors = [cmap(v) for v in norm_values]
            else:
                colors = [random_color() for _ in range(actual_n_points)]
        else:
            gradient_type = random.choice(['x', 'y', 'distance', 'angle', 'random'])
            cmap = cm.get_cmap(random_colormap())
            
            if gradient_type == 'x':
                norm_values = (x - x.min()) / (x.max() - x.min() + 0.001)
            elif gradient_type == 'y':
                norm_values = (y - y.min()) / (y.max() - y.min() + 0.001)
            elif gradient_type == 'distance':
                distances = np.sqrt((x - 50)**2 + (y - 50)**2)
                norm_values = (distances - distances.min()) / (distances.max() - distances.min() + 0.001)
            elif gradient_type == 'angle':
                angles = np.arctan2(y - 50, x - 50)
                norm_values = (angles - angles.min()) / (angles.max() - angles.min() + 0.001)
            else:
                norm_values = np.random.random(actual_n_points)
                
            colors = [cmap(v) for v in norm_values]
        
        # Edge settings con più varietà
        edge_style = random.choice(['none', 'thin', 'medium', 'thick', 'random'])
        
        if edge_style == 'none':
            edge_width = 0
            edge_color = 'none'
        elif edge_style == 'thin':
            edge_width = random.uniform(0.3, 0.8)
            edge_color = random.choice(['black', 'white', 'gray'])
        elif edge_style == 'medium':
            edge_width = random.uniform(0.8, 1.5)
            edge_color = random.choice(['black', 'white', 'gray', random_color()])
        elif edge_style == 'thick':
            edge_width = random.uniform(1.5, 3)
            edge_color = random.choice(['black', 'white'])
        else:
            edge_width = random.uniform(0, 2.5)
            edge_color = random_color() if random.random() > 0.5 else 'black'
        
        # Alpha con più varietà
        alpha = random.uniform(0.4, 0.95)
        
        # Label completamente random
        label = random_number_text() if is_multi_category and has_legend else None
        
        ax.scatter(x, y, s=sizes, c=colors, alpha=alpha, 
                  edgecolors=edge_color, linewidths=edge_width,
                  marker=marker, label=label)
    
    # Grid con più stili
    if has_grid:
        grid_styles = ['-', '--', '-.', ':']
        grid_alpha = random.uniform(0.2, 0.7)
        grid_linewidth = random.uniform(0.5, 2)
        grid_color = random.choice(['gray', 'black', 'lightgray', random_color()])
        ax.grid(True, linestyle=random.choice(grid_styles), 
               alpha=grid_alpha, linewidth=grid_linewidth, color=grid_color)
        
        if random.random() > 0.7:
            ax.grid(axis=random.choice(['x', 'y']))
    
    # Labels completamente random
    if has_xlabel:
        ax.set_xlabel(random_text(1, 3), fontsize=font_size_labels, 
                     family=font_family, weight=random.choice(['normal', 'bold']),
                     style=random.choice(['normal', 'italic']))
    
    if has_ylabel:
        ax.set_ylabel(random_text(1, 3), fontsize=font_size_labels, 
                     family=font_family, weight=random.choice(['normal', 'bold']),
                     style=random.choice(['normal', 'italic']))
    
    # Title completamente random
    if has_title:
        ax.set_title(random_text(2, 5), fontsize=font_size_title, 
                    family=font_family, weight=random.choice(['normal', 'bold']),
                    style=random.choice(['normal', 'italic']),
                    pad=random.uniform(10, 25))
    
    # Ticks
    if not has_xticks:
        ax.set_xticks([])
    else:
        ax.tick_params(axis='x', labelsize=font_size_ticks, 
                      length=random.uniform(2, 8),
                      width=random.uniform(0.5, 2))
        if random.random() > 0.5:
            plt.xticks(rotation=random.randint(0, 90))
    
    if not has_yticks:
        ax.set_yticks([])
    else:
        ax.tick_params(axis='y', labelsize=font_size_ticks,
                      length=random.uniform(2, 8),
                      width=random.uniform(0.5, 2))
    
    # Box/Spines con più varietà
    if not has_box:
        if random.random() > 0.5:
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            spines_to_remove = random.sample(list(ax.spines.keys()), k=random.randint(1, 3))
            for spine_name in spines_to_remove:
                ax.spines[spine_name].set_visible(False)
    else:
        spine_width = random.uniform(0.5, 3)
        spine_color = random.choice(['black', 'gray', 'darkgray', random_color()])
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
            spine.set_edgecolor(spine_color)
    
    # Legend con più varietà
    if has_legend and is_multi_category:
        legend_positions = ['best', 'upper right', 'upper left', 'lower right', 
                          'lower left', 'center right', 'center left', 'upper center', 'lower center']
        ax.legend(loc=random.choice(legend_positions), 
                 fontsize=random.randint(7, 12),
                 framealpha=random.uniform(0.5, 1.0),
                 edgecolor=random.choice(['black', 'gray', 'none']),
                 fancybox=random.random() > 0.5,
                 shadow=random.random() > 0.7,
                 ncol=random.choice([1, 1, 1, 2]))
    
    # Limiti assi con più varietà
    margin_strategy = random.choice(['tight', 'normal', 'loose', 'asymmetric'])
    
    if margin_strategy == 'tight':
        margin = random.uniform(0, 5)
        ax.set_xlim(-margin, 100 + margin)
        ax.set_ylim(-margin, 100 + margin)
    elif margin_strategy == 'normal':
        margin = random.uniform(5, 15)
        ax.set_xlim(-margin, 100 + margin)
        ax.set_ylim(-margin, 100 + margin)
    elif margin_strategy == 'loose':
        margin = random.uniform(15, 25)
        ax.set_xlim(-margin, 100 + margin)
        ax.set_ylim(-margin, 100 + margin)
    else:
        ax.set_xlim(-random.uniform(0, 20), 100 + random.uniform(0, 20))
        ax.set_ylim(-random.uniform(0, 20), 100 + random.uniform(0, 20))
    
    # Aspect ratio
    if random.random() > 0.5:
        ax.set_aspect(random.choice(['equal', 'auto']))
    
    # Tight layout
    if random.random() > 0.3:
        plt.tight_layout()
    
    # Salva
    plt.savefig(f'{save_path}/bubble_chart_{idx:04d}.png', 
                dpi=random.randint(80, 150), 
                bbox_inches='tight' if random.random() > 0.5 else None,
                facecolor=fig.get_facecolor())
    plt.close()

def generate_dataset(n_charts, output_dir='bubble_charts_dataset'):
    """Genera un dataset di bubble charts"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generazione di {n_charts} bubble charts...")
    print(f"Salvando in: {output_dir}/")
    
    for i in range(0, n_charts):
        try:
            generate_bubble_chart(output_dir, i)
            if (i + 1) % 10 == 0:
                print(f"Generati {i + 1}/{n_charts} grafici...")
        except Exception as e:
            print(f"Errore nel grafico {i}: {e}")
            continue
    
    print(f"\n✓ Dataset completato! {n_charts} grafici salvati in '{output_dir}/'")

# Esecuzione
if __name__ == "__main__":
    n = int(input("Quanti bubble charts vuoi generare? "))
    generate_dataset(n)