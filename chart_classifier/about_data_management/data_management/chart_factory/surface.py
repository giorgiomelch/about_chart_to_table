import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import string
from matplotlib import cm
import os

def generate_random_string(min_length=5, max_length=15):
    """Genera una stringa casuale"""
    length = random.randint(min_length, max_length)
    chars = string.ascii_letters + string.digits + ' '
    return ''.join(random.choice(chars) for _ in range(length))

def generate_random_data_pattern():
    """Genera pattern di dati diversificati per superficie 3D"""
    pattern_type = random.choice([
        'gaussian', 'sinusoidal', 'polynomial', 'exponential', 
        'saddle', 'waves', 'peaks', 'random_smooth', 'ripple',
        'volcano', 'paraboloid', 'hyperboloid'
    ])
    
    # Dimensioni variabili della griglia
    grid_size_x = random.randint(20, 80)
    grid_size_y = random.randint(20, 80)
    
    x = np.linspace(random.uniform(-10, -2), random.uniform(2, 10), grid_size_x)
    y = np.linspace(random.uniform(-10, -2), random.uniform(2, 10), grid_size_y)
    X, Y = np.meshgrid(x, y)
    
    # Parametri casuali
    a = random.uniform(0.1, 3)
    b = random.uniform(0.1, 3)
    c = random.uniform(0.1, 2)
    offset = random.uniform(-5, 5)
    
    if pattern_type == 'gaussian':
        Z = a * np.exp(-(X**2 + Y**2) / (2 * b**2)) + offset
    elif pattern_type == 'sinusoidal':
        Z = a * np.sin(b * X) * np.cos(c * Y) + offset
    elif pattern_type == 'polynomial':
        power = random.randint(2, 4)
        Z = a * (X**power) + b * (Y**power) + c * X * Y + offset
    elif pattern_type == 'exponential':
        Z = a * np.exp(b * (X + Y) / 10) + offset
    elif pattern_type == 'saddle':
        Z = a * (X**2 - Y**2) + offset
    elif pattern_type == 'waves':
        Z = a * np.sin(b * np.sqrt(X**2 + Y**2)) + offset
    elif pattern_type == 'peaks':
        Z = a * (3 * (1-X)**2 * np.exp(-(X**2) - (Y+1)**2) 
                 - 10 * (X/5 - X**3 - Y**5) * np.exp(-X**2-Y**2) 
                 - 1/3 * np.exp(-(X+1)**2 - Y**2)) + offset
    elif pattern_type == 'ripple':
        R = np.sqrt(X**2 + Y**2) + 0.01
        Z = a * np.sin(b * R) / R + offset
    elif pattern_type == 'volcano':
        R = np.sqrt(X**2 + Y**2)
        Z = a * R * np.exp(-b * R) + offset
    elif pattern_type == 'paraboloid':
        Z = a * (X**2 + Y**2) + offset
    elif pattern_type == 'hyperboloid':
        Z = a * (X**2 - Y**2) + b * X * Y + offset
    else:  # random_smooth
        Z = np.random.randn(grid_size_y, grid_size_x) * a + offset
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=random.uniform(1, 5))
    
    # Aggiungi rumore casuale
    if random.random() > 0.5:
        noise_level = random.uniform(0.01, 0.3)
        Z += np.random.randn(*Z.shape) * noise_level * np.std(Z)
    
    return X, Y, Z

def get_random_colormap():
    """Restituisce una colormap casuale"""
    colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper',
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
        'twilight', 'twilight_shifted', 'hsv', 'ocean', 'gist_earth',
        'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
        'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo'
    ]
    return random.choice(colormaps)

def get_random_font():
    """Restituisce proprietà font casuali"""
    fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    sizes = range(8, 18)
    weights = ['normal', 'bold', 'light']
    styles = ['normal', 'italic']
    
    return {
        'family': random.choice(fonts),
        'size': random.choice(sizes),
        'weight': random.choice(weights),
        'style': random.choice(styles)
    }

def create_3d_surface_chart(output_path, index):
    """Crea un singolo surface chart 3D con stile casuale"""
    
    # Genera dati
    X, Y, Z = generate_random_data_pattern()
    
    # Dimensioni figura casuali
    fig_width = random.uniform(8, 16)
    fig_height = random.uniform(6, 12)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # DPI casuale
    dpi = random.choice([80, 100, 120, 150, 200])
    fig.set_dpi(dpi)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Colore di sfondo casuale
    bg_colors = ['white', 'lightgray', 'ivory', 'azure', 'lavender', 
                 'honeydew', 'mintcream', 'aliceblue', 'ghostwhite', 'floralwhite']
    fig.patch.set_facecolor(random.choice(bg_colors))
    ax.set_facecolor(random.choice(bg_colors))
    
    # Colormap casuale
    cmap = get_random_colormap()
    
    # Stile di superficie casuale
    surface_style = random.choice(['surface', 'wireframe', 'surface_with_edges'])
    
    if surface_style == 'surface':
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, 
                               alpha=random.uniform(0.7, 1.0),
                               linewidth=0,
                               antialiased=random.choice([True, False]))
    elif surface_style == 'wireframe':
        surf = ax.plot_wireframe(X, Y, Z, 
                                 color=random.choice(['black', 'blue', 'red', 'green', 'purple']),
                                 linewidth=random.uniform(0.5, 2),
                                 alpha=random.uniform(0.5, 1.0))
    else:  # surface_with_edges
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, 
                               alpha=random.uniform(0.7, 1.0),
                               linewidth=random.uniform(0.1, 1),
                               edgecolor=random.choice(['black', 'white', 'gray', 'none']),
                               antialiased=random.choice([True, False]))
    
    # Colorbar (casuale se mostrarla o no)
    if random.random() > 0.3 and surface_style != 'wireframe':
        cbar = fig.colorbar(surf, shrink=random.uniform(0.4, 0.8), 
                           aspect=random.randint(5, 20),
                           pad=random.uniform(0.05, 0.15))
        cbar_font = get_random_font()
        cbar.ax.tick_params(labelsize=cbar_font['size'])
    
    # Font per titoli e etichette
    title_font = get_random_font()
    label_font = get_random_font()
    tick_font = get_random_font()
    
    # Titolo
    title = generate_random_string(10, 30)
    ax.set_title(title, fontdict=title_font, pad=random.randint(10, 30))
    
    # Etichette assi
    xlabel = generate_random_string(5, 15)
    ylabel = generate_random_string(5, 15)
    zlabel = generate_random_string(5, 15)
    
    ax.set_xlabel(xlabel, fontdict=label_font, labelpad=random.randint(5, 15))
    ax.set_ylabel(ylabel, fontdict=label_font, labelpad=random.randint(5, 15))
    ax.set_zlabel(zlabel, fontdict=label_font, labelpad=random.randint(5, 15))
    
    # Tick font
    ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
    
    # Angolo di vista casuale
    elev = random.uniform(10, 50)
    azim = random.uniform(-180, 180)
    ax.view_init(elev=elev, azim=azim)
    
    # Griglia casuale
    if random.random() > 0.5:
        ax.grid(random.choice([True, False]), 
               alpha=random.uniform(0.2, 0.7),
               linestyle=random.choice(['-', '--', '-.', ':']),
               linewidth=random.uniform(0.5, 1.5))
    
    # Colore assi e piani
    if random.random() > 0.5:
        ax.xaxis.pane.fill = random.choice([True, False])
        ax.yaxis.pane.fill = random.choice([True, False])
        ax.zaxis.pane.fill = random.choice([True, False])
        
        if ax.xaxis.pane.fill:
            ax.xaxis.pane.set_facecolor(random.choice(['lightblue', 'lightgray', 'wheat', 'lightgreen']))
        if ax.yaxis.pane.fill:
            ax.yaxis.pane.set_facecolor(random.choice(['lightblue', 'lightgray', 'wheat', 'lightgreen']))
        if ax.zaxis.pane.fill:
            ax.zaxis.pane.set_facecolor(random.choice(['lightblue', 'lightgray', 'wheat', 'lightgreen']))
    
    # Layout
    plt.tight_layout(pad=random.uniform(0.5, 3))
    
    # Salva
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    
    print(f"Grafico {index + 1} salvato: {output_path}")

def main():
    """Funzione principale"""
    print("=" * 60)
    print("GENERATORE DATASET SURFACE CHART 3D PER ADDESTRAMENTO AI")
    print("=" * 60)
    
    try:
        num_charts = int(input("\nQuanti grafici vuoi generare? "))
        
        if num_charts <= 0:
            print("Il numero deve essere maggiore di 0!")
            return
        
        # Crea cartella output
        output_dir = "dataset_surface_charts_3d"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nGenerazione di {num_charts} surface chart 3D...")
        print(f"I file saranno salvati in: {output_dir}/\n")
        
        # Genera i grafici
        for i in range(3000, num_charts):
            output_path = os.path.join(output_dir, f"surface_chart_3d_{i+1:05d}.png")
            create_3d_surface_chart(output_path, i)
        
        print("\n" + "=" * 60)
        print(f"COMPLETATO! {num_charts} grafici generati con successo!")
        print(f"Percorso: {os.path.abspath(output_dir)}")
        print("=" * 60)
        
    except ValueError:
        print("Errore: inserisci un numero valido!")
    except KeyboardInterrupt:
        print("\n\nGenerazione interrotta dall'utente.")
    except Exception as e:
        print(f"\nErrore durante la generazione: {e}")

if __name__ == "__main__":
    main()