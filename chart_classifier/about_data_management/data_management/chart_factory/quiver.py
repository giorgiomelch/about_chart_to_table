import matplotlib.pyplot as plt
import numpy as np
import random
import string
from matplotlib import cm
import os

def generate_random_string(min_len=3, max_len=12):
    """Genera una stringa random di lunghezza variabile"""
    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits + ' '
    return ''.join(random.choice(chars) for _ in range(length))

def get_random_colormap():
    """Restituisce una colormap casuale"""
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'twilight', 'turbo', 'jet', 'rainbow', 'gist_rainbow',
                 'cool', 'hot', 'spring', 'summer', 'autumn', 'winter',
                 'coolwarm', 'bwr', 'seismic', 'RdYlBu', 'RdYlGn', 'Spectral']
    return random.choice(colormaps)

def get_random_font():
    """Restituisce proprietà font casuali"""
    fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    sizes = range(8, 20)
    weights = ['normal', 'bold', 'light']
    styles = ['normal', 'italic']
    
    return {
        'family': random.choice(fonts),
        'size': random.choice(sizes),
        'weight': random.choice(weights),
        'style': random.choice(styles)
    }

def generate_vector_field(field_type):
    """Genera diversi tipi di campi vettoriali"""
    x = np.linspace(-3, 3, random.randint(8, 20))
    y = np.linspace(-3, 3, random.randint(8, 20))
    X, Y = np.meshgrid(x, y)
    
    if field_type == 'rotation':
        U = -Y
        V = X
    elif field_type == 'radial':
        U = X
        V = Y
    elif field_type == 'saddle':
        U = X
        V = -Y
    elif field_type == 'spiral':
        U = -Y + 0.3 * X
        V = X + 0.3 * Y
    elif field_type == 'wave':
        U = np.sin(X) * np.cos(Y)
        V = np.cos(X) * np.sin(Y)
    elif field_type == 'vortex':
        R = np.sqrt(X**2 + Y**2) + 0.1
        U = -Y / R
        V = X / R
    elif field_type == 'shear':
        U = Y
        V = np.zeros_like(X)
    elif field_type == 'random':
        U = np.random.randn(*X.shape)
        V = np.random.randn(*X.shape)
    elif field_type == 'circular':
        theta = np.arctan2(Y, X)
        U = -np.sin(theta)
        V = np.cos(theta)
    else:
        U = np.sin(X) * np.cos(Y) + np.random.randn(*X.shape) * 0.1
        V = np.cos(X) * np.sin(Y) + np.random.randn(*X.shape) * 0.1
    
    return X, Y, U, V

def create_quiver_plot(index, output_dir='quiver_dataset'):
    """Crea un singolo quiver plot con stile casuale"""
    
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Parametri casuali per la figura
    fig_width = random.uniform(6, 12)
    fig_height = random.uniform(5, 10)
    dpi = random.choice([80, 100, 150, 200, 300])
    
    # Crea figura con dimensioni casuali
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Genera campo vettoriale casuale
    field_types = ['rotation', 'radial', 'saddle', 'spiral', 'wave', 
                   'vortex', 'shear', 'random', 'circular', 'complex']
    X, Y, U, V = generate_vector_field(random.choice(field_types))
    
    # Colormap e stili casuali
    cmap = get_random_colormap()
    
    # Calcola magnitudine per il colore
    M = np.sqrt(U**2 + V**2)
    
    # Parametri quiver casuali
    scale = random.uniform(5, 50) if random.random() > 0.3 else None
    width = random.uniform(0.002, 0.01)
    headwidth = random.uniform(2, 6)
    headlength = random.uniform(3, 7)
    headaxislength = random.uniform(2.5, 6)
    
    # Stili freccia
    arrow_styles = [
        {'scale': scale, 'width': width},
        {'scale': scale, 'width': width, 'headwidth': headwidth},
        {'scale': scale, 'width': width, 'headwidth': headwidth, 'headlength': headlength},
        {'width': width, 'headwidth': headwidth, 'headlength': headlength, 'headaxislength': headaxislength}
    ]
    arrow_params = random.choice(arrow_styles)
    
    # Crea quiver plot
    if random.random() > 0.5:
        # Con colore basato su magnitudine
        Q = ax.quiver(X, Y, U, V, M, cmap=cmap, **arrow_params)
        if random.random() > 0.6:
            cbar = plt.colorbar(Q, ax=ax)
            cbar_font = get_random_font()
            cbar.ax.tick_params(labelsize=cbar_font['size'])
            cbar.set_label(generate_random_string(5, 15), fontdict=cbar_font)
    else:
        # Colore singolo casuale
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black']
        Q = ax.quiver(X, Y, U, V, color=random.choice(colors), **arrow_params)
    
    # Aggiungi sfondo casuale
    if random.random() > 0.5:
        # Contour plot di sfondo
        levels = random.randint(5, 25)
        if random.random() > 0.5:
            ax.contourf(X, Y, M, levels=levels, cmap=get_random_colormap(), alpha=random.uniform(0.2, 0.6))
        else:
            ax.contour(X, Y, M, levels=levels, cmap=get_random_colormap(), alpha=random.uniform(0.3, 0.7))
    
    # Griglia casuale
    if random.random() > 0.4:
        grid_styles = [
            {'linestyle': '--', 'alpha': 0.5},
            {'linestyle': ':', 'alpha': 0.7},
            {'linestyle': '-', 'alpha': 0.3},
            {'linestyle': '-.', 'alpha': 0.6}
        ]
        ax.grid(True, **random.choice(grid_styles))
    
    # Font casuali per titolo e assi
    title_font = get_random_font()
    xlabel_font = get_random_font()
    ylabel_font = get_random_font()
    
    # Titoli e label con stringhe casuali
    ax.set_title(generate_random_string(10, 30), fontdict=title_font, pad=random.uniform(10, 25))
    ax.set_xlabel(generate_random_string(5, 20), fontdict=xlabel_font)
    ax.set_ylabel(generate_random_string(5, 20), fontdict=ylabel_font)
    
    # Tick labels casuali
    tick_font_size = random.randint(7, 14)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    
    # Colori di sfondo
    bg_colors = ['white', 'lightgray', 'lavender', 'honeydew', 'azure', 'ivory', 'beige']
    if random.random() > 0.7:
        ax.set_facecolor(random.choice(bg_colors))
    
    # Bordi
    if random.random() > 0.5:
        for spine in ax.spines.values():
            spine.set_linewidth(random.uniform(0.5, 3))
            spine.set_edgecolor(random.choice(['black', 'gray', 'blue', 'red']))
    
    # Aspect ratio
    if random.random() > 0.5:
        ax.set_aspect(random.choice(['equal', 'auto']))
    
    # Layout
    plt.tight_layout()
    
    # Salva
    filename = os.path.join(output_dir, f'quiver_plot_{index:05d}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=dpi, 
                facecolor=random.choice(['white', 'lightgray', None]))
    plt.close()
    
    print(f"Creato: {filename}")

def main():
    """Funzione principale"""
    print("=" * 60)
    print("GENERATORE DATASET QUIVER PLOTS PER AI TRAINING")
    print("=" * 60)
    
    try:
        num_plots = int(input("\nQuanti grafici vuoi generare? "))
        
        if num_plots <= 0:
            print("Errore: Inserisci un numero positivo!")
            return
        
        print(f"\nGenerazione di {num_plots} quiver plots in corso...")
        print("-" * 60)
        
        for i in range(3730 , num_plots):
            create_quiver_plot(i)
            if (i + 1) % 10 == 0:
                print(f"Progresso: {i + 1}/{num_plots} grafici completati")
        
        print("-" * 60)
        print(f"\n✓ Completato! {num_plots} grafici salvati nella cartella 'quiver_dataset'")
        
    except ValueError:
        print("Errore: Inserisci un numero valido!")
    except KeyboardInterrupt:
        print("\n\nInterrotto dall'utente.")
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()