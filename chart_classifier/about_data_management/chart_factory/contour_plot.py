import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import random
import os
import string  # Necessario per generare stringhe casuali

def get_random_text(length=None):
    """Genera una stringa di caratteri alfanumerici casuali."""
    if length is None:
        length = random.randint(4, 15)
    
    # Lettere (maiuscole/minuscole) e numeri
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def generate_contour_dataset(num_plots):
    """
    Genera un dataset di contour plot con variazioni massime di stile.
    
    Args:
        num_plots: Numero di grafici da generare
    """
    
    # Crea cartella per salvare i grafici
    output_dir = f"contour_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Liste di opzioni per massima varietà
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                 'twilight', 'twilight_shifted', 'hsv']
    
    fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    
    # Funzioni matematiche per generare dati diversi
    def func1(x, y): return np.sin(np.sqrt(x**2 + y**2))
    def func2(x, y): return np.cos(x) * np.sin(y)
    def func3(x, y): return x**2 + y**2
    def func4(x, y): return np.exp(-(x**2 + y**2))
    def func5(x, y): return np.sin(x) * np.cos(y)
    def func6(x, y): return np.tan(x) * np.tan(y)
    def func7(x, y): return np.sin(x**2 + y**2)
    def func8(x, y): return x**3 - 3*x*y**2
    def func9(x, y): return np.cos(x**2) + np.sin(y**2)
    def func10(x, y): return (x**2 - y**2) * np.exp(-(x**2 + y**2)/2)
    def func11(x, y): return np.sin(x*y)
    def func12(x, y): return np.log(np.abs(x*y) + 1)
    
    functions = [func1, func2, func3, func4, func5, func6, 
                 func7, func8, func9, func10, func11, func12]
    
    print(f"Generazione di {num_plots} contour plots...")
    
    for i in range(0, num_plots):
        # Dimensioni casuali della figura
        figsize = (random.uniform(6, 12), random.uniform(6, 12))
        fig, ax = plt.subplots(figsize=figsize, dpi=random.choice([80, 100, 150, 200]))
        
        # Range casuale per i dati
        x_range = random.uniform(2, 10)
        y_range = random.uniform(2, 10)
        resolution = random.randint(50, 200)
        
        x = np.linspace(-x_range, x_range, resolution)
        y = np.linspace(-y_range, y_range, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Seleziona funzione casuale
        func = random.choice(functions)
        
        # Aggiungi rumore casuale
        noise_level = random.uniform(0, 0.3)
        try:
            Z = func(X, Y) + noise_level * np.random.randn(*X.shape)
        except:
            Z = X**2 + Y**2 + noise_level * np.random.randn(*X.shape)
        
        # Colormap casuale
        cmap = random.choice(colormaps)
        
        # Inverti colormap casualmente
        if random.random() > 0.5:
            cmap = cmap + '_r'
        
        # Numero di livelli casuale
        num_levels = random.randint(5, 30)
        
        # Tipo di contour plot
        plot_type = random.choice(['contour', 'contourf', 'both'])
        
        if plot_type == 'contour':
            # Solo linee di contorno
            linewidths = random.uniform(0.5, 3.0)
            cs = ax.contour(X, Y, Z, levels=num_levels, cmap=cmap, linewidths=linewidths)
            if random.random() > 0.5:
                ax.clabel(cs, inline=True, fontsize=random.randint(6, 12))
        elif plot_type == 'contourf':
            # Solo riempimento
            cs = ax.contourf(X, Y, Z, levels=num_levels, cmap=cmap)
        else:
            # Entrambi
            cs = ax.contourf(X, Y, Z, levels=num_levels, cmap=cmap, alpha=random.uniform(0.6, 1.0))
            cs2 = ax.contour(X, Y, Z, levels=num_levels, colors='black', 
                            linewidths=random.uniform(0.3, 1.5), alpha=random.uniform(0.3, 0.8))
            if random.random() > 0.7:
                ax.clabel(cs2, inline=True, fontsize=random.randint(6, 10))
        
        # Aggiungi colorbar casualmente
        if random.random() > 0.3:
            cbar = plt.colorbar(cs, ax=ax)
            cbar_fontsize = random.randint(8, 14)
            cbar.ax.tick_params(labelsize=cbar_fontsize)
            # Opzionale: anche la label della colorbar random
            if random.random() > 0.5:
                cbar.set_label(get_random_text(random.randint(4, 10)), fontsize=cbar_fontsize)
        
        # === MODIFICHE QUI: LABEL ASSI CASUALI ===
        ax.set_xlabel(get_random_text(random.randint(3, 8)), 
                      fontsize=random.randint(10, 16), 
                      family=random.choice(fonts))
                      
        ax.set_ylabel(get_random_text(random.randint(3, 8)), 
                      fontsize=random.randint(10, 16), 
                      family=random.choice(fonts))
        
        # === MODIFICHE QUI: TITOLO CASUALE ===
        if random.random() > 0.4:
            title = get_random_text(random.randint(5, 20))
            ax.set_title(title, fontsize=random.randint(12, 18), 
                         family=random.choice(fonts))
        
        # Stile griglia
        if random.random() > 0.5:
            ax.grid(True, alpha=random.uniform(0.2, 0.6), 
                   linestyle=random.choice(['-', '--', '-.', ':']),
                   linewidth=random.uniform(0.5, 1.5))
        
        # Stile dei tick
        tick_fontsize = random.randint(8, 14)
        ax.tick_params(labelsize=tick_fontsize)
        
        # Background color casuale
        bg_colors = ['white', 'lightgray', 'lavender', 'lightyellow', 
                     'lightcyan', 'honeydew', 'mistyrose', 'azure']
        ax.set_facecolor(random.choice(bg_colors))
        
        # Aspetto ratio
        if random.random() > 0.5:
            ax.set_aspect('equal')
        
        # Bordi casuali
        for spine in ax.spines.values():
            spine.set_linewidth(random.uniform(0.5, 2.5))
        
        # Salva il grafico
        filename = f"{output_dir}/contour_plot_{i+1:04d}.png"
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', 
                    facecolor=random.choice(['white', 'lightgray', None]))
        plt.close()
        
        # Progresso
        if (i + 1) % 10 == 0 or i == num_plots - 1:
            print(f"Generati {i+1}/{num_plots} grafici...")
    
    print(f"\n✓ Dataset completato!")
    print(f"✓ {num_plots} contour plots salvati in '{output_dir}'")
    print(f"✓ Ogni grafico ha stile, colori, titoli e labels unici")

if __name__ == "__main__":
    try:
        num = int(input("Inserisci il numero di contour plots da generare: "))
        if num <= 0:
            print("Errore: inserisci un numero positivo")
        else:
            generate_contour_dataset(num)
    except ValueError:
        print("Errore: inserisci un numero valido")