import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-GUI per evitare problemi con Wayland
import matplotlib.pyplot as plt
import random
import string
import os
from matplotlib import cm
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class ScatterDatasetGenerator:
    def __init__(self):
        # Temi matplotlib disponibili
        self.styles = ['default', 'classic', 'seaborn-v0_8', 'seaborn-v0_8-darkgrid', 
                       'seaborn-v0_8-whitegrid', 'ggplot', 'bmh', 'fivethirtyeight',
                       'seaborn-v0_8-dark', 'seaborn-v0_8-pastel', 'seaborn-v0_8-bright',
                       'Solarize_Light2', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
                       'seaborn-v0_8-talk', 'seaborn-v0_8-poster', 'seaborn-v0_8-muted',
                       'seaborn-v0_8-deep', 'seaborn-v0_8-colorblind', 'fast', 'tableau-colorblind10']
        
        # Markers disponibili
        self.markers = ['o']#, 's', '^', 'v', 'D', 'p', 'h', 'H', '+', 'x', 'd', '|', '_']
        
        # Colormaps
        self.colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight',
                         'rainbow', 'jet', 'hsv', 'spring', 'summer', 'autumn', 'winter',
                         'cool', 'hot', 'copper', 'Spectral', 'coolwarm', 'seismic']
        
        # Posizioni legenda
        self.legend_positions = ['best', 'upper right', 'upper left', 'lower left', 
                                'lower right', 'right', 'center left', 'center right']
        
    def random_string(self, min_length=3, max_length=15):
        """Genera una stringa completamente random"""
        length = random.randint(min_length, max_length)
        # Mix di lettere maiuscole, minuscole e occasionalmente numeri
        chars = string.ascii_letters
        if random.random() > 0.7:  # 30% chance di includere numeri
            chars += string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    def random_text(self, word_count=2):
        """Genera testo con parole random"""
        words = [self.random_string(random.randint(3, 12)) for _ in range(word_count)]
        # Capitalizza la prima lettera di ogni parola con il 70% di probabilità
        if random.random() > 0.3:
            words = [w.capitalize() for w in words]
        return ' '.join(words)
    
    def generate_distribution(self, n_points):
        """Genera diverse distribuzioni di dati"""
        dist_type = random.randint(0, 19)
        
        # Funzione helper per verificare validità dei dati
        def validate_data(x, y):
            # Rimuovi NaN e Inf
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            # Assicura almeno 5 punti
            if len(x) < 5:
                return None, None
            return x, y
        
        if dist_type == 0:  # Lineare con rumore
            x = np.random.uniform(0, 100, n_points)
            slope = random.uniform(-5, 5)
            intercept = random.uniform(-50, 50)
            noise = np.random.normal(0, random.uniform(5, 30), n_points)
            y = slope * x + intercept + noise
            
        elif dist_type == 1:  # Quadratica
            x = np.random.uniform(-10, 10, n_points)
            a, b, c = random.uniform(-2, 2), random.uniform(-5, 5), random.uniform(-20, 20)
            noise = np.random.normal(0, random.uniform(2, 10), n_points)
            y = a * x**2 + b * x + c + noise
            
        elif dist_type == 2:  # Esponenziale
            x = np.random.uniform(0, 5, n_points)
            y = np.exp(x * random.uniform(0.3, 1.5)) + np.random.normal(0, random.uniform(1, 5), n_points)
            
        elif dist_type == 3:  # Logaritmica
            x = np.random.uniform(1, 100, n_points)
            y = np.log(x) * random.uniform(5, 20) + np.random.normal(0, random.uniform(1, 5), n_points)
            
        elif dist_type == 4:  # Circolare
            theta = np.random.uniform(0, 2*np.pi, n_points)
            radius = random.uniform(5, 20)
            noise = random.uniform(0.5, 3)
            x = radius * np.cos(theta) + np.random.normal(0, noise, n_points)
            y = radius * np.sin(theta) + np.random.normal(0, noise, n_points)
            
        elif dist_type == 5:  # Spirale
            theta = np.random.uniform(0, 4*np.pi, n_points)
            r = theta * random.uniform(0.5, 2)
            noise = random.uniform(0.1, 1)
            x = r * np.cos(theta) + np.random.normal(0, noise, n_points)
            y = r * np.sin(theta) + np.random.normal(0, noise, n_points)
            
        elif dist_type == 6:  # Clusters multipli
            n_clusters = random.randint(2, 5)
            x, y = [], []
            for _ in range(n_clusters):
                cx, cy = random.uniform(-50, 50), random.uniform(-50, 50)
                cluster_size = n_points // n_clusters
                x.extend(np.random.normal(cx, random.uniform(3, 10), cluster_size))
                y.extend(np.random.normal(cy, random.uniform(3, 10), cluster_size))
            x, y = np.array(x), np.array(y)
            
        elif dist_type == 7:  # Correlazione negativa forte
            x = np.random.uniform(0, 100, n_points)
            y = -x * random.uniform(0.5, 2) + random.uniform(50, 150) + np.random.normal(0, random.uniform(5, 15), n_points)
            
        elif dist_type == 8:  # Nessuna correlazione
            x = np.random.uniform(0, 100, n_points)
            y = np.random.uniform(0, 100, n_points)
            
        elif dist_type == 9:  # Sinusoidale
            x = np.random.uniform(0, 4*np.pi, n_points)
            amplitude = random.uniform(5, 20)
            frequency = random.uniform(0.5, 3)
            y = amplitude * np.sin(frequency * x) + np.random.normal(0, random.uniform(1, 5), n_points)
            
        elif dist_type == 10:  # Distribuzione bivariata normale
            mean = [random.uniform(-10, 10), random.uniform(-10, 10)]
            cov = [[random.uniform(5, 20), random.uniform(-5, 5)],
                   [random.uniform(-5, 5), random.uniform(5, 20)]]
            x, y = np.random.multivariate_normal(mean, cov, n_points).T
            
        elif dist_type == 11:  # Parabolica con orientamento casuale
            t = np.random.uniform(-10, 10, n_points)
            x = t
            y = random.uniform(0.2, 2) * t**2 + np.random.normal(0, random.uniform(2, 8), n_points)
            if random.random() > 0.5:
                x, y = y, x
                
        elif dist_type == 12:  # Pattern a ventaglio
            x = np.random.uniform(0, 100, n_points)
            spread = x * random.uniform(0.1, 0.5)
            y = random.uniform(-50, 50) + np.random.normal(0, 1, n_points) * spread
            
        elif dist_type == 13:  # Pattern ad imbuto
            x = np.random.uniform(0, 100, n_points)
            spread = (100 - x) * random.uniform(0.1, 0.5)
            y = random.uniform(-50, 50) + np.random.normal(0, 1, n_points) * spread
            
        elif dist_type == 14:  # Ellisse
            theta = np.random.uniform(0, 2*np.pi, n_points)
            a, b = random.uniform(10, 30), random.uniform(5, 15)
            noise = random.uniform(0.5, 2)
            x = a * np.cos(theta) + np.random.normal(0, noise, n_points)
            y = b * np.sin(theta) + np.random.normal(0, noise, n_points)
            
        elif dist_type == 15:  # Distribuzione con outliers
            x = np.random.normal(50, 15, n_points)
            y = x * random.uniform(0.5, 1.5) + np.random.normal(0, 10, n_points)
            # Aggiungi outliers
            n_outliers = int(n_points * random.uniform(0.02, 0.1))
            outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
            x[outlier_idx] = np.random.uniform(-50, 150, n_outliers)
            y[outlier_idx] = np.random.uniform(-50, 150, n_outliers)
            
        elif dist_type == 16:  # Cubi e potenze
            x = np.random.uniform(-5, 5, n_points)
            power = random.choice([3, 4, 5])
            y = x**power + np.random.normal(0, random.uniform(5, 20), n_points)
            
        elif dist_type == 17:  # Pattern a strisce
            n_stripes = random.randint(3, 6)
            x = np.random.uniform(0, 100, n_points)
            stripe_values = np.linspace(0, 100, n_stripes)
            y = np.random.choice(stripe_values, n_points) + np.random.normal(0, 2, n_points)
            
        elif dist_type == 18:  # Distribuzione mista
            n_half = n_points // 2
            x1 = np.random.normal(30, 10, n_half)
            y1 = np.random.normal(30, 10, n_half)
            x2 = np.random.normal(70, 10, n_points - n_half)
            y2 = np.random.normal(70, 10, n_points - n_half)
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            
        else:  # Distribuzione complessa combinata
            x = np.random.uniform(0, 100, n_points)
            y1 = np.sin(x * 0.1) * 20
            y2 = x * 0.5
            weight = np.random.uniform(0, 1, n_points)
            y = y1 * weight + y2 * (1 - weight) + np.random.normal(0, 5, n_points)
        
        # Valida i dati prima di restituirli
        x, y = validate_data(x, y)
        
        # Se la validazione fallisce, genera dati semplici garantiti
        if x is None or len(x) < 5:
            x = np.random.uniform(0, 100, max(n_points, 10))
            y = np.random.uniform(0, 100, max(n_points, 10))
            x, y = validate_data(x, y)
        
        return x, y
    
    def generate_colors(self, n_points, color_mode, is_bw=False):
        """Genera colori per i punti"""
        if is_bw:  # Bianco e nero
            gray_value = random.randint(0, 200)
            return f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
        
        if color_mode == 0:  # Colore singolo
            return '#%02x%02x%02x' % (random.randint(0, 255), 
                                      random.randint(0, 255), 
                                      random.randint(0, 255))
        elif color_mode == 1:  # Gradiente basato su indice
            cmap = cm.get_cmap(random.choice(self.colormaps))
            return np.array([cmap(i/(n_points-1)) for i in range(n_points)])
        elif color_mode == 2:  # Colori random per ogni punto
            return np.array(['#%02x%02x%02x' % (random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255)) for _ in range(n_points)])
        else:  # Basato su valore y
            return np.linspace(0, 1, n_points)
    
    def generate_chart(self, idx, output_dir='scatter_dataset'):
        """Genera un singolo scatter chart"""
        # 10% dei grafici in bianco e nero
        is_bw = random.random() < 0.1
        
        # Imposta stile random
        plt.style.use(random.choice(self.styles))
        
        # Numero di punti variabile (minimo 10 per sicurezza)
        n_points = random.randint(20, 500)
        
        # Genera dati
        x, y = self.generate_distribution(n_points)
        
        # Assicurati che x e y abbiano la stessa lunghezza e almeno 5 punti
        min_len = min(len(x), len(y))
        if min_len < 5:
            # Rigenera dati semplici se troppo pochi
            x = np.random.uniform(0, 100, 20)
            y = np.random.uniform(0, 100, 20)
            min_len = 20
        x, y = x[:min_len], y[:min_len]
        n_points = min_len
        
        # Dimensioni figura variabili
        fig_width = random.uniform(6, 12)
        fig_height = random.uniform(4, 10)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Colori
        color_mode = random.randint(0, 3)
        colors = self.generate_colors(n_points, color_mode, is_bw)
        
        # Dimensioni marker FISSE (no bubble chart)
        sizes = random.uniform(10, 100)
        
        # Trasparenza
        alpha = random.uniform(0.3, 1.0)
        
        # Marker
        marker = random.choice(self.markers)
        line_markers = ['+', 'x', '|', '_']

        # Edge color
        # Se il marker è basato su linee, forziamo la presenza del bordo
        if marker in line_markers or random.random() > 0.5:
            if is_bw:
                edgecolors = 'black'
            else:
                edgecolors = 'black' if random.random() > 0.5 else 'white'
            linewidths = random.uniform(0.5, 2)
        else:
            edgecolors = 'none'
            linewidths = 0
        
        # Plot
        scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=alpha, 
                           marker=marker, edgecolors=edgecolors, 
                           linewidths=linewidths, 
                           cmap=random.choice(self.colormaps) if color_mode == 3 and not is_bw else None)
        
        # Titolo con stringhe random
        if random.random() > 0.3:
            title = self.random_text(random.randint(2, 4))
            title_color = 'black' if is_bw else 'black'
            ax.set_title(title, fontsize=random.uniform(10, 16), 
                        fontweight=random.choice(['normal', 'bold']),
                        color=title_color)
        
        # Label assi con stringhe random
        if random.random() > 0.2:
            ax.set_xlabel(self.random_text(random.randint(1, 3)), 
                         fontsize=random.uniform(8, 12),
                         color='black' if is_bw else 'black')
        if random.random() > 0.2:
            ax.set_ylabel(self.random_text(random.randint(1, 3)), 
                         fontsize=random.uniform(8, 12),
                         color='black' if is_bw else 'black')
        
        # Griglia
        if random.random() > 0.4:
            grid_color = 'gray' if is_bw else random.choice(['gray', 'black', 'lightgray'])
            ax.grid(True, alpha=random.uniform(0.2, 0.6), 
                   linestyle=random.choice(['-', '--', '-.', ':']),
                   linewidth=random.uniform(0.5, 1.5),
                   color=grid_color)
        
        # Colorbar per alcuni (non per BW)
        if color_mode == 3 and random.random() > 0.6 and not is_bw:
            plt.colorbar(scatter, ax=ax)
        
        # Linea di tendenza
        if random.random() > 0.7:
            z = np.polyfit(x, y, random.choice([1, 2, 3]))
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            line_color = 'black' if is_bw else random.choice(['red', 'blue', 'green', 'black', 'orange'])
            ax.plot(x_line, p(x_line), 
                   linestyle=random.choice(['-', '--', '-.']),
                   color=line_color,
                   linewidth=random.uniform(1, 3),
                   alpha=random.uniform(0.5, 0.9))
        
        # Colore sfondo (solo se non BW)
        if random.random() > 0.7 and not is_bw:
            bg_color = '#%02x%02x%02x' % (random.randint(240, 255), 
                                          random.randint(240, 255), 
                                          random.randint(240, 255))
            ax.set_facecolor(bg_color)
        elif is_bw:
            # Sfondo bianco per grafici BW
            ax.set_facecolor('white')
        
        # Spine (bordi)
        if random.random() > 0.6:
            for spine in ax.spines.values():
                spine.set_visible(random.choice([True, False]))
        
        # Colori assi per BW
        if is_bw:
            ax.tick_params(colors='black')
            for spine in ax.spines.values():
                spine.set_color('black')
        
        # Tight layout
        plt.tight_layout()
        
        # Salva
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/scatter_{idx:05d}.png'
        dpi = random.choice([100, 150, 200, 300])
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white')
        plt.close()
        
        return filename

def main():
    print("=" * 60)
    print("GENERATORE DATASET SCATTER CHARTS")
    print("=" * 60)
    
    try:
        n_charts = int(input("\nQuanti grafici vuoi generare? "))
        if n_charts <= 0:
            print("Errore: inserisci un numero positivo!")
            return
    except ValueError:
        print("Errore: inserisci un numero valido!")
        return
    
    output_dir = 'scatter_dataset'
    
    print(f"\n{'='*60}")
    print(f"Generazione di {n_charts} scatter charts in corso...")
    print(f"Cartella output: {output_dir}/")
    print(f"{'='*60}\n")
    
    generator = ScatterDatasetGenerator()
    
    for i in range(1001, n_charts):
        filename = generator.generate_chart(i, output_dir)
        if (i + 1) % 10 == 0 or i == n_charts - 1:
            print(f"Progresso: {i+1}/{n_charts} grafici generati ({(i+1)/n_charts*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"✓ Completato! {n_charts} grafici salvati in '{output_dir}/'")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()