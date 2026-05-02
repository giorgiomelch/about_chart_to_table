import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per generazione batch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import string
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedHeatmapGenerator:
    def __init__(self):
        # Palette colori estese con stili scientifici
        self.color_palettes = {
            # Sequenziali
            'sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                          'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
            
            # Divergenti (comuni nei paper scientifici)
            'diverging': ['RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 
                         'bwr', 'seismic', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy'],
            
            # Custom scientifici
            'scientific': ['jet', 'rainbow', 'turbo', 'twilight', 'twilight_shifted',
                          'hsv', 'nipy_spectral', 'gist_ncar', 'gist_rainbow'],
            
            # Monocromatici (per paper in bianco e nero)
            'monochrome': ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 
                          'pink', 'copper'],
        }
        
        # Tutti i colori disponibili
        self.all_palettes = [p for cat in self.color_palettes.values() for p in cat]
        
        # Colori di sfondo
        self.background_colors = {
            'light': ['white', 'whitesmoke', 'floralwhite', 'ivory', 'snow', 
                     'seashell', 'oldlace', 'linen'],
            'neutral': ['gainsboro', 'lightgray', 'silver', 'lightsteelblue'],
            'scientific': ['white', 'white', 'white', 'whitesmoke']  # Più white per stile paper
        }
        
        # Pattern di dati scientifici
        self.data_patterns = ['random', 'uniform', 'gaussian', 'gradient_horizontal', 
                             'gradient_vertical', 'checkerboard', 'blocks', 'clusters',
                             'radial', 'wave', 'sparse', 'correlation']
        
        # Configurazioni dimensioni (ridotte per performance)
        self.size_configs = {
            'small': (5, 10, 5, 10),      # (min_rows, max_rows, min_cols, max_cols)
            'medium': (10, 25, 10, 25),
            'large': (25, 50, 25, 50),
            'rectangular_h': (5, 15, 20, 50),
            'rectangular_v': (20, 50, 5, 15),
        }
        
    def generate_random_word(self, min_length=3, max_length=10):
        """Genera una parola completamente casuale"""
        length = random.randint(min_length, max_length)
        
        # 70% lettere, 30% mix lettere e numeri
        if random.random() < 0.7:
            # Solo lettere
            word = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=length))
        else:
            # Mix lettere e numeri
            chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
            word = ''.join(random.choices(chars, k=length))
        
        # Aggiungi occasionalmente un separatore
        if random.random() < 0.3 and length > 5:
            separator = random.choice(['-', '_', '.'])
            split_pos = random.randint(2, length - 2)
            word = word[:split_pos] + separator + word[split_pos:]
        
        return word
    
    def generate_data_pattern(self, rows, cols, pattern):
        """Genera pattern di dati diversificati - versione ottimizzata"""
        
        if pattern == 'random':
            return np.random.rand(rows, cols) * 100
        
        elif pattern == 'uniform':
            base = random.uniform(30, 70)
            noise = random.uniform(5, 15)
            return np.ones((rows, cols)) * base + np.random.randn(rows, cols) * noise
        
        elif pattern == 'gaussian':
            return np.random.randn(rows, cols) * 20 + 50
        
        elif pattern == 'gradient_horizontal':
            data = np.linspace(0, 100, cols)
            return np.tile(data, (rows, 1)) + np.random.randn(rows, cols) * 5
        
        elif pattern == 'gradient_vertical':
            data = np.linspace(0, 100, rows)
            return np.tile(data.reshape(-1, 1), (1, cols)) + np.random.randn(rows, cols) * 5
        
        elif pattern == 'checkerboard':
            data = np.zeros((rows, cols))
            size = random.randint(1, 3)
            for i in range(rows):
                for j in range(cols):
                    data[i, j] = 50 if ((i // size) + (j // size)) % 2 == 0 else 100
            return data + np.random.randn(rows, cols) * 8
        
        elif pattern == 'blocks':
            data = np.random.rand(rows, cols) * 20
            num_blocks = random.randint(3, 8)
            for _ in range(num_blocks):
                block_size = random.randint(3, max(3, min(rows, cols) // 5))
                r = random.randint(0, max(0, rows - block_size))
                c = random.randint(0, max(0, cols - block_size))
                value = random.uniform(60, 100)
                data[r:r+block_size, c:c+block_size] = value + np.random.randn(block_size, block_size) * 5
            return data
        
        elif pattern == 'clusters':
            data = np.random.rand(rows, cols) * 20
            num_clusters = random.randint(3, 6)
            for _ in range(num_clusters):
                r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                radius = random.randint(2, max(2, min(rows, cols) // 8))
                value = random.uniform(60, 100)
                for i in range(max(0, r-radius), min(rows, r+radius)):
                    for j in range(max(0, c-radius), min(cols, c+radius)):
                        if np.sqrt((i-r)**2 + (j-c)**2) <= radius:
                            data[i, j] = value + np.random.randn() * 5
            return np.clip(data, 0, 100)
        
        elif pattern == 'radial':
            center_r, center_c = rows // 2, cols // 2
            data = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    dist = np.sqrt((i - center_r)**2 + (j - center_c)**2)
                    max_dist = np.sqrt(center_r**2 + center_c**2)
                    data[i, j] = (dist / max_dist) * 100
            return data + np.random.randn(rows, cols) * 5
        
        elif pattern == 'wave':
            data = np.zeros((rows, cols))
            freq = random.uniform(0.3, 1.0)
            for i in range(rows):
                for j in range(cols):
                    data[i, j] = 50 * (np.sin(i * freq) + np.cos(j * freq)) + 50
            return data + np.random.randn(rows, cols) * 5
        
        elif pattern == 'sparse':
            data = np.random.rand(rows, cols) * 15
            sparsity = random.uniform(0.7, 0.9)
            mask = np.random.rand(rows, cols) > sparsity
            data[mask] += random.uniform(60, 100)
            return data
        
        elif pattern == 'correlation':
            # Matrice di correlazione simulata
            data = np.random.rand(rows, cols)
            if rows == cols:
                data = (data + data.T) / 2
                np.fill_diagonal(data, 1)
            return data * 100
        
        else:
            return np.random.rand(rows, cols) * 100
    
    def create_custom_colormap(self):
        """Crea colormap personalizzate"""
        cmap_type = random.choice(['sequential', 'diverging'])
        
        if cmap_type == 'sequential':
            colors = [
                (random.random(), random.random(), random.random()),
                (random.random(), random.random(), random.random())
            ]
            return LinearSegmentedColormap.from_list('custom', colors)
        else:  # diverging
            colors = [
                (random.random(), random.random(), random.random()),
                (1, 1, 1),
                (random.random(), random.random(), random.random())
            ]
            return LinearSegmentedColormap.from_list('custom', colors)
    
    def generate_heatmap(self, index, size_config='random', output_dir='heatmap_dataset'):
        """Genera un singolo heatmap"""
        
        try:
            # Seleziona configurazione dimensioni
            if size_config == 'random':
                config_type = random.choice(list(self.size_configs.keys()))
            else:
                config_type = size_config if size_config in self.size_configs else 'medium'
            
            min_r, max_r, min_c, max_c = self.size_configs[config_type]
            rows = random.randint(min_r, max_r)
            cols = random.randint(min_c, max_c)
            
            # Pattern
            pattern = random.choice(self.data_patterns)
            
            # Genera dati
            data = self.generate_data_pattern(rows, cols, pattern)
            
            # Genera etichette completamente casuali e uniche
            x_labels = []
            for i in range(cols):
                label = self.generate_random_word()
                # Assicura unicità
                attempts = 0
                while label in x_labels and attempts < 10:
                    label = self.generate_random_word()
                    attempts += 1
                if attempts >= 10:
                    label = f"{label}_{i}"
                x_labels.append(label)
            
            y_labels = []
            for i in range(rows):
                label = self.generate_random_word()
                attempts = 0
                while label in y_labels and attempts < 10:
                    label = self.generate_random_word()
                    attempts += 1
                if attempts >= 10:
                    label = f"{label}_{i}"
                y_labels.append(label)
            
            # Crea DataFrame
            df = pd.DataFrame(data, columns=x_labels, index=y_labels)
            
            # Configurazione figura
            aspect_ratio = cols / rows
            if aspect_ratio > 2:
                fig_width = min(14, cols * 0.3)
                fig_height = fig_width / aspect_ratio
            elif aspect_ratio < 0.5:
                fig_height = min(14, rows * 0.3)
                fig_width = fig_height * aspect_ratio
            else:
                fig_width = min(12, max(8, cols * 0.3))
                fig_height = fig_width / aspect_ratio
            
            fig_width = max(6, min(fig_width, 20))
            fig_height = max(6, min(fig_height, 20))
            
            dpi = random.choice([72, 100, 150])
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            
            # Background
            bg_category = random.choice(list(self.background_colors.keys()))
            bg_color = random.choice(self.background_colors[bg_category])
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            # Colormap
            if random.random() > 0.85:
                cmap = self.create_custom_colormap()
            else:
                palette_category = random.choice(list(self.color_palettes.keys()))
                cmap = random.choice(self.color_palettes[palette_category])
            
            # Parametri heatmap
            annot = False if (rows * cols > 200) else random.choice([True, False, False])
            fmt = random.choice(['.0f', '.1f', '.2f'])
            
            if rows * cols > 500:
                linewidths = 0
            elif rows * cols > 200:
                linewidths = random.uniform(0, 0.5)
            else:
                linewidths = random.uniform(0, 1.5)
            
            linecolor = random.choice(['white', 'black', 'gray', 'lightgray'])
            cbar = random.choice([True, True, False])
            square = random.choice([True, False])
            
            # Crea heatmap
            annot_kws = {'size': random.randint(6, 9)} if annot else {}
            
            sns.heatmap(df, 
                       annot=annot,
                       fmt=fmt if annot else '',
                       cmap=cmap,
                       linewidths=linewidths,
                       linecolor=linecolor,
                       cbar=cbar,
                       square=square,
                       ax=ax,
                       annot_kws=annot_kws,
                       robust=random.choice([True, False]))
            
            # Titolo
            if random.random() > 0.3:
                # Genera titolo casuale
                title_length = random.randint(10, 40)
                title_words = [self.generate_random_word(3, 8) for _ in range(random.randint(2, 6))]
                title = ' '.join(title_words)
                
                title_size = random.randint(10, 14)
                title_weight = random.choice(['normal', 'bold'])
                ax.set_title(title, fontsize=title_size, weight=title_weight, pad=random.randint(10, 20))
            
            # Rotazione etichette
            if cols > 20:
                x_rotation = random.choice([90, 45])
            else:
                x_rotation = random.choice([0, 45, 90])
            
            y_rotation = 0 if rows > 30 else random.choice([0, 45])
            
            # Dimensione font
            if rows > 30 or cols > 30:
                label_size = random.randint(5, 8)
            else:
                label_size = random.randint(7, 10)
            
            plt.xticks(rotation=x_rotation, 
                      ha='right' if x_rotation in [45, 90] else 'center',
                      fontsize=label_size)
            plt.yticks(rotation=y_rotation, fontsize=label_size)
            
            # Labels assi
            if random.random() > 0.4:
                xlabel_words = [self.generate_random_word(4, 10) for _ in range(random.randint(1, 3))]
                xlabel = ' '.join(xlabel_words)
                ax.set_xlabel(xlabel, fontsize=random.randint(9, 12), 
                             weight=random.choice(['normal', 'bold']))
            
            if random.random() > 0.4:
                ylabel_words = [self.generate_random_word(4, 10) for _ in range(random.randint(1, 3))]
                ylabel = ' '.join(ylabel_words)
                ax.set_ylabel(ylabel, fontsize=random.randint(9, 12),
                             weight=random.choice(['normal', 'bold']))
            
            # Layout
            plt.tight_layout()
            
            # Assicura che la directory esista
            os.makedirs(output_dir, exist_ok=True)
            
            # Salva con nome univoco
            filename = f'{output_dir}/heatmap_{index:06d}.png'
            plt.savefig(filename, facecolor=bg_color, bbox_inches='tight', 
                       dpi=dpi, pad_inches=0.1)
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            print(f"Errore nella generazione del grafico {index}: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def generate_batch(self, num_graphs, size_config='random', start_index=0):
        """Genera un batch di heatmap"""
        successful = 0
        failed = 0
        
        output_dir = 'heatmap_dataset'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerazione in corso...")
        print(f"Directory output: {os.path.abspath(output_dir)}\n")
        
        for i in range(num_graphs):
            try:
                filename = self.generate_heatmap(start_index + i, size_config, output_dir)
                if filename and os.path.exists(filename):
                    successful += 1
                    # Stampa progresso immediato
                    if (i + 1) % 10 == 0:
                        print(f"✓ Generati: {i+1}/{num_graphs} | Successi: {successful} | Falliti: {failed}")
                else:
                    failed += 1
                    print(f"✗ Fallito: {start_index + i}")
                    
            except Exception as e:
                print(f"✗ Errore nel grafico {start_index + i}: {str(e)}")
                failed += 1
        
        return successful, failed

def main():
    print("=" * 70)
    print("GENERATORE AVANZATO DI HEATMAP DATASET PER AI TRAINING")
    print("=" * 70)
    print("\nCaratteristiche:")
    print("✓ Parole completamente casuali generate automaticamente")
    print("✓ Pattern scientifici ottimizzati")
    print("✓ Supporto per diverse dimensioni di matrici")
    print("✓ Salvataggio garantito in 'heatmap_dataset/'")
    print("=" * 70)
    
    # Input numero grafici
    while True:
        try:
            num_input = input("\nQuanti grafici vuoi generare? [default: 10]: ").strip()
            num_graphs = int(num_input) if num_input else 10
            if num_graphs <= 0:
                print("Inserisci un numero positivo!")
                continue
            break
        except ValueError:
            print("Inserisci un numero valido!")
    
    # Input configurazione dimensioni
    print("\nConfigurazioni dimensioni disponibili:")
    print("1. Random (mix di tutte le dimensioni)")
    print("2. Small (5-10 righe/colonne)")
    print("3. Medium (10-25 righe/colonne)")
    print("4. Large (25-50 righe/colonne)")
    print("5. Rectangular horizontal (poche righe, molte colonne)")
    print("6. Rectangular vertical (molte righe, poche colonne)")
    
    size_map = {
        '1': 'random',
        '2': 'small',
        '3': 'medium',
        '4': 'large',
        '5': 'rectangular_h',
        '6': 'rectangular_v'
    }
    
    while True:
        size_choice = input("\nScegli configurazione (1-6) [default: 1]: ").strip() or '1'
        if size_choice in size_map:
            size_config = size_map[size_choice]
            break
        print("Scelta non valida!")
    
    generator = AdvancedHeatmapGenerator()
    
    print(f"\n{'='*70}")
    print(f"Generazione di {num_graphs} heatmap con configurazione '{size_config}'...")
    print(f"{'='*70}")
    
    import time
    start_time = time.time()
    
    successful, failed = generator.generate_batch(num_graphs, size_config)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETATO!")
    print(f"{'='*70}")
    print(f"  • Grafici generati con successo: {successful}")
    print(f"  • Grafici falliti: {failed}")
    print(f"  • Tempo impiegato: {elapsed_time:.2f} secondi")
    if num_graphs > 0:
        print(f"  • Velocità media: {elapsed_time/num_graphs:.3f} secondi/grafico")
    print(f"  • Cartella output: '{os.path.abspath('heatmap_dataset')}'")
    print(f"{'='*70}")
    
    # Verifica finale
    if successful > 0:
        print(f"\n✓ Verifica: trovati {len(os.listdir('heatmap_dataset'))} file nella cartella")

if __name__ == "__main__":
    main()