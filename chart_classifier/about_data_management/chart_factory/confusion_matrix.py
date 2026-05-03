import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, RegularPolygon
import random
import string
import os

class ConfusionMatrixGenerator:
    def __init__(self):
        self.color_palettes = [
            ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a'],
            ['#fff3e0', '#ffe0b2', '#ffcc80', '#ffb74d', '#ffa726'],
            ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5'],
            ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc'],
            ['#fce4ec', '#f8bbd0', '#f48fb1', '#f06292', '#ec407a'],
            ['#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', '#26a69a'],
            ['#fff9c4', '#fff59d', '#fff176', '#ffee58', '#ffeb3b'],
            ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350'],
            ['#f1f8e9', '#dcedc8', '#c5e1a5', '#aed581', '#9ccc65'],
            ['#e8eaf6', '#c5cae9', '#9fa8da', '#7986cb', '#5c6bc0'],
            ['#fafafa', '#f5f5f5', '#eeeeee', '#e0e0e0', '#bdbdbd'],
            ['#ffe0b2', '#ffcc80', '#ffb74d', '#ffa726', '#ff9800'],
            ['#d1c4e9', '#b39ddb', '#9575cd', '#7e57c2', '#673ab7'],
            ['#b2dfdb', '#80cbc4', '#4db6ac', '#26a69a', '#009688'],
            ['#ffccbc', '#ffab91', '#ff8a65', '#ff7043', '#ff5722']
        ]
        
        self.cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 
                      'YlOrRd', 'YlGnBu', 'RdPu', 'BuPu', 'GnBu',
                      'PuRd', 'YlOrBr', 'OrRd', 'PuBu', 'BuGn']
        
        self.fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
        
        self.shapes = ['square', 'rounded', 'circle', 'diamond', 'hexagon']
        
        self.line_styles = ['-', '--', '-.', ':']
        
    def generate_random_string(self, min_len=3, max_len=12, include_numbers=True, include_special=False):
        """Genera una stringa casuale"""
        length = random.randint(min_len, max_len)
        
        # Caratteri disponibili
        chars = string.ascii_letters
        if include_numbers:
            chars += string.digits
        if include_special:
            chars += '_-'
        
        # Genera stringa
        result = ''.join(random.choice(chars) for _ in range(length))
        
        # Capitalizza casualmente
        if random.choice([True, False]):
            result = result.capitalize()
        elif random.choice([True, False]):
            result = result.upper()
        
        return result
    
    def generate_matrix(self):
        """Genera una confusion matrix con diverse distribuzioni"""
        size = random.randint(2, 10)
        matrix = np.zeros((size, size), dtype=int)
        
        # Scegli distribuzione casuale
        distribution_type = random.choice([
            'balanced',           # Distribuzione bilanciata
            'diagonal_heavy',     # Diagonale molto forte
            'diagonal_weak',      # Diagonale debole
            'random',             # Completamente random
            'sparse',             # Matrice sparsa
            'dense',              # Matrice densa
            'clustered',          # Valori clusterizzati
            'exponential',        # Distribuzione esponenziale
            'uniform',            # Distribuzione uniforme
            'normal',             # Distribuzione normale
            'skewed'              # Distribuzione asimmetrica
        ])
        
        total = random.randint(500, 5000)
        
        if distribution_type == 'balanced':
            # Distribuzione bilanciata tra tutte le celle
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i][j] = random.randint(int(total * 0.15), int(total * 0.25))
                    else:
                        matrix[i][j] = random.randint(int(total * 0.02), int(total * 0.08))
        
        elif distribution_type == 'diagonal_heavy':
            # Diagonale molto dominante
            for i in range(size):
                matrix[i][i] = random.randint(int(total * 0.3), int(total * 0.5))
            for i in range(size):
                for j in range(size):
                    if i != j:
                        matrix[i][j] = random.randint(0, int(total * 0.02))
        
        elif distribution_type == 'diagonal_weak':
            # Diagonale debole, molti errori
            for i in range(size):
                matrix[i][i] = random.randint(int(total * 0.05), int(total * 0.15))
            for i in range(size):
                for j in range(size):
                    if i != j:
                        matrix[i][j] = random.randint(int(total * 0.05), int(total * 0.15))
        
        elif distribution_type == 'random':
            # Completamente casuale
            for i in range(size):
                for j in range(size):
                    matrix[i][j] = random.randint(0, int(total * 0.2))
        
        elif distribution_type == 'sparse':
            # Matrice sparsa con molti zeri
            for i in range(size):
                matrix[i][i] = random.randint(int(total * 0.2), int(total * 0.4))
            # Solo alcuni elementi non-diagonali hanno valori
            num_nonzero = random.randint(size, size * 2)
            for _ in range(num_nonzero):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                if i != j:
                    matrix[i][j] = random.randint(1, int(total * 0.05))
        
        elif distribution_type == 'dense':
            # Matrice densa, pochi zeri
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i][j] = random.randint(int(total * 0.1), int(total * 0.3))
                    else:
                        matrix[i][j] = random.randint(int(total * 0.03), int(total * 0.12))
        
        elif distribution_type == 'clustered':
            # Errori clusterizzati in alcune aree
            for i in range(size):
                matrix[i][i] = random.randint(int(total * 0.2), int(total * 0.35))
            # Crea cluster di errori
            cluster_center = random.randint(0, size-1)
            for i in range(max(0, cluster_center-1), min(size, cluster_center+2)):
                for j in range(max(0, cluster_center-1), min(size, cluster_center+2)):
                    if i != j:
                        matrix[i][j] = random.randint(int(total * 0.05), int(total * 0.15))
        
        elif distribution_type == 'exponential':
            # Distribuzione esponenziale decrescente
            for i in range(size):
                for j in range(size):
                    distance = abs(i - j)
                    if distance == 0:
                        matrix[i][j] = random.randint(int(total * 0.25), int(total * 0.4))
                    else:
                        max_val = int(total * 0.15 * np.exp(-distance))
                        matrix[i][j] = random.randint(0, max(1, max_val))
        
        elif distribution_type == 'uniform':
            # Distribuzione uniforme
            uniform_val = total // (size * size)
            for i in range(size):
                for j in range(size):
                    matrix[i][j] = random.randint(int(uniform_val * 0.7), int(uniform_val * 1.3))
        
        elif distribution_type == 'normal':
            # Distribuzione normale attorno alla diagonale
            mean = total // (size * size)
            std = mean * 0.5
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i][j] = int(abs(np.random.normal(mean * 3, std)))
                    else:
                        matrix[i][j] = int(abs(np.random.normal(mean * 0.5, std * 0.3)))
        
        elif distribution_type == 'skewed':
            # Distribuzione asimmetrica
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i][j] = random.randint(int(total * 0.2), int(total * 0.4))
                    elif i < j:
                        # Sopra la diagonale: valori più alti
                        matrix[i][j] = random.randint(int(total * 0.05), int(total * 0.15))
                    else:
                        # Sotto la diagonale: valori più bassi
                        matrix[i][j] = random.randint(0, int(total * 0.05))
        
        return matrix
    
    def get_labels(self, size):
        """Genera etichette casuali con stringhe random"""
        label_style = random.choice([
            'random_strings',
            'random_codes',
            'mixed_alphanumeric',
            'short_codes',
            'long_names',
            'numeric_strings',
            'single_chars'
        ])
        
        labels = []
        
        if label_style == 'random_strings':
            # Stringhe completamente random
            for _ in range(size):
                labels.append(self.generate_random_string(4, 10, include_numbers=False))
        
        elif label_style == 'random_codes':
            # Codici con numeri e lettere
            for _ in range(size):
                labels.append(self.generate_random_string(3, 8, include_numbers=True))
        
        elif label_style == 'mixed_alphanumeric':
            # Mix di prefisso e numeri
            prefix = self.generate_random_string(2, 4, include_numbers=False)
            for i in range(size):
                labels.append(f"{prefix}{random.randint(0, 999)}")
        
        elif label_style == 'short_codes':
            # Codici corti
            for _ in range(size):
                labels.append(self.generate_random_string(2, 4, include_numbers=True))
        
        elif label_style == 'long_names':
            # Nomi lunghi
            for _ in range(size):
                labels.append(self.generate_random_string(8, 15, include_numbers=False))
        
        elif label_style == 'numeric_strings':
            # Solo numeri
            for _ in range(size):
                labels.append(str(random.randint(0, 9999)))
        
        elif label_style == 'single_chars':
            # Singoli caratteri
            for i in range(size):
                if i < 26:
                    labels.append(chr(65 + i))
                else:
                    labels.append(f"{chr(65 + (i % 26))}{i // 26}")
        
        return labels
    
    def get_random_title(self):
        """Genera un titolo casuale con stringhe random"""
        title_styles = [
            lambda: self.generate_random_string(5, 15, include_numbers=False),
            lambda: f"{self.generate_random_string(4, 10)} {self.generate_random_string(4, 10)}",
            lambda: f"{self.generate_random_string(3, 8)} - {self.generate_random_string(3, 8)}",
            lambda: f"{self.generate_random_string(6, 12).upper()}",
            lambda: f"{self.generate_random_string(4, 8)} {random.randint(1, 100)}",
            lambda: "",  # Nessun titolo
            lambda: f"{self.generate_random_string(3, 6)}-{self.generate_random_string(3, 6)}",
            lambda: f"{self.generate_random_string(8, 20, include_numbers=True, include_special=True)}"
        ]
        
        return random.choice(title_styles)()
    
    def create_chart(self, matrix, idx, output_dir='confusion_matrices'):
        """Crea un singolo grafico confusion matrix con stile casuale"""
        
        # Parametri casuali
        use_cmap = random.choice([True, False])
        if use_cmap:
            cmap = random.choice(self.cmaps)
            colors = None
        else:
            colors = random.choice(self.color_palettes)
            cmap = None
        
        size = matrix.shape[0]
        labels = self.get_labels(size)
        
        # Dimensioni e stili - adattate per matrici fino a 10x10
        base_size = max(8, size * 0.8)
        figsize = (random.uniform(base_size, base_size + 4), random.uniform(base_size, base_size + 4))
        fontsize_title = random.randint(10, 18)
        fontsize_labels = random.randint(6, 12)
        fontsize_values = random.randint(7, 14)
        font_family = random.choice(self.fonts)
        
        # Stili
        shape = random.choice(self.shapes)
        show_percentage = random.choice([True, False])
        show_gridlines = random.choice([True, False])
        line_style = random.choice(self.line_styles)
        linewidth = random.uniform(0.5, 3)
        
        # Rotazione etichette
        rotate_x = random.choice([0, 45, 90])
        rotate_y = random.choice([0, 45])
        
        # Creazione figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalizzazione per colormap
        if use_cmap:
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
            
            # Aggiungi colorbar in modo casuale
            if random.choice([True, False]):
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize_labels)
        else:
            # Usa palette di colori personalizzata
            max_val = matrix.max()
            min_val = matrix.min()
            
            for i in range(size):
                for j in range(size):
                    value = matrix[i, j]
                    # Normalizza il valore
                    normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    color_idx = min(int(normalized * len(colors)), len(colors) - 1)
                    color = colors[color_idx]
                    
                    # Disegna forme diverse
                    if shape == 'square':
                        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                linewidth=linewidth, 
                                                edgecolor='black' if show_gridlines else 'none',
                                                facecolor=color, 
                                                linestyle=line_style)
                        ax.add_patch(rect)
                    elif shape == 'rounded':
                        radius = random.uniform(0.1, 0.3)
                        rect = FancyBboxPatch((j-0.45, i-0.45), 0.9, 0.9,
                                            boxstyle=f"round,pad=0.05,rounding_size={radius}",
                                            linewidth=linewidth,
                                            edgecolor='black' if show_gridlines else 'none',
                                            facecolor=color,
                                            linestyle=line_style)
                        ax.add_patch(rect)
                    elif shape == 'circle':
                        circle = Circle((j, i), 0.45, 
                                      linewidth=linewidth,
                                      edgecolor='black' if show_gridlines else 'none',
                                      facecolor=color,
                                      linestyle=line_style)
                        ax.add_patch(circle)
                    elif shape == 'diamond':
                        diamond = RegularPolygon((j, i), 4, radius=0.5, 
                                               orientation=np.pi/4,
                                               linewidth=linewidth,
                                               edgecolor='black' if show_gridlines else 'none',
                                               facecolor=color,
                                               linestyle=line_style)
                        ax.add_patch(diamond)
                    elif shape == 'hexagon':
                        hexagon = RegularPolygon((j, i), 6, radius=0.45,
                                               linewidth=linewidth,
                                               edgecolor='black' if show_gridlines else 'none',
                                               facecolor=color,
                                               linestyle=line_style)
                        ax.add_patch(hexagon)
        
        # Annotazioni - SEMPRE PRESENTI
        total = matrix.sum()
        for i in range(size):
            for j in range(size):
                value = matrix[i, j]
                
                # Colore testo basato su luminosità sfondo
                if use_cmap:
                    text_color = 'white' if value > matrix.max() * 0.5 else 'black'
                else:
                    text_color = 'black' if i == j else 'darkslategray'
                
                # Testo - adatta dimensione per matrici grandi
                if show_percentage:
                    percentage = (value / total) * 100 if total > 0 else 0
                    if size > 6:
                        # Per matrici grandi, percentuale più piccola
                        text = f'{value}\n{percentage:.1f}%'
                    else:
                        text = f'{value}\n({percentage:.1f}%)'
                else:
                    text = f'{value}'
                
                # Stile testo
                fontweight = 'bold' if i == j else 'normal'
                # Riduci font per matrici grandi
                fontsize_adjusted = fontsize_values if size <= 5 else max(6, fontsize_values - (size - 5))
                
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontsize=fontsize_adjusted,
                       fontweight=fontweight, family=font_family)
        
        # Etichette assi
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(labels, fontsize=fontsize_labels, 
                          rotation=rotate_x, ha='right' if rotate_x > 0 else 'center',
                          family=font_family)
        ax.set_yticklabels(labels, fontsize=fontsize_labels, 
                          rotation=rotate_y, family=font_family)
        
        # Titolo random
        title = self.get_random_title()
        
        if title:
            ax.set_title(title, fontsize=fontsize_title, 
                        fontweight='bold', pad=20, family=font_family)
        
        # Label assi con stringhe random
        if random.choice([True, False]):
            xlabel = self.generate_random_string(4, 12, include_numbers=False)
            ylabel = self.generate_random_string(4, 12, include_numbers=False)
            ax.set_xlabel(xlabel, fontsize=fontsize_labels + 2, 
                         fontweight='bold', family=font_family)
            ax.set_ylabel(ylabel, fontsize=fontsize_labels + 2, 
                         fontweight='bold', family=font_family)
        
        # Griglia
        if show_gridlines and use_cmap:
            ax.set_xticks(np.arange(size + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(size + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='black', linestyle=line_style, linewidth=linewidth)
        
        # Margini
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        
        # Stile sfondo
        bg_colors = ['white', '#f5f5f5', '#fafafa', '#f0f0f0']
        fig.patch.set_facecolor(random.choice(bg_colors))
        
        # Salva
        plt.tight_layout()
        
        # Crea directory se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva con DPI casuale per variare la risoluzione
        dpi = random.randint(80, 200)
        filename = f'{output_dir}/confusion_matrix_{idx+1}.png'
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor=fig.get_facecolor())
        plt.close()
        
        return filename
    
    def generate_dataset(self, num_charts):
        """Genera un dataset di confusion matrices"""
        print(f"Generazione di {num_charts} confusion matrices...")
        
        filenames = []
        for i in range(num_charts):
            matrix = self.generate_matrix()
            filename = self.create_chart(matrix, i)
            filenames.append(filename)
            
            if (i + 1) % 10 == 0:
                print(f"Generati {i + 1}/{num_charts} grafici...")
        
        print(f"\n✓ Completato! {num_charts} grafici salvati nella cartella 'confusion_matrices'")
        return filenames


# Utilizzo
if __name__ == "__main__":
    generator = ConfusionMatrixGenerator()
    
    # Richiedi all'utente quanti grafici vuole
    num_charts = int(input("Quanti grafici confusion matrix vuoi generare? "))
    
    # Genera il dataset
    filenames = generator.generate_dataset(num_charts)
    
    print(f"\nFile generati:")
    for f in filenames[:5]:  # Mostra solo i primi 5
        print(f"  - {f}")
    if len(filenames) > 5:
        print(f"  ... e altri {len(filenames) - 5} file")