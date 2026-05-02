import matplotlib.pyplot as plt
import numpy as np
import random
import string
from pathlib import Path
from tqdm import tqdm
import matplotlib.patches as mpatches

# Configurazione per evitare warning
import warnings
warnings.filterwarnings('ignore')

class ParetoChartGenerator:
    def __init__(self, output_dir="pareto_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Palette di colori molto più diverse
        self.color_palettes = [
            ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
            ['#2C3E50', '#E74C3C', '#ECF0F1', '#3498DB', '#95A5A6'],
            ['#8B4513', '#D2691E', '#CD853F', '#DEB887', '#F4A460'],
            ['#4A148C', '#7B1FA2', '#9C27B0', '#BA68C8', '#E1BEE7'],
            ['#1B5E20', '#388E3C', '#4CAF50', '#81C784', '#C8E6C9'],
            ['#E65100', '#F57C00', '#FF9800', '#FFB74D', '#FFCC80'],
            ['#006064', '#00838F', '#00ACC1', '#26C6DA', '#80DEEA'],
            ['#BF360C', '#D84315', '#E64A19', '#FF5722', '#FF8A65'],
            ['#FF1744', '#F50057', '#D500F9', '#651FFF', '#3D5AFE'],
            ['#00B8D4', '#00BFA5', '#00C853', '#64DD17', '#AEEA00'],
            ['#FFD600', '#FFAB00', '#FF6D00', '#DD2C00', '#C51162'],
            ['#1A237E', '#311B92', '#4A148C', '#880E4F', '#BF360C'],
        ]
        
        # Stili di pattern per le barre
        self.bar_patterns = ['', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '//', '\\\\', '||', '--', '++', 'xx', '..', '**']
        
        # Font disponibili
        self.fonts = ['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy']
        
        # Stili di grid
        self.grid_styles = [
            {'visible': True, 'linestyle': '-', 'alpha': 0.3},
            {'visible': True, 'linestyle': '--', 'alpha': 0.5},
            {'visible': True, 'linestyle': ':', 'alpha': 0.4},
            {'visible': True, 'linestyle': '-.', 'alpha': 0.6},
            {'visible': False},
        ]
        
        # Caratteri possibili per stringhe casuali
        self.char_sets = [
            string.ascii_uppercase,
            string.ascii_lowercase,
            string.ascii_uppercase + string.digits,
            string.ascii_lowercase + string.digits,
            string.ascii_letters,
            string.ascii_letters + string.digits,
        ]
    
    def generate_random_string(self, length=None):
        """Genera una stringa completamente casuale"""
        if length is None:
            length = random.randint(3, 12)
        
        char_set = random.choice(self.char_sets)
        
        # Varia il formato
        format_type = random.choice(['plain', 'prefix', 'suffix', 'combined', 'numeric'])
        
        if format_type == 'plain':
            return ''.join(random.choices(char_set, k=length))
        elif format_type == 'prefix':
            prefix = random.choice(['X', 'Y', 'Z', 'A', 'B', 'P', 'Q', 'R'])
            return prefix + ''.join(random.choices(char_set, k=length-1))
        elif format_type == 'suffix':
            suffix = random.choice(['1', '2', '3', 'X', 'Y', 'Z'])
            return ''.join(random.choices(char_set, k=length-1)) + suffix
        elif format_type == 'combined':
            part1 = ''.join(random.choices(char_set, k=length//2))
            part2 = ''.join(random.choices(char_set, k=length-length//2))
            separator = random.choice(['', '-', '_', '.'])
            return part1 + separator + part2
        else:  # numeric
            return ''.join(random.choices(string.digits, k=length))
    
    def generate_categories(self, n_categories):
        """Genera nomi di categorie completamente casuali"""
        categories = []
        
        # Varia il tipo di categoria
        category_type = random.choice(['random', 'random_prefix', 'random_suffix', 'mixed', 'numeric_only'])
        
        for i in range(n_categories):
            if category_type == 'random':
                # Completamente casuale
                cat = self.generate_random_string()
            elif category_type == 'random_prefix':
                # Prefisso casuale + numero
                prefix = self.generate_random_string(random.randint(2, 5))
                cat = f"{prefix}{random.randint(1, 999)}"
            elif category_type == 'random_suffix':
                # Numero + suffisso casuale
                suffix = self.generate_random_string(random.randint(2, 5))
                cat = f"{random.randint(1, 999)}{suffix}"
            elif category_type == 'numeric_only':
                # Solo numeri con formattazione varia
                num_format = random.choice(['simple', 'padded', 'long'])
                if num_format == 'simple':
                    cat = str(random.randint(1, 9999))
                elif num_format == 'padded':
                    cat = str(random.randint(1, 999)).zfill(random.randint(3, 6))
                else:
                    cat = str(random.randint(10000, 999999))
            else:  # mixed
                # Combinazione casuale
                parts = []
                n_parts = random.randint(1, 3)
                for _ in range(n_parts):
                    if random.random() > 0.5:
                        parts.append(self.generate_random_string(random.randint(2, 4)))
                    else:
                        parts.append(str(random.randint(1, 999)))
                separator = random.choice(['', '-', '_', '.', '/'])
                cat = separator.join(parts)
            
            categories.append(cat)
        
        return categories
    
    def generate_pareto_data(self, n_categories):
        """Genera dati che seguono approssimativamente il principio di Pareto"""
        # Varia il grado di concentrazione Pareto
        concentration = random.uniform(0.5, 0.95)
        
        # Varia il tipo di distribuzione
        distribution_type = random.choice(['standard', 'extreme', 'moderate', 'irregular'])
        
        values = []
        for i in range(n_categories):
            if distribution_type == 'standard':
                if i < n_categories * 0.2:
                    base_value = random.uniform(100, 1000) * concentration
                else:
                    base_value = random.uniform(10, 200) * (1 - concentration)
            elif distribution_type == 'extreme':
                if i < n_categories * 0.1:
                    base_value = random.uniform(500, 2000)
                else:
                    base_value = random.uniform(1, 50)
            elif distribution_type == 'moderate':
                base_value = random.uniform(50, 500) * (1 - i/n_categories)
            else:  # irregular
                base_value = random.uniform(10, 1000) * random.choice([0.1, 0.5, 1, 2, 5])
            
            values.append(base_value)
        
        # Shuffle per rendere più realistico
        random.shuffle(values)
        values = np.array(values)
        
        # Scala casuale per varietà maggiore
        scale_factor = random.choice([0.1, 1, 10, 100, 1000, 10000, 100000])
        values = values * scale_factor
        
        # Aggiungi rumore casuale
        if random.random() > 0.5:
            noise = np.random.uniform(0.9, 1.1, n_categories)
            values = values * noise
        
        return values
    
    def calculate_cumulative_percentage(self, values):
        """Calcola la percentuale cumulativa"""
        sorted_values = np.sort(values)[::-1]
        cumsum = np.cumsum(sorted_values)
        cumsum_percentage = (cumsum / cumsum[-1]) * 100
        return cumsum_percentage, sorted_values
    
    def create_pareto_chart(self, index):
        """Crea un singolo grafico Pareto con parametri casuali"""
        # Parametri casuali con range più ampio
        n_categories = random.randint(4, 25)
        categories = self.generate_categories(n_categories)
        values = self.generate_pareto_data(n_categories)
        
        # Ordina per valore decrescente
        sorted_indices = np.argsort(values)[::-1]
        sorted_categories = [categories[i] for i in sorted_indices]
        sorted_values = values[sorted_indices]
        
        # Calcola percentuali cumulative
        cumsum_percentage, _ = self.calculate_cumulative_percentage(values)
        
        # Verifica qualità: assicurati che ci sia variazione significativa
        if np.std(sorted_values) < np.mean(sorted_values) * 0.05:
            return None  # Dati troppo uniformi, rigenera
        
        # Selezione stile casuale
        palette = random.choice(self.color_palettes)
        grid_style = random.choice(self.grid_styles)
        font_family = random.choice(self.fonts)
        
        # Crea figura con dimensioni casuali più varie
        fig_width = random.uniform(8, 16)
        fig_height = random.uniform(5, 11)
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        
        # Colore di background più vario
        bg_colors = ['white', '#F5F5F5', '#FAFAFA', '#F0F0F0', '#FFFEF7', '#F8F8FF', '#FFF5EE']
        fig.patch.set_facecolor(random.choice(bg_colors))
        ax1.set_facecolor(random.choice(bg_colors))
        
        # Barre con colori e pattern casuali
        # Varia la strategia di colorazione
        color_strategy = random.choice(['gradient', 'random', 'single', 'alternating'])
        
        if color_strategy == 'gradient':
            # Sfumatura di un colore
            base_color = random.choice(palette)
            bar_colors = [base_color] * n_categories
        elif color_strategy == 'random':
            bar_colors = [random.choice(palette) for _ in range(n_categories)]
        elif color_strategy == 'single':
            bar_colors = [random.choice(palette)] * n_categories
        else:  # alternating
            colors = random.sample(palette, 2)
            bar_colors = [colors[i % 2] for i in range(n_categories)]
        
        # Scegli se usare pattern o colori solidi
        use_patterns = random.random() > 0.4
        patterns = random.choices(self.bar_patterns, k=n_categories) if use_patterns else ['']*n_categories
        
        # Varia lo stile delle barre
        edge_color = random.choice(['black', 'gray', 'darkgray', 'none', random.choice(palette)])
        edge_width = random.uniform(0.3, 2.5)
        bar_alpha = random.uniform(0.6, 1.0)
        
        # Disegna barre
        bars = ax1.bar(range(n_categories), sorted_values, color=bar_colors, 
                       edgecolor=edge_color, linewidth=edge_width, alpha=bar_alpha,
                       width=random.uniform(0.6, 0.95))
        
        # Applica pattern
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)
        
        # Asse primario (valori) con etichette completamente casuali
        xlabel_text = self.generate_random_string(random.randint(3, 10))
        ylabel_text = self.generate_random_string(random.randint(3, 10))
        
        ax1.set_xlabel(xlabel_text, fontsize=random.randint(9, 15), fontfamily=font_family, 
                      fontweight=random.choice(['normal', 'bold', 'light']))
        ax1.set_ylabel(ylabel_text, fontsize=random.randint(9, 15), fontfamily=font_family,
                      fontweight=random.choice(['normal', 'bold', 'light']), color=random.choice(palette))
        ax1.tick_params(axis='y', labelcolor=random.choice(palette), labelsize=random.randint(7, 11))
        
        # Rotazione label asse x più varia
        rotation = random.choice([0, 15, 30, 45, 60, 75, 90])
        ha = 'right' if rotation > 0 else 'center'
        ax1.set_xticks(range(n_categories))
        ax1.set_xticklabels(sorted_categories, rotation=rotation, ha=ha, 
                           fontsize=random.randint(6, 11), fontfamily=font_family)
        
        # Asse secondario (percentuale cumulativa) con più variazioni
        ax2 = ax1.twinx()
        line_color = random.choice(['red', 'darkred', 'crimson', 'orangered', 'darkblue', 'navy', 
                                   'darkgreen', 'purple', 'darkmagenta', 'darkorange'])
        line_style = random.choice(['-', '--', '-.', ':'])
        line_width = random.uniform(1.5, 4.5)
        marker_style = random.choice(['o', 's', 'D', '^', 'v', '*', 'p', 'h', 'H', '+', 'x', ''])
        marker_size = random.randint(4, 10)
        
        ax2.plot(range(n_categories), cumsum_percentage, color=line_color, marker=marker_style,
                linestyle=line_style, linewidth=line_width, markersize=marker_size,
                alpha=random.uniform(0.7, 1.0))
        
        ylabel2_text = self.generate_random_string(random.randint(3, 12))
        ax2.set_ylabel(ylabel2_text, fontsize=random.randint(9, 15), 
                      fontfamily=font_family, fontweight=random.choice(['normal', 'bold', 'light']), 
                      color=line_color)
        ax2.tick_params(axis='y', labelcolor=line_color, labelsize=random.randint(7, 11))
        ax2.set_ylim(0, random.choice([100, 105, 110]))
        
        # Linea di riferimento (80% o altra)
        if random.random() > 0.25:
            ref_value = random.choice([70, 75, 80, 85, 90])
            ref_style = random.choice([':', '--', '-.'])
            ax2.axhline(y=ref_value, color='gray', linestyle=ref_style, 
                       linewidth=random.uniform(0.8, 2.5), alpha=random.uniform(0.4, 0.8))
            if random.random() > 0.4:
                ax2.text(n_categories*random.uniform(0.85, 0.98), ref_value+random.randint(1, 4), 
                        f'{ref_value}%', fontsize=random.randint(7, 10), 
                        color='gray', fontfamily=font_family)
        
        # Titolo completamente casuale
        title_templates = [
            self.generate_random_string(),
            f'{self.generate_random_string()} - {self.generate_random_string()}',
            f'Pareto: {self.generate_random_string()}',
            f'Chart {self.generate_random_string()}',
            f'{self.generate_random_string()} Analysis',
            f'Report {self.generate_random_string()}',
        ]
        title = random.choice(title_templates)
        plt.title(title, fontsize=random.randint(11, 18), fontfamily=font_family,
                 fontweight=random.choice(['normal', 'bold', 'light']), 
                 pad=random.randint(10, 30))
        
        # Grid con più variazioni
        if grid_style['visible']:
            ax1.grid(True, linestyle=grid_style['linestyle'], alpha=grid_style['alpha'], 
                    axis=random.choice(['both', 'y', 'x']),
                    color=random.choice(['gray', 'lightgray', 'darkgray']))
        
        # Spine (bordi) variabili
        if random.random() > 0.5:
            for spine in ax1.spines.values():
                spine.set_linewidth(random.uniform(0.5, 2.0))
                spine.set_color(random.choice(['black', 'gray', 'darkgray']))
        
        # Layout
        plt.tight_layout()
        
        # Salva con parametri variabili
        filename = self.output_dir / f"pareto_{index:05d}.jpg"
        dpi = random.randint(80, 200)
        quality = random.randint(75, 95)
        plt.savefig(filename, format='jpg', dpi=dpi, bbox_inches='tight', 
                   pil_kwargs={'quality': quality})
        plt.close()
        
        return filename
    
    def generate_dataset(self, n_charts):
        """Genera l'intero dataset di grafici Pareto"""
        print(f"\n🎨 Generazione di {n_charts} grafici Pareto...")
        print(f"📁 Cartella output: {self.output_dir}\n")
        
        successful = 3500 
        failed = 0
        
        with tqdm(total=n_charts, desc="Progresso", unit="grafico") as pbar:
            while successful < n_charts:
                result = self.create_pareto_chart(successful)
                
                if result is not None:
                    successful += 1
                    pbar.update(1)
                else:
                    failed += 1
        
        print(f"\n✅ Generazione completata!")
        print(f"📊 Grafici creati con successo: {successful}")
        if failed > 0:
            print(f"⚠️  Grafici scartati (controllo qualità): {failed}")
        print(f"📁 Percorso: {self.output_dir.absolute()}\n")


def main():
    print("=" * 60)
    print("   GENERATORE DATASET PARETO CHART - VERSIONE AVANZATA")
    print("=" * 60)
    
    # Input utente
    while True:
        try:
            n_charts = int(input("\n🔢 Quanti grafici Pareto vuoi generare? "))
            if n_charts > 0:
                break
            else:
                print("❌ Inserisci un numero positivo!")
        except ValueError:
            print("❌ Inserisci un numero valido!")
    
    # Genera dataset
    generator = ParetoChartGenerator()
    generator.generate_dataset(n_charts)


if __name__ == "__main__":
    main()