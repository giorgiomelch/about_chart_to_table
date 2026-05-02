#!/usr/bin/env python3
"""
Generatore di Grafici a Corda per Training AI - VERSIONE MIGLIORATA
Genera migliaia di chord diagrams con massima variabilità e qualità
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import string
import os
from pathlib import Path
from matplotlib.patches import Wedge
import colorsys

# Configurazione per evitare warning
import warnings
warnings.filterwarnings('ignore')

class ChordDiagramGenerator:
    def __init__(self):
        self.styles = ['default', 'seaborn-v0_8', 'ggplot', 'bmh']
        self.color_schemes = [
            'rainbow', 'pastel', 'vibrant', 'monochrome', 
            'complementary', 'neon', 'earth', 'ocean', 'fire'
        ]
        
    def generate_random_text(self, min_len=3, max_len=12):
        """Genera testo random"""
        length = random.randint(min_len, max_len)
        chars = string.ascii_letters + string.digits + ' '
        return ''.join(random.choice(chars) for _ in range(length)).strip()
    
    def generate_color_scheme(self, n_colors, scheme_type):
        """Genera schema di colori basato sul tipo"""
        colors = []
        
        if scheme_type == 'rainbow':
            colors = [plt.cm.rainbow(i/n_colors) for i in range(n_colors)]
        elif scheme_type == 'pastel':
            for i in range(n_colors):
                h = random.random()
                s = random.uniform(0.25, 0.5)
                v = random.uniform(0.85, 0.95)
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.7, 0.95),))
        elif scheme_type == 'vibrant':
            for i in range(n_colors):
                h = random.random()
                s = random.uniform(0.8, 1.0)
                v = random.uniform(0.8, 1.0)
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.75, 1.0),))
        elif scheme_type == 'monochrome':
            base_hue = random.random()
            for i in range(n_colors):
                v = 0.3 + (0.7 * i / n_colors)
                rgb = colorsys.hsv_to_rgb(base_hue, random.uniform(0.1, 0.3), v)
                colors.append(rgb + (random.uniform(0.7, 1.0),))
        elif scheme_type == 'complementary':
            base_hue = random.random()
            for i in range(n_colors):
                if i % 2 == 0:
                    h = base_hue
                else:
                    h = (base_hue + 0.5) % 1.0
                rgb = colorsys.hsv_to_rgb(h, random.uniform(0.6, 0.9), random.uniform(0.7, 0.95))
                colors.append(rgb + (random.uniform(0.75, 1.0),))
        elif scheme_type == 'neon':
            for i in range(n_colors):
                h = random.random()
                s = 1.0
                v = 1.0
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.8, 1.0),))
        elif scheme_type == 'earth':
            earth_hues = [0.05, 0.08, 0.12, 0.15, 0.25]
            for i in range(n_colors):
                h = random.choice(earth_hues) + random.uniform(-0.02, 0.02)
                s = random.uniform(0.3, 0.7)
                v = random.uniform(0.4, 0.7)
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.75, 0.95),))
        elif scheme_type == 'ocean':
            for i in range(n_colors):
                h = random.uniform(0.5, 0.65)
                s = random.uniform(0.5, 0.9)
                v = random.uniform(0.5, 0.9)
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.75, 0.95),))
        elif scheme_type == 'fire':
            for i in range(n_colors):
                h = random.uniform(0.0, 0.15)
                s = random.uniform(0.7, 1.0)
                v = random.uniform(0.7, 1.0)
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.75, 1.0),))
        else:
            base_hue = random.random()
            for i in range(n_colors):
                h = (base_hue + (i * 0.1)) % 1.0
                s = random.uniform(0.5, 0.9)
                v = random.uniform(0.6, 0.95)
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb + (random.uniform(0.75, 1.0),))
        
        return colors
    
    def draw_chord_diagram(self, ax, matrix, labels, colors, params):
        """Disegna il chord diagram con label interne/esterne variabili"""
        n = len(matrix)
        
        # Parametri
        gap = params['gap']
        arc_width = params['arc_width']
        show_labels = params['show_labels']
        label_position = params['label_position']  # 'inside', 'outside', 'on_arc'
        rotation = params['rotation']
        label_size = params['label_size']
        
        # Calcola dimensioni degli archi in base ai dati
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        node_values = row_sums + col_sums
        
        # Se tutti zeri, usa valori uniformi
        if np.sum(node_values) == 0:
            node_values = np.ones(n)
        
        # Converti in angoli
        total_value = np.sum(node_values)
        node_angles = (node_values / total_value) * 2 * np.pi
        
        # Applica gap
        gap_angle = gap * np.pi / 180
        total_gap = gap_angle * n
        
        # Scala gli angoli per fare spazio ai gap
        available_angle = 2 * np.pi - total_gap
        if available_angle > 0:
            scaling = available_angle / (2 * np.pi)
            node_angles = node_angles * scaling
        
        # Calcola posizioni angolari di inizio e fine per ogni arco
        start_angles = []
        end_angles = []
        current_angle = rotation * np.pi / 180
        
        for i in range(n):
            start_angles.append(current_angle)
            current_angle += node_angles[i]
            end_angles.append(current_angle)
            current_angle += gap_angle
        
        # Disegna gli archi
        outer_radius = 1.0
        inner_radius = outer_radius - arc_width
        
        for i in range(n):
            wedge = Wedge(
                (0, 0), 
                outer_radius,
                np.degrees(start_angles[i]), 
                np.degrees(end_angles[i]),
                width=arc_width,
                facecolor=colors[i],
                edgecolor=colors[i],
                linewidth=1,
                alpha=random.uniform(0.8, 1.0)
            )
            ax.add_patch(wedge)
            
            # Label con posizionamento variabile
            if show_labels:
                mid_angle = (start_angles[i] + end_angles[i]) / 2
                
                # POSIZIONAMENTO LABEL
                if label_position == 'outside':
                    # Fuori dal cerchio
                    label_distance = random.uniform(1.15, 1.35)
                    x = label_distance * np.cos(mid_angle)
                    y = label_distance * np.sin(mid_angle)
                    
                    rotation_deg = np.degrees(mid_angle)
                    if 90 < rotation_deg < 270:
                        rotation_deg += 180
                    
                    ha = 'left' if -90 <= rotation_deg <= 90 else 'right'
                    va = 'center'
                    
                elif label_position == 'inside':
                    # Dentro il blocco semicircolare (sull'arco)
                    label_distance = outer_radius - (arc_width / 2)
                    x = label_distance * np.cos(mid_angle)
                    y = label_distance * np.sin(mid_angle)
                    
                    rotation_deg = np.degrees(mid_angle)
                    if 90 < rotation_deg < 270:
                        rotation_deg += 180
                    
                    ha = 'center'
                    va = 'center'
                    
                else:  # 'on_arc' - sul bordo esterno dell'arco
                    label_distance = outer_radius + 0.05
                    x = label_distance * np.cos(mid_angle)
                    y = label_distance * np.sin(mid_angle)
                    
                    rotation_deg = np.degrees(mid_angle)
                    if 90 < rotation_deg < 270:
                        rotation_deg += 180
                    
                    ha = 'center'
                    va = 'bottom' if -90 <= rotation_deg <= 90 else 'top'
                
                # Colore del testo
                text_color = random.choice([
                    'black', 'white', colors[i], 
                    '#333333', '#666666', '#999999'
                ])
                
                ax.text(x, y, labels[i], 
                       ha=ha, va=va,
                       rotation=rotation_deg,
                       fontsize=label_size,
                       weight=random.choice(['normal', 'bold', 'light']),
                       color=text_color,
                       alpha=random.uniform(0.8, 1.0))
        
        # Disegna le connessioni (curve di Bezier)
        for i in range(n):
            for j in range(i+1, n):
                value = matrix[i][j] + matrix[j][i]
                
                if value > 0:
                    # Punti di partenza e arrivo
                    theta1 = (start_angles[i] + end_angles[i]) / 2
                    theta2 = (start_angles[j] + end_angles[j]) / 2
                    
                    r = inner_radius
                    x1 = r * np.cos(theta1)
                    y1 = r * np.sin(theta1)
                    x2 = r * np.cos(theta2)
                    y2 = r * np.sin(theta2)
                    
                    # Punti di controllo per la curva di Bezier
                    control_factor = random.uniform(0.3, 0.7)
                    cx1 = x1 * control_factor
                    cy1 = y1 * control_factor
                    cx2 = x2 * control_factor
                    cy2 = y2 * control_factor
                    
                    # Genera la curva
                    t = np.linspace(0, 1, 100)
                    bx = (1-t)**3 * x1 + 3*(1-t)**2*t * cx1 + 3*(1-t)*t**2 * cx2 + t**3 * x2
                    by = (1-t)**3 * y1 + 3*(1-t)**2*t * cy1 + 3*(1-t)*t**2 * cy2 + t**3 * y2
                    
                    # Colore della connessione
                    color_mode = random.choice(['source', 'target', 'blend', 'custom'])
                    
                    if color_mode == 'source':
                        line_color = colors[i]
                    elif color_mode == 'target':
                        line_color = colors[j]
                    elif color_mode == 'blend':
                        # Media dei due colori
                        c1 = np.array(colors[i][:3])
                        c2 = np.array(colors[j][:3])
                        line_color = tuple((c1 + c2) / 2) + (random.uniform(0.5, 0.9),)
                    else:  # custom
                        line_color = (random.random(), random.random(), random.random(), 
                                     random.uniform(0.5, 0.9))
                    
                    # Spessore basato sul valore
                    thickness_mode = random.choice([
                        'thin', 'medium', 'thick', 'very_thick', 'ultra_thick', 'proportional'
                    ])
                    
                    if thickness_mode == 'thin':
                        line_width = random.uniform(0.2, 1.5)
                    elif thickness_mode == 'medium':
                        line_width = random.uniform(1.5, 4.0)
                    elif thickness_mode == 'thick':
                        line_width = random.uniform(4.0, 8.0)
                    elif thickness_mode == 'very_thick':
                        line_width = random.uniform(8.0, 15.0)
                    elif thickness_mode == 'ultra_thick':
                        line_width = random.uniform(15.0, 25.0)
                    else:  # proportional
                        max_value = np.max(matrix)
                        if max_value > 0:
                            line_width = (value / max_value) * random.uniform(8, 20)
                        else:
                            line_width = random.uniform(1, 5)
                    
                    # Trasparenza
                    line_alpha = random.uniform(0.3, 0.9)
                    
                    # Stile linea
                    line_style = random.choice(['-', '--', '-.', ':'] if random.random() > 0.8 else ['-'])
                    
                    # Disegna la linea
                    ax.plot(bx, by,
                           color=line_color,
                           linewidth=line_width,
                           alpha=line_alpha,
                           linestyle=line_style,
                           solid_capstyle='round',
                           zorder=random.randint(1, 10))
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def generate_single_diagram(self, output_path, index):
        """Genera un singolo chord diagram"""
        # Numero di nodi
        n_nodes = random.randint(4, 12)
        
        # Stile e colori
        style = random.choice(self.styles)
        color_scheme = random.choice(self.color_schemes)
        
        # Matrice di adiacenza
        matrix = np.random.randint(0, 100, size=(n_nodes, n_nodes))
        
        # Rendi simmetrica
        if random.random() > 0.3:
            matrix = (matrix + matrix.T) // 2
        
        # Rendi sparsa
        if random.random() > 0.4:
            sparsity = random.uniform(0.4, 0.7)
            mask = np.random.random((n_nodes, n_nodes)) > sparsity
            matrix = matrix * mask
        
        # Label
        labels = [self.generate_random_text() for _ in range(n_nodes)]
        
        # Colori
        colors = self.generate_color_scheme(n_nodes, color_scheme)
        
        # Parametri con LABEL POSITION variabile
        params = {
            'gap': random.uniform(2, 10),
            'arc_width': random.uniform(0.08, 0.25),
            'show_labels': random.choice([True, True, True, False]),  # 75% con label
            'label_position': random.choice(['inside', 'outside', 'on_arc']),  # NUOVO!
            'rotation': random.uniform(0, 360),
            'label_size': random.randint(7, 12)
        }
        
        # Crea figura
        plt.style.use(style)
        fig_width = random.uniform(10, 14)
        fig_height = random.uniform(10, 14)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Sfondo (90% bianco)
        if random.random() < 0.9:
            white_variations = [
                'white', '#ffffff', '#fefefe', '#fdfdfd', '#fcfcfc', '#fbfbfb',
                '#fafafa', '#f9f9f9', '#f8f8f8', '#f7f7f7', '#f5f5f5', '#f3f3f3',
                '#f0f0f0', '#eeeeee', '#ebebeb', '#e8e8e8'
            ]
            bg_color = random.choice(white_variations)
        else:
            other_colors = ['black', '#1a1a1a', '#2a2a2a']
            bg_color = random.choice(other_colors)
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Titolo
        if random.random() > 0.5:
            title = self.generate_random_text(5, 20)
            ax.set_title(title, 
                        fontsize=random.randint(12, 18),
                        weight=random.choice(['normal', 'bold']),
                        pad=random.randint(15, 25))
        
        # Disegna il diagramma
        self.draw_chord_diagram(ax, matrix, labels, colors, params)
        
        # Note
        if random.random() > 0.7:
            note = self.generate_random_text(10, 35)
            fig.text(0.5, random.uniform(0.03, 0.08), note,
                    ha='center', fontsize=random.randint(7, 10),
                    style=random.choice(['normal', 'italic']))
        
        # Salva
        dpi = random.choice([100, 150, 200, 300])
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor=fig.patch.get_facecolor())
        plt.close()
        
        return True

def main():
    print("=" * 60)
    print("GENERATORE DI CHORD DIAGRAMS - VERSIONE MIGLIORATA")
    print("=" * 60)
    print()
    
    # Chiedi quanti grafici generare
    while True:
        try:
            n_diagrams = int(input("Quanti chord diagrams vuoi generare? "))
            if n_diagrams > 0:
                break
            else:
                print("Inserisci un numero positivo!")
        except ValueError:
            print("Inserisci un numero valido!")
    
    # Chiedi la cartella di output
    output_folder = input("In quale cartella vuoi salvare i grafici? (premi invio per './chord_diagrams'): ").strip()
    if not output_folder:
        output_folder = './chord_diagrams'
    
    # Crea la cartella
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"Generazione di {n_diagrams} chord diagrams in '{output_folder}'...")
    print()
    
    generator = ChordDiagramGenerator()
    
    # Genera i diagrammi
    success_count = 0
    for i in range(0, n_diagrams):
        try:
            output_path = os.path.join(output_folder, f'chord_diagram_{i+1:06d}.png')
            generator.generate_single_diagram(output_path, i)
            success_count += 1
            
            # Progresso
            if (i + 1) % 10 == 0 or (i + 1) == n_diagrams:
                percentage = ((i + 1) / n_diagrams) * 100
                print(f"Progresso: {i+1}/{n_diagrams} ({percentage:.1f}%)")
        
        except Exception as e:
            print(f"Errore nel generare il diagramma {i+1}: {e}")
    
    print()
    print("=" * 60)
    print(f"COMPLETATO!")
    print(f"Generati con successo: {success_count}/{n_diagrams} chord diagrams")
    print(f"Salvati in: {os.path.abspath(output_folder)}")
    print("=" * 60)

if __name__ == "__main__":
    main()