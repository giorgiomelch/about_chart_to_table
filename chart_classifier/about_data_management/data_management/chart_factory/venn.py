import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import random
import numpy as np
import os
from datetime import datetime
import string

def generate_random_string(min_length=3, max_length=12):
    """Genera una stringa casuale"""
    length = random.randint(min_length, max_length)
    # Mix di lettere maiuscole, minuscole e occasionalmente numeri
    chars = string.ascii_letters
    if random.random() > 0.7:  # 30% chance di includere numeri
        chars += string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_venn_diagram(index, output_dir):
    """Genera un singolo diagramma di Venn con stile completamente randomizzato"""
    
    # Decide se creare venn2 o venn3
    venn_type = random.choice([2, 3])
    
    # Dimensione figura randomizzata
    fig_width = random.uniform(6, 12)
    fig_height = random.uniform(6, 12)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 20% sfondo bianco, 80% altri colori
    if random.random() < 0.2:
        bg_color = 'white'
    else:
        bg_colors = ['lightgray', 'ivory', 'azure', 'honeydew', 'lavender', 
                     'mistyrose', 'wheat', 'lightcyan', 'lemonchiffon']
        bg_color = random.choice(bg_colors)
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Font randomizzato
    fonts = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    font_family = random.choice(fonts)
    font_size = random.randint(8, 16)
    font_weight = random.choice(['normal', 'bold', 'light'])
    
    # Colori randomizzati per i cerchi
    def random_color():
        return (random.random(), random.random(), random.random())
    
    # Genera nomi casuali per gli insiemi
    set_names = [generate_random_string() for _ in range(3)]
    
    if venn_type == 2:
        # Venn a 2 insiemi
        set1 = random.randint(50, 500)
        set2 = random.randint(50, 500)
        overlap = random.randint(10, min(set1, set2))
        
        v = venn2(subsets=(set1, set2, overlap), 
                  set_labels=(set_names[0], set_names[1]),
                  alpha=random.uniform(0.3, 0.7))
        
        # Colori randomizzati
        colors = [random_color(), random_color()]
        for idx, patch in enumerate(v.patches):
            if patch:
                patch.set_facecolor(colors[idx % 2])
                patch.set_edgecolor(random_color())
                patch.set_linewidth(random.uniform(1, 4))
                patch.set_linestyle(random.choice(['-', '--', '-.', ':']))
        
    else:
        # Venn a 3 insiemi
        set1 = random.randint(50, 500)
        set2 = random.randint(50, 500)
        set3 = random.randint(50, 500)
        overlap12 = random.randint(10, min(set1, set2) // 2)
        overlap13 = random.randint(10, min(set1, set3) // 2)
        overlap23 = random.randint(10, min(set2, set3) // 2)
        overlap123 = random.randint(5, min(overlap12, overlap13, overlap23))
        
        v = venn3(subsets=(set1, set2, overlap12, set3, overlap13, overlap23, overlap123),
                  set_labels=(set_names[0], set_names[1], set_names[2]),
                  alpha=random.uniform(0.3, 0.7))
        
        # Colori randomizzati
        colors = [random_color(), random_color(), random_color()]
        for idx, patch in enumerate(v.patches):
            if patch:
                patch.set_facecolor(colors[idx % 3])
                patch.set_edgecolor(random_color())
                patch.set_linewidth(random.uniform(1, 4))
                patch.set_linestyle(random.choice(['-', '--', '-.', ':']))
    
    # Stile testo randomizzato
    for text in v.set_labels:
        if text:
            text.set_fontfamily(font_family)
            text.set_fontsize(font_size + random.randint(2, 6))
            text.set_fontweight(font_weight)
            text.set_color(random_color())
    
    for text in v.subset_labels:
        if text:
            text.set_fontfamily(font_family)
            text.set_fontsize(font_size)
            text.set_fontweight(font_weight)
            text.set_color(random_color())
            text.set_style(random.choice(['normal', 'italic']))
    
    # Titolo con stringa casuale
    title = generate_random_string(5, 15)
    title_size = random.randint(12, 20)
    plt.title(title, fontfamily=font_family, fontsize=title_size, 
             fontweight=random.choice(['normal', 'bold']), 
             color=random_color())
    
    # Rimuovi assi con probabilità randomizzata
    if random.random() > 0.5:
        ax.axis('off')
    
    # Salva il file
    filename = f"venn_diagram_{index:04d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=random.randint(100, 300), bbox_inches='tight', 
                facecolor=fig.get_facecolor())
    plt.close()
    
    return filename

def main():
    # Chiedi all'utente quanti grafici vuole generare
    try:
        num_diagrams = int(input("Quanti diagrammi di Venn vuoi generare? "))
        if num_diagrams <= 0:
            print("Inserisci un numero positivo!")
            return
    except ValueError:
        print("Inserisci un numero valido!")
        return
    
    # Crea la directory di output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"venn_dataset_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerazione di {num_diagrams} diagrammi di Venn...")
    print(f"Cartella di output: {output_dir}\n")
    
    # Genera i diagrammi
    for i in range(2932, num_diagrams):
        filename = generate_venn_diagram(i, output_dir)
        print(f"[{i+1}/{num_diagrams}] Generato: {filename}")
    
    print(f"\n✓ Completato! {num_diagrams} diagrammi salvati in '{output_dir}'")

if __name__ == "__main__":
    main()