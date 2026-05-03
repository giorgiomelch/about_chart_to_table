import matplotlib
matplotlib.use('Agg')  # IMPORTANTE: Risolve errori Qt/Wayland

import matplotlib.pyplot as plt
import numpy as np
import random
import string
from pathlib import Path

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', 
          '#BB8FCE', '#85C1E2', '#F8B739', '#52B788', '#E63946', '#A8DADC',
          '#457B9D', '#1D3557', '#F1FAEE', '#E76F51', '#264653', '#2A9D8F']

STYLES = ['default', 'classic', 'seaborn-v0_8', 'ggplot', 'fivethirtyeight', 
          'bmh', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid']

EDGE_STYLES = ['solid', 'dashed', 'dotted', 'dashdot']

def genera_testo_random(min_len=3, max_len=12):
    """Genera stringa random"""
    lunghezza = random.randint(min_len, max_len)
    if random.random() < 0.3:
        return ''.join(random.choices(string.digits, k=lunghezza))
    elif random.random() < 0.5:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=lunghezza))
    else:
        return ''.join(random.choices(string.ascii_letters, k=lunghezza))

def genera_distribuzione():
    """Genera dati da diverse distribuzioni statistiche"""
    n_samples = random.randint(100, 5000)
    
    # Numero di distribuzioni da sovrapporre (1-4)
    n_distributions = random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0]
    
    all_data = []
    
    for _ in range(n_distributions):
        tipo = random.choice(['normal', 'exponential', 'poisson', 'uniform',
                              'beta', 'gamma', 'lognormal', 'triangular',
                              'weibull', 'pareto', 'binomial', 'chi2',
                              'multimodal', 'skewed', 'bimodal',
                              'normal', 'normal', 'poisson', 'poisson',
                              'binomial', 'gamma', 'beta'])
        
        samples = n_samples // n_distributions
        
        if tipo == 'normal':
            mu = random.uniform(-10, 100)
            sigma = random.uniform(3, 30) if random.random() < 0.6 else random.uniform(5, 50)
            data = np.random.normal(mu, sigma, samples)
        elif tipo == 'exponential':
            data = np.random.exponential(random.uniform(10, 100), samples)
        elif tipo == 'poisson':
            data = np.random.poisson(random.uniform(5, 50), samples)
        elif tipo == 'uniform':
            low = random.uniform(0, 50)
            data = np.random.uniform(low, low + random.uniform(50, 200), samples)
        elif tipo == 'beta':
            a = random.uniform(2, 8) if random.random() < 0.6 else random.uniform(0.5, 5)
            b = random.uniform(2, 8) if random.random() < 0.6 else random.uniform(0.5, 5)
            data = np.random.beta(a, b, samples) * random.uniform(50, 200)
        elif tipo == 'gamma':
            shape = random.uniform(3, 12) if random.random() < 0.6 else random.uniform(1, 10)
            data = np.random.gamma(shape, random.uniform(5, 20), samples)
        elif tipo == 'lognormal':
            mu = random.uniform(2, 5)
            sigma = random.uniform(0.3, 1.0) if random.random() < 0.6 else random.uniform(0.3, 1.5)
            data = np.random.lognormal(mu, sigma, samples)
        elif tipo == 'triangular':
            left = random.uniform(0, 50)
            mode = left + random.uniform(20, 50)
            right = mode + random.uniform(20, 50)
            data = np.random.triangular(left, mode, right, samples)
        elif tipo == 'weibull':
            a = random.uniform(2, 6) if random.random() < 0.6 else random.uniform(1, 5)
            data = np.random.weibull(a, samples) * random.uniform(20, 100)
        elif tipo == 'pareto':
            data = (np.random.pareto(random.uniform(1, 5), samples) + 1) * random.uniform(10, 50)
        elif tipo == 'binomial':
            n = random.randint(20, 100)
            p = random.uniform(0.3, 0.7)
            data = np.random.binomial(n, p, samples)
        elif tipo == 'chi2':
            data = np.random.chisquare(random.randint(2, 15), samples)
        elif tipo == 'bimodal':
            mu1 = random.uniform(20, 50)
            mu2 = random.uniform(70, 120)
            sigma = random.uniform(5, 12) if random.random() < 0.6 else random.uniform(5, 15)
            n1 = samples // 2
            data = np.concatenate([
                np.random.normal(mu1, sigma, n1),
                np.random.normal(mu2, sigma, samples - n1)
            ])
        elif tipo == 'multimodal':
            sigma = random.uniform(5, 10) if random.random() < 0.6 else random.uniform(5, 15)
            data = np.concatenate([
                np.random.normal(random.uniform(0, 100), sigma, samples // 3)
                for _ in range(3)
            ])
        else:  # skewed
            data = np.random.gamma(2, 20, samples)
            if random.random() < 0.5:
                data = -data + data.max()
        
        all_data.append(data)
    
    combined_data = np.concatenate(all_data)
    
    # Outliers ridotti
    if random.random() < 0.2:
        n_outliers = random.randint(5, 30)
        outlier_indices = random.sample(range(len(combined_data)), min(n_outliers, len(combined_data)))
        for idx in outlier_indices:
            combined_data[idx] += random.uniform(-30, 30) + random.choice([-1, 1]) * random.uniform(30, 100)
    
    return combined_data

def crea_istogramma(idx, output_dir):
    plt.clf()
    plt.close('all')
    
    # Stile random
    try:
        plt.style.use(random.choice(STYLES))
    except:
        pass
    
    # 10% istogramma vuoto
    
    # Numero sovrapposizioni: 60% uno, 25% due, 15% tre
    n_overlapping = random.choices([1, 2, 3], weights=[0.6, 0.25, 0.15])[0]
    
    # Generazione dati
    all_datasets = [genera_distribuzione() for _ in range(n_overlapping)]
    
    # Setup figura
    fig, ax = plt.subplots(figsize=(random.uniform(6, 14), random.uniform(4, 10)), 
                           dpi=random.choice([80, 100, 150]))
    
    # Parametri grafici
    n_bins = random.choice([5, 8, 10, 15, 20, 25, 30, 40, 50])
    horizontal = random.choice([True, False])
    use_attached_bars = random.random() < 0.4
    hist_type = 'bar' if use_attached_bars else random.choice(['stepfilled', 'step'])
    rwidth = 1.0 if use_attached_bars else None
    
    # Setup colori con contrasto garantito
    use_bg_color = random.random() < 0.4
    bg_color = None
    
    if use_bg_color:
        bg_color = random.choice(['#F5F5F5', '#FFFFFF', '#E8E8E8', '#FFF8DC', '#F0F8FF'])
        # Colori scuri per sfondo chiaro
        strong_colors = ['#FF6B6B', '#4ECDC4', '#1D3557', '#E76F51', '#264653', 
                        '#DC143C', '#8B008B', '#006400', '#FF4500', '#457B9D']
        colors = [random.choice(strong_colors) for _ in range(n_overlapping)]
    else:
        colors = [random.choice(COLORS) for _ in range(n_overlapping)]
    
    # Edge sempre visibile
    edge_color = random.choice(['black', 'darkblue', 'darkred', 'darkgreen'])
    edge_width = random.uniform(0.5, 2.0)
    
    # Alpha base
    alpha_base = random.uniform(0.6, 0.9) if n_overlapping > 1 else random.uniform(0.7, 1.0)
    
    # Scaling dati se necessario
    for i in range(len(all_datasets)):
        dataset = all_datasets[i]
        if len(dataset) > 0:
            data_range = dataset.max() - dataset.min()
            if data_range < n_bins * 0.1:
                all_datasets[i] = dataset * random.uniform(5, 20)
    
    # Plotting istogrammi sovrapposti
    all_n = []
    orientation = 'horizontal' if horizontal else 'vertical'
    
    for i, dataset in enumerate(all_datasets):
        if len(dataset) == 0:
            continue
        
        color = colors[i]
        alpha = alpha_base if i == 0 else alpha_base * (0.9 - i * 0.1)
        label = f'Dist {i+1}' if n_overlapping > 1 else None
        
        try:
            n, bins, patches = ax.hist(dataset, bins=n_bins, orientation=orientation,
                                       histtype=hist_type, color=color, alpha=alpha,
                                       edgecolor=edge_color, linewidth=edge_width,
                                       rwidth=rwidth, label=label)
            all_n.append(n)
            
            # Verifica visibilità del primo istogramma
            if i == 0:
                visible_bins = np.sum(n > 0)
                max_freq = np.max(n) if len(n) > 0 else 0
                min_freq = np.min(n[n > 0]) if np.any(n > 0) else 0
                
                # Se troppa disparità o pochi bins, applica trasformazione
                if visible_bins < 3 or (max_freq > 0 and min_freq > 0 and max_freq / min_freq > 100):
                    if random.random() < 0.5:
                        dataset_transformed = np.log1p(np.abs(dataset - dataset.min()))
                    else:
                        dataset_transformed = np.sqrt(np.abs(dataset - dataset.min()))
                    
                    ax.clear()
                    n, bins, patches = ax.hist(dataset_transformed, bins=n_bins, 
                                               orientation=orientation, histtype=hist_type,
                                               color=color, alpha=alpha, edgecolor=edge_color,
                                               linewidth=edge_width, rwidth=rwidth, label=label)
                    all_n[0] = n
        except:
            continue
    
    # Limiti assi
    if len(all_n) == 0:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    else:
        max_freq = max([np.max(n) for n in all_n if len(n) > 0], default=0)
        if max_freq > 0:
            if horizontal:
                ax.set_xlim(0, max_freq * 1.15)
            else:
                ax.set_ylim(0, max_freq * 1.15)
    
    # Titolo e labels
    if random.random() < 0.8:
        ax.set_title(genera_testo_random(5, 20), fontsize=random.randint(10, 22),
                    fontweight=random.choice(['normal', 'bold']))
    
    if random.random() < 0.7:
        label_x = random.choice(['Value', 'X', genera_testo_random(3, 8)])
        label_y = random.choice(['Frequency', 'Count', genera_testo_random(3, 8)])
        
        if horizontal:
            ax.set_xlabel(label_y, fontsize=random.randint(8, 14))
            ax.set_ylabel(label_x, fontsize=random.randint(8, 14))
        else:
            ax.set_xlabel(label_x, fontsize=random.randint(8, 14))
            ax.set_ylabel(label_y, fontsize=random.randint(8, 14))
    
    # Linee statistiche (solo se non vuoto e ci sono dati validi)
    if len(all_datasets[0]) > 1 and random.random() < 0.5:
        data = all_datasets[0]
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        data_range = data.max() - data.min()
        line_colors = ['red', 'darkred', 'darkgreen', 'darkblue', 'purple', 'darkorange']
        
        # Media
        if horizontal:
            ax.axhline(mean_val, color=random.choice(line_colors), linestyle='--',
                      linewidth=random.uniform(1.5, 3), alpha=0.8, label=f'Mean: {mean_val:.1f}')
        else:
            ax.axvline(mean_val, color=random.choice(line_colors), linestyle='--',
                      linewidth=random.uniform(1.5, 3), alpha=0.8, label=f'Mean: {mean_val:.1f}')
        
        # Mediana (solo se vicina)
        if random.random() < 0.5 and abs(median_val - mean_val) < data_range * 0.3:
            if horizontal:
                ax.axhline(median_val, color=random.choice(line_colors), linestyle='-.',
                          linewidth=random.uniform(1.5, 3), alpha=0.8, label=f'Median: {median_val:.1f}')
            else:
                ax.axvline(median_val, color=random.choice(line_colors), linestyle='-.',
                          linewidth=random.uniform(1.5, 3), alpha=0.8, label=f'Median: {median_val:.1f}')
        
        # ±1 std
        if random.random() < 0.3:
            thresh_low = mean_val - std_val
            thresh_high = mean_val + std_val
            
            if horizontal:
                if thresh_low > data.min():
                    ax.axhline(thresh_low, color=random.choice(line_colors), linestyle=':',
                              linewidth=random.uniform(1, 2), alpha=0.6, label=f'-1σ: {thresh_low:.1f}')
                if thresh_high < data.max():
                    ax.axhline(thresh_high, color=random.choice(line_colors), linestyle=':',
                              linewidth=random.uniform(1, 2), alpha=0.6, label=f'+1σ: {thresh_high:.1f}')
            else:
                if thresh_low > data.min():
                    ax.axvline(thresh_low, color=random.choice(line_colors), linestyle=':',
                              linewidth=random.uniform(1, 2), alpha=0.6, label=f'-1σ: {thresh_low:.1f}')
                if thresh_high < data.max():
                    ax.axvline(thresh_high, color=random.choice(line_colors), linestyle=':',
                              linewidth=random.uniform(1, 2), alpha=0.6, label=f'+1σ: {thresh_high:.1f}')
    
    # Griglia
    if random.random() < 0.5:
        ax.grid(True, alpha=random.uniform(0.2, 0.7), linestyle=random.choice(['-', '--', '-.', ':']),
               axis=random.choice(['both', 'x', 'y']))
    
    # Spine (bordi)
    if random.random() < 0.3:
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Legenda
    if (n_overlapping > 1 and random.random() < 0.6) or (random.random() < 0.4):
        ax.legend(fontsize=random.randint(8, 12), loc='best')
    
    # Background
    if bg_color:
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)
    
    # Margini
    if random.random() < 0.5:
        plt.tight_layout(pad=random.uniform(0.5, 3.0))
    
    # Salvataggio
    formato = random.choice(['png', 'jpg'])
    filename = output_dir / f'histogram_{idx:05d}.{formato}'
    plt.savefig(filename, bbox_inches='tight', dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    return filename

def main():
    try:
        n_grafici = int(input("Quanti istogrammi vuoi generare? "))
        if n_grafici <= 0:
            print("Inserisci un numero positivo!")
            return
    except ValueError:
        print("Inserisci un numero valido!")
        return
    
    output_dir = Path("dataset_istogrammi")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerazione di {n_grafici} istogrammi in corso...")
    print(f"Directory output: {output_dir.absolute()}\n")
    
    for i in range(0 , n_grafici):
        filename = crea_istogramma(i, output_dir)
        if (i + 1) % 10 == 0 or i == n_grafici - 1:
            print(f"Generati: {i + 1}/{n_grafici}")
    
    print(f"\n✓ Completato! {n_grafici} istogrammi salvati in '{output_dir}'")

if __name__ == "__main__":
    main()