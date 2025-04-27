import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal

df = pd.read_csv("original_dataset.csv")

def extract_raw_impedance_curves(df):
    """Estrae tutte le curve di impedenza dal dataframe"""
    curves = []
    temps = []
    sohs = []
    
    for i in range(len(df)):
        freqs = np.array([df.iloc[i][f'f_{j+1}'] for j in range(59)])
        real = np.array([df.iloc[i][f'r_{j+1}'] for j in range(59)])
        imag = np.array([df.iloc[i][f'i_{j+1}'] for j in range(59)])
        temp = df.iloc[i]['Temperature']
        soh = df.iloc[i]['SOH']
        
        curves.append((freqs, real, imag))
        temps.append(temp)
        sohs.append(soh)
    
    return curves, np.array(temps), np.array(sohs)

def find_similar_curves(curves, temps, sohs, target_temp, target_soh, n=5):
    """Trova le curve più simili a una data temperatura e SOH"""
    distances = []
    
    for i, (_, temp, soh) in enumerate(zip(curves, temps, sohs)):
        # Normalizza le distanze per temperatura e SOH
        temp_dist = abs(temp - target_temp) / 30  # Assumendo un range di temperatura di circa 30°C
        soh_dist = abs(soh - target_soh) / 40     # Assumendo un range di SOH di circa 40%
        
        # Distanza pesata combinata
        dist = temp_dist * 0.3 + soh_dist * 0.7   # Diamo più peso al SOH
        distances.append((i, dist))
    
    # Ordina per distanza e prendi i primi n
    distances.sort(key=lambda x: x[1])
    similar_indices = [idx for idx, _ in distances[:n]]
    
    return [curves[i] for i in similar_indices]

def interpolate_curves(similar_curves, weights=None):
    """Interpola tra curve simili per generare una nuova curva"""
    if weights is None:
        # Pesi uniformi
        weights = np.ones(len(similar_curves)) / len(similar_curves)
    
    # Assumiamo che tutte le curve abbiano le stesse frequenze
    freqs = similar_curves[0][0]
    
    # Combina le parti reali e immaginarie con i pesi
    real_combined = np.zeros_like(similar_curves[0][1])
    imag_combined = np.zeros_like(similar_curves[0][2])
    
    for i, (_, real, imag) in enumerate(similar_curves):
        real_combined += real * weights[i]
        imag_combined += imag * weights[i]
    
    return freqs, real_combined, imag_combined

def add_realistic_noise(real, imag, noise_level=0.02):
    """Aggiunge rumore realistico alle parti reale e immaginaria"""
    # Aggiungi rumore proporzionale all'ampiezza della curva
    real_noise = np.random.normal(0, noise_level * np.abs(real).mean(), size=real.shape)
    imag_noise = np.random.normal(0, noise_level * np.abs(imag).mean(), size=imag.shape)
    
    # Applica un filtro per avere rumore correlato (più realistico)
    real_noise = signal.savgol_filter(real_noise, 5, 2)
    imag_noise = signal.savgol_filter(imag_noise, 5, 2)
    
    return real + real_noise, imag + imag_noise

def generate_synthetic_eis_dataset(df, n_cells=8, samples_per_cell=1200):
    """Genera un dataset sintetico di EIS basato su interpolazione di curve reali"""
    # Estrai le curve reali
    real_curves, real_temps, real_sohs = extract_raw_impedance_curves(df)
    
    # Memorizza le frequenze (assumiamo che siano le stesse per tutte le curve)
    freqs = real_curves[0][0]
    
    synthetic_data = []
    
    for cell_id in range(n_cells):
        # Determina range di SOH per questa cella (degradazione realistica)
        max_soh = min(100, 95 - (cell_id * 5 % 20))  # Varia il SOH iniziale
        min_soh = max(60, max_soh - 20 - cell_id % 10)  # Varia la velocità di degradazione
        
        # Genera SOH con trend decrescente e piccole variazioni
        base_sohs = np.linspace(max_soh, min_soh, samples_per_cell)
        # Aggiungi oscillazioni realistiche al SOH
        cycles = np.linspace(0, 8 * np.pi, samples_per_cell)
        small_oscillations = 1.5 * np.sin(cycles) + 0.8 * np.sin(2.5 * cycles)
        cell_sohs = base_sohs + small_oscillations
        cell_sohs = np.clip(cell_sohs, min_soh-3, max_soh+3).round().astype(int)
        
        # Genera temperature con pattern stagionale e utilizzo realistico
        base_temp = 22 + (cell_id % 4) * 3  # Temperature base diverse
        seasonal = 10 * np.sin(np.linspace(0, 2*np.pi, samples_per_cell))
        daily = 3 * np.sin(np.linspace(0, 40*np.pi, samples_per_cell))
        cell_temps = base_temp + seasonal + daily
        cell_temps = np.clip(cell_temps, 10, 42).round().astype(int)
        
        for i in range(samples_per_cell):
            # Trova curve simili nel dataset reale
            target_temp = cell_temps[i]
            target_soh = cell_sohs[i]
            similar_curves = find_similar_curves(real_curves, real_temps, real_sohs, 
                                                target_temp, target_soh, n=5)
            
            # Genera pesi per l'interpolazione (più peso alle curve più simili)
            weights = np.linspace(0.5, 0.1, len(similar_curves))
            weights = weights / weights.sum()
            
            # Interpola per creare una nuova curva
            _, real_interp, imag_interp = interpolate_curves(similar_curves, weights)
            
            # Aggiungi un po' di rumore realistico
            real_final, imag_final = add_realistic_noise(real_interp, imag_interp, 
                                                        noise_level=0.01 + 0.02 * np.random.random())
            
            # Assicurati che la curva abbia caratteristiche fisicamente valide
            # (Applica vincoli fisici se necessario)
            
            # Aggiungi alla lista dei dati sintetici
            row = {'Cell': f'{cell_id+1}'}
            
            for j in range(len(freqs)):
                row[f'f_{j+1}'] = freqs[j]
                row[f'r_{j+1}'] = real_final[j]
                row[f'i_{j+1}'] = imag_final[j]
                
            row['Temperature'] = target_temp
            row['SOH'] = target_soh
            
            synthetic_data.append(row)
    
    # Crea il dataframe
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Riordina le colonne
    columns = ['Cell'] + [f'f_{i+1}' for i in range(59)] + [f'r_{i+1}' for i in range(59)] + [f'i_{i+1}' for i in range(59)] + ['Temperature', 'SOH']
    synthetic_df = synthetic_df[columns]
    
    return synthetic_df

# Genera il nuovo dataset sintetico

synthetic_df = generate_synthetic_eis_dataset(df, n_cells=8, samples_per_cell=1250)

    # Salva il dataset
synthetic_df.to_csv(f"dataset_sintetico_5.csv", index=False)

# Visualizza qualche dato di esempio per verificare la qualità
# Plotta alcune curve di Nyquist per verifica
plt.figure(figsize=(12, 8))
for i in range(0, len(synthetic_df), len(synthetic_df)//16):
    real = np.array([synthetic_df.iloc[i][f'r_{j+1}'] for j in range(59)])
    imag = np.array([synthetic_df.iloc[i][f'i_{j+1}'] for j in range(59)])
    plt.plot(real, -imag)
    
plt.xlabel('Re(Z) [Ohm]')
plt.ylabel('-Im(Z) [Ohm]')
plt.title('Curve di Nyquist dei dati sintetici generati')
plt.grid(True)
plt.savefig('verifica_curve_sintetiche.png')
plt.close()