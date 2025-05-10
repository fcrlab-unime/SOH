import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.neighbors import NearestNeighbors

# Carica dataset reale
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

def find_similar_curves_knn(curves, temps, sohs, target_temp, target_soh, n=5):
    """Trova le curve più simili usando KNN basato su temperatura e SOH"""
    features = np.column_stack((temps, sohs))  # shape: (N, 2)
    target = np.array([[target_temp, target_soh]])

    knn = NearestNeighbors(n_neighbors=n, metric='euclidean')
    knn.fit(features)
    distances, indices = knn.kneighbors(target)

    similar_curves = [curves[i] for i in indices[0]]
    return similar_curves

def interpolate_curves(similar_curves, weights=None):
    """Interpola tra curve simili per generare una nuova curva"""
    if weights is None:
        weights = np.ones(len(similar_curves)) / len(similar_curves)
    
    freqs = similar_curves[0][0]
    real_combined = np.zeros_like(similar_curves[0][1])
    imag_combined = np.zeros_like(similar_curves[0][2])
    
    for i, (_, real, imag) in enumerate(similar_curves):
        real_combined += real * weights[i]
        imag_combined += imag * weights[i]
    
    return freqs, real_combined, imag_combined

def add_realistic_noise(real, imag, noise_level=0.02):
    """Aggiunge rumore realistico alle parti reale e immaginaria"""
    real_noise = np.random.normal(0, noise_level * np.abs(real).mean(), size=real.shape)
    imag_noise = np.random.normal(0, noise_level * np.abs(imag).mean(), size=imag.shape)
    
    real_noise = signal.savgol_filter(real_noise, 5, 2)
    imag_noise = signal.savgol_filter(imag_noise, 5, 2)
    
    return real + real_noise, imag + imag_noise

def generate_synthetic_eis_dataset(df, n_cells=8, samples_per_cell=1200):
    """Genera un dataset sintetico di EIS basato su interpolazione di curve reali"""
    real_curves, real_temps, real_sohs = extract_raw_impedance_curves(df)
    freqs = real_curves[0][0]
    
    synthetic_data = []
    
    for cell_id in range(n_cells):
        max_soh = min(100, 95 - (cell_id * 5 % 20))
        min_soh = max(60, max_soh - 20 - cell_id % 10)
        
        base_sohs = np.linspace(max_soh, min_soh, samples_per_cell)
        cycles = np.linspace(0, 8 * np.pi, samples_per_cell)
        small_oscillations = 1.5 * np.sin(cycles) + 0.8 * np.sin(2.5 * cycles)
        cell_sohs = base_sohs + small_oscillations
        cell_sohs = np.clip(cell_sohs, min_soh - 3, max_soh + 3).round().astype(int)
        
        base_temp = 22 + (cell_id % 4) * 3
        seasonal = 10 * np.sin(np.linspace(0, 2 * np.pi, samples_per_cell))
        daily = 3 * np.sin(np.linspace(0, 40 * np.pi, samples_per_cell))
        cell_temps = base_temp + seasonal + daily
        cell_temps = np.clip(cell_temps, 10, 42).round().astype(int)
        
        for i in range(samples_per_cell):
            target_temp = cell_temps[i]
            target_soh = cell_sohs[i]
            
            similar_curves = find_similar_curves_knn(real_curves, real_temps, real_sohs, 
                                                     target_temp, target_soh, n=5)
            
            weights = np.linspace(0.5, 0.1, len(similar_curves))
            weights = weights / weights.sum()
            
            _, real_interp, imag_interp = interpolate_curves(similar_curves, weights)
            real_final, imag_final = add_realistic_noise(real_interp, imag_interp, 
                                                         noise_level=0.005 + 0.01 * np.random.random())
            
            row = {'Cell': f'{cell_id+1}'}
            for j in range(len(freqs)):
                row[f'f_{j+1}'] = freqs[j]
                row[f'r_{j+1}'] = real_final[j]
                row[f'i_{j+1}'] = imag_final[j]
            row['Temperature'] = target_temp
            row['SOH'] = target_soh
            
            synthetic_data.append(row)
    
    synthetic_df = pd.DataFrame(synthetic_data)
    columns = ['Cell'] + [f for j in range(59) for f in (f'f_{j+1}', f'r_{j+1}', f'i_{j+1}')] + ['Temperature', 'SOH']
    synthetic_df = synthetic_df[columns]
    
    return synthetic_df

# Generazione ed esportazione
synthetic_df = generate_synthetic_eis_dataset(df, n_cells=8, samples_per_cell=1250)
synthetic_df.to_csv("synthetic_dataset_knn.csv", index=False)
