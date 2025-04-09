import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. Caricamento dati
df = pd.read_csv("prova2.csv")

# 2. Preparazione dati per fitting
def prepare_impedance_data(df, row_index):
    """Estrae i dati di impedanza da una riga specifica"""
    freqs = np.array([df.iloc[row_index][f'f_{i+1}'] for i in range(59)])
    real = np.array([df.iloc[row_index][f'r_{i+1}'] for i in range(59)])
    imag = np.array([df.iloc[row_index][f'i_{i+1}'] for i in range(59)])
    temp = df.iloc[row_index]['Temperature']
    soh = df.iloc[row_index]['SOH']
    return freqs, real, imag, temp, soh

# 3. Modello del circuito equivalente (Randles semplificato)
def randles_model(freq, Rs, Rct, CPE_T, CPE_P, Ws):
    """
    Modello circuito equivalente Randles con CPE
    Rs: resistenza di serie
    Rct: resistenza al trasferimento di carica
    CPE_T, CPE_P: parametri elemento a fase costante
    Ws: elemento Warburg short
    """
    omega = 2 * np.pi * freq
    Z_cpe = 1 / ((1j * omega) ** CPE_P * CPE_T)
    Z_w = Ws / np.sqrt(1j * omega)
    
    Z_faradaic = 1 / (1/Rct + 1/Z_w)
    Z_parallel = 1 / (1/Z_cpe + 1/Z_faradaic)
    Z = Rs + Z_parallel
    
    return np.hstack((Z.real, Z.imag))

# 4. Fit del modello per ogni riga
def fit_model_to_row(df, row_index):
    freqs, real, imag, temp, soh = prepare_impedance_data(df, row_index)
    
    # Parametri iniziali
    p0 = [0.01, 0.05, 1e-3, 0.8, 0.1]
    
    # Ottimizzazione
    try:
        data = np.hstack((real, imag))
        popt, _ = curve_fit(lambda f, Rs, Rct, CPE_T, CPE_P, Ws: 
                          randles_model(f, Rs, Rct, CPE_T, CPE_P, Ws), 
                          freqs, data, p0=p0, bounds=([0, 0, 0, 0.5, 0], [1, 1, 0.1, 1, 1]))
        return popt, temp, soh
    except:
        return None

# 5. Estrazione parametri da tutto il dataset
def extract_parameters(df):
    params = []
    temps = []
    sohs = []
    
    for i in range(len(df)):
        result = fit_model_to_row(df, i)
        if result:
            popt, temp, soh = result
            params.append(popt)
            temps.append(temp)
            sohs.append(soh)
    
    return np.array(params), np.array(temps), np.array(sohs)

# 6. Generazione di dati sintetici usando campionamento con perturbazioni
def generate_synthetic_cell_data(params, temps, sohs, n_cells=8, samples_per_cell=1200):
    """
    Genera dati sintetici utilizzando un approccio di campionamento statistico
    """
    # Calcola media e covarianza dei parametri
    params_mean = np.mean(params, axis=0)
    params_cov = np.cov(params.T)
    
    # Trova correlazioni tra parametri, temperatura e SOH
    # Creiamo una matrice completa di correlazione
    full_data = np.column_stack((params, temps.reshape(-1, 1), sohs.reshape(-1, 1)))
    full_mean = np.mean(full_data, axis=0)
    full_cov = np.cov(full_data.T)
    
    synthetic_data = []
    
    for cell_id in range(n_cells):
        # Genera una base di parametri per questa cella
        cell_base_params = multivariate_normal.rvs(mean=params_mean, cov=params_cov)
        
        # Determina un range di SOH per questa cella
        max_soh = 100 - (cell_id * 5 % 15)  # Genera celle con diversi SOH massimi
        min_soh = max(60, max_soh - 25)     # SOH minimo (non scende sotto 60%)
        
        # Aggiungi decadimento naturale del SOH basato sul numero di cicli
        cycle_numbers = np.arange(samples_per_cell)
        base_sohs = max_soh - (max_soh - min_soh) * (cycle_numbers / samples_per_cell)
        
        # Aggiungi un po' di rumore
        soh_noise = np.random.normal(0, 0.5, samples_per_cell)
        cell_sohs = base_sohs + soh_noise
        cell_sohs = np.clip(cell_sohs, min_soh, max_soh)
        
        # Genera pattern di temperatura con variazioni stagionali
        base_temp = 25 + (cell_id % 3) * 5  # Temperature di base diverse
        seasonal_variation = 10 * np.sin(np.linspace(0, 4*np.pi, samples_per_cell))
        random_variation = np.random.normal(0, 2, samples_per_cell)
        cell_temps = base_temp + seasonal_variation + random_variation
        cell_temps = np.clip(cell_temps, 10, 45)  # Limita a temperature ragionevoli
        
        for i in range(samples_per_cell):
            # Genera parametri per questo campione
            # Usa le correlazioni osservate per modificare i parametri in base a temp e SOH
            
            # Calcola quanto modifica rispetto alla baseline
            delta_soh = cell_sohs[i] - np.mean(sohs)
            delta_temp = cell_temps[i] - np.mean(temps)
            
            # Trova gli indici dei parametri, temp e SOH nella matrice di correlazione
            param_indices = range(params.shape[1])
            temp_index = params.shape[1]
            soh_index = params.shape[1] + 1
            
            # Calcola l'effetto di temp e SOH sui parametri
            temp_effect = np.zeros(params.shape[1])
            soh_effect = np.zeros(params.shape[1])
            
            # Applica l'effetto basato sulle correlazioni
            for p_idx in param_indices:
                # Normalizza l'effetto usando la deviazione standard
                temp_effect[p_idx] = delta_temp * full_cov[p_idx, temp_index] / full_cov[temp_index, temp_index]
                soh_effect[p_idx] = delta_soh * full_cov[p_idx, soh_index] / full_cov[soh_index, soh_index]
            
            # Parametri finali con effetti di temp e SOH + leggero rumore
            current_params = cell_base_params + temp_effect + soh_effect
            current_params += np.random.normal(0, 0.01, size=params.shape[1])  # Piccolo rumore
            
            # Applica vincoli per parametri fisicamente significativi
            current_params[0] = max(0.001, min(0.5, current_params[0]))  # Rs
            current_params[1] = max(0.001, min(0.5, current_params[1]))  # Rct
            current_params[2] = max(0.0001, min(0.1, current_params[2]))  # CPE_T
            current_params[3] = max(0.5, min(1.0, current_params[3]))     # CPE_P
            current_params[4] = max(0.001, min(0.5, current_params[4]))  # Ws
            
            # Aggiungi alla lista
            synthetic_data.append({
                'cell_id': f'Cell_{cell_id+1}',
                'params': current_params,
                'temp': cell_temps[i],
                'soh': cell_sohs[i]
            })
    
    # Estrai i risultati
    all_params = np.array([item['params'] for item in synthetic_data])
    all_temps = np.array([item['temp'] for item in synthetic_data])
    all_sohs = np.array([item['soh'] for item in synthetic_data])
    all_cell_ids = [item['cell_id'] for item in synthetic_data]
    
    return all_params, all_temps, all_sohs, all_cell_ids

# 7. Ricostruire curve di impedenza dai parametri sintetici
def reconstruct_impedance_curves(synthetic_params, freqs):
    synthetic_curves = []
    
    for params in synthetic_params:
        Rs, Rct, CPE_T, CPE_P, Ws = params
        try:
            Z = randles_model(freqs, Rs, Rct, CPE_T, CPE_P, Ws)
            real = Z[:len(freqs)]
            imag = Z[len(freqs):]
            synthetic_curves.append((real, imag))
        except:
            # Se c'è un problema, usa valori predefiniti
            real = np.linspace(0.01, 0.1, len(freqs))
            imag = -np.linspace(0.01, 0.1, len(freqs))
            synthetic_curves.append((real, imag))
    
    return synthetic_curves

# 8. Creare dataset sintetico finale
def create_synthetic_dataset(synthetic_curves, synthetic_temps, synthetic_sohs, cell_ids, freqs):
    data = []
    
    for i, ((real, imag), temp, soh, cell_id) in enumerate(zip(synthetic_curves, synthetic_temps, synthetic_sohs, cell_ids)):
        row = {'Cell': cell_id, 'Temperature': temp, 'SOH': soh}
        
        for j in range(len(freqs)):
            row[f'f_{j+1}'] = freqs[j]
            row[f'r_{j+1}'] = real[j]
            row[f'i_{j+1}'] = imag[j]
        
        data.append(row)
    
    return pd.DataFrame(data)

# Estrai le frequenze comuni
freqs = np.array([df.iloc[0][f'f_{i+1}'] for i in range(59)])

# Estrai parametri dai dati reali
params, temps, sohs = extract_parameters(df)

# Genera parametri sintetici per 8 celle, 1200 misurazioni ciascuna
synthetic_params, synthetic_temps, synthetic_sohs, cell_ids = generate_synthetic_cell_data(
    params, temps, sohs, n_cells=8, samples_per_cell=1200)

# Ricostruisci curve
synthetic_curves = reconstruct_impedance_curves(synthetic_params, freqs)

# Crea dataset sintetico
synthetic_df = create_synthetic_dataset(
    synthetic_curves, synthetic_temps, synthetic_sohs, cell_ids, freqs)

# Verifica la distribuzione dei dati
print(synthetic_df['Cell'].value_counts())

# Salva il dataset
synthetic_df.to_csv("dataset_sintetico_eis_8celle.csv", index=False)