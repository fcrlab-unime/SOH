import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Caricamento dati
# Assumi che df sia il tuo dataframe con i dati EIS
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

# 6. Variational Autoencoder per generare nuovi parametri
def build_vae(input_dim, latent_dim=2):
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(16, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    
    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(8, activation='relu')(decoder_inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    decoder_outputs = tf.keras.layers.Dense(input_dim)(x)
    
    # Definire i modelli
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name='decoder')
    
    # Definire il VAE collegando encoder e decoder
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = tf.keras.Model(encoder_inputs, outputs, name='vae')
    
    # Definire la loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.mean_squared_error(encoder_inputs, outputs),
            axis=1
        )
    )
    
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    
    vae_loss = reconstruction_loss + kl_loss
    
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae, encoder, decoder

# 7. Generare dati sintetici
def generate_synthetic_data(params, temps, sohs, n_samples=1000):
    # Preprocessare i parametri
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params)
    
    # Costruire VAE condizionato
    input_dim = params.shape[1] + 2  # parametri + temp + soh
    
    # Preparare dati condizionati
    X = np.column_stack((params_scaled, (temps - temps.mean()) / temps.std(), 
                         (sohs - sohs.mean()) / sohs.std()))
    
    # Creare e addestrare VAE
    vae, encoder, decoder = build_vae(input_dim)
    vae.fit(X, epochs=100, batch_size=32, verbose=0)
    
    # Campionare variabile latente
    z_sample = np.random.normal(size=(n_samples, 2))  # 2 è latent_dim
    
    # Generare nuovi dati
    synthetic_data = decoder.predict(z_sample)
    
    # Separare parametri, temp e soh
    synthetic_params = synthetic_data[:, :-2]
    synthetic_temps = synthetic_data[:, -2]
    synthetic_sohs = synthetic_data[:, -1]
    
    # Denormalizzare
    synthetic_params = scaler.inverse_transform(synthetic_params)
    synthetic_temps = synthetic_temps * temps.std() + temps.mean()
    synthetic_sohs = synthetic_sohs * sohs.std() + sohs.mean()
    
    return synthetic_params, synthetic_temps, synthetic_sohs

# 8. Ricostruire curve di impedenza dai parametri sintetici
def reconstruct_impedance_curves(synthetic_params, freqs):
    synthetic_curves = []
    
    for params in synthetic_params:
        Rs, Rct, CPE_T, CPE_P, Ws = params
        Z = randles_model(freqs, Rs, Rct, CPE_T, CPE_P, Ws)
        real = Z[:len(freqs)]
        imag = Z[len(freqs):]
        synthetic_curves.append((real, imag))
    
    return synthetic_curves

# 9. Creare dataset sintetico finale
def create_synthetic_dataset(synthetic_curves, synthetic_temps, synthetic_sohs, freqs):
    data = []
    
    for i, ((real, imag), temp, soh) in enumerate(zip(synthetic_curves, synthetic_temps, synthetic_sohs)):
        row = {'Cell': f'Synth_{i}', 'Temperature': temp, 'SOH': soh}
        
        for j in range(len(freqs)):
            row[f'f_{j+1}'] = freqs[j]
            row[f'r_{j+1}'] = real[j]
            row[f'i_{j+1}'] = imag[j]
        
        data.append(row)
    
    return pd.DataFrame(data)

# Modifica alla funzione per generare dati per celle specifiche
def generate_cell_specific_data(params, temps, sohs, n_cells=8, samples_per_cell=1200):
    """
    Genera dati sintetici specifici per un numero di celle target
    params: parametri del modello dai dati reali
    temps: temperature dai dati reali
    sohs: SOH dai dati reali
    n_cells: numero di celle da generare
    samples_per_cell: numero di misurazioni per cella
    """
    # Preprocessare i parametri
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params)
    
    # Costruire VAE condizionato
    input_dim = params.shape[1] + 2  # parametri + temp + soh
    
    # Normalizzare temperature e SOH
    temp_scaler = StandardScaler()
    soh_scaler = StandardScaler()
    
    temps_scaled = temp_scaler.fit_transform(temps.reshape(-1, 1)).flatten()
    sohs_scaled = soh_scaler.fit_transform(sohs.reshape(-1, 1)).flatten()
    
    # Preparare dati condizionati
    X = np.column_stack((params_scaled, temps_scaled, sohs_scaled))
    
    # Creare e addestrare VAE
    vae, encoder, decoder = build_vae(input_dim)
    vae.fit(X, epochs=100, batch_size=32, verbose=0)
    
    # Generare dati per ogni cella
    cell_data = []
    
    for cell_id in range(n_cells):
        # Per ogni cella, vogliamo generare dati che seguano un pattern di degradazione realistico
        # Assumiamo che SOH diminuisca gradualmente
        
        # Determina un range di SOH per questa cella
        # Distribuisci le celle in vari stati di salute per avere diversità
        max_soh = 100 - (cell_id * 5 % 15)  # Genera celle con diversi SOH massimi
        min_soh = max(60, max_soh - 25)     # SOH minimo (non scende sotto 60%)
        
        # Genera campioni linearmente decrescenti per SOH con un po' di rumore
        base_sohs = np.linspace(max_soh, min_soh, samples_per_cell)
        noise = np.random.normal(0, 1, samples_per_cell)  # Piccolo rumore
        cell_sohs = base_sohs + noise
        cell_sohs = np.clip(cell_sohs, min_soh, max_soh)  # Limita ai valori validi
        
        # Aggiungi variazione alla temperatura (ciclo stagionale + rumore)
        base_temp = 25 + (cell_id % 3) * 5  # Temperature di base diverse
        seasonal_variation = 10 * np.sin(np.linspace(0, 4*np.pi, samples_per_cell))  # Variazione stagionale
        random_variation = np.random.normal(0, 2, samples_per_cell)  # Variazione casuale
        cell_temps = base_temp + seasonal_variation + random_variation
        cell_temps = np.clip(cell_temps, 10, 45)  # Limita a temperature ragionevoli
        
        # Normalizza per il VAE usando gli stessi scalers di prima
        cell_temps_norm = temp_scaler.transform(cell_temps.reshape(-1, 1)).flatten()
        cell_sohs_norm = soh_scaler.transform(cell_sohs.reshape(-1, 1)).flatten()
        
        # Genera vettori latenti con piccole variazioni per simulare misurazioni della stessa cella
        z_base = np.random.normal(size=(1, 2))  # Base per questa cella
        z_variations = np.random.normal(0, 0.2, size=(samples_per_cell, 2))  # Piccole variazioni
        z_samples = z_base + z_variations  # Vettori latenti per questa cella
        
        # Genera parametri base per questa cella
        synthetic_base = decoder.predict(z_samples)
        
        # Introduci variazioni correlate alla temperatura e SOH
        for i in range(samples_per_cell):
            # Crea un nuovo punto in cui modifichiamo solo temp e SOH
            input_point = synthetic_base[i].copy()
            input_point[-2] = cell_temps_norm[i]
            input_point[-1] = cell_sohs_norm[i]
            
            # Rifai il passaggio attraverso VAE per avere coerenza
            _, _, z_new = encoder.predict(input_point.reshape(1, -1))
            cell_params_full = decoder.predict(z_new)
            
            # Estrai solo i parametri (senza temp e SOH)
            cell_params_scaled = cell_params_full[0, :-2]
            cell_params = scaler.inverse_transform(cell_params_scaled.reshape(1, -1))[0]
            
            # Aggiungi alla lista
            cell_data.append({
                'cell_id': f'Cell_{cell_id+1}',
                'params': cell_params,
                'temp': cell_temps[i],
                'soh': cell_sohs[i]
            })
    
    # Estrai i risultati
    all_params = np.array([item['params'] for item in cell_data])
    all_temps = np.array([item['temp'] for item in cell_data])
    all_sohs = np.array([item['soh'] for item in cell_data])
    all_cell_ids = [item['cell_id'] for item in cell_data]
    
    return all_params, all_temps, all_sohs, all_cell_ids
# Modifica alla funzione per creare dataset finale
def create_synthetic_dataset_with_cells(synthetic_curves, synthetic_temps, synthetic_sohs, cell_ids, freqs):
    data = []
    
    for i, ((real, imag), temp, soh, cell_id) in enumerate(zip(synthetic_curves, synthetic_temps, synthetic_sohs, cell_ids)):
        row = {'Cell': cell_id, 'Temperature': temp, 'SOH': soh}
        
        for j in range(len(freqs)):
            row[f'f_{j+1}'] = freqs[j]
            row[f'r_{j+1}'] = real[j]
            row[f'i_{j+1}'] = imag[j]
        
        data.append(row)
    
    return pd.DataFrame(data)


# Assumendo che df sia il tuo dataframe originale
# Estrai le frequenze comuni
freqs = np.array([df.iloc[0][f'f_{i+1}'] for i in range(59)])

# Estrai parametri dai dati reali
params, temps, sohs = extract_parameters(df)

# Genera parametri sintetici per 8 celle, 1200 misurazioni ciascuna
synthetic_params, synthetic_temps, synthetic_sohs, cell_ids = generate_cell_specific_data(
    params, temps, sohs, n_cells=8, samples_per_cell=1200)

# Ricostruisci curve
synthetic_curves = reconstruct_impedance_curves(synthetic_params, freqs)

# Crea dataset sintetico
synthetic_df = create_synthetic_dataset_with_cells(
    synthetic_curves, synthetic_temps, synthetic_sohs, cell_ids, freqs)

# Verifica la distribuzione dei dati
print(synthetic_df['Cell'].value_counts())

# Salva il dataset
synthetic_df.to_csv("dataset_sintetico_eis_8celle.csv", index=False)