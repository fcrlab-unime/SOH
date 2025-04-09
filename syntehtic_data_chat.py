import pandas as pd
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.column_pairs import InterRowMSAS
from datetime import datetime, timedelta

def convert_synthetic_long_to_wide(synthetic_data):
    """
    Converte un DataFrame sequenziale sintetico (long) in formato wide,
    con colonne ordinate come f_0, r_0, i_0, ..., f_58, r_58, i_58, Temperature, SOH.
    """
    # Gestione dei duplicati: eliminiamo i duplicati o prendiamo la media
    synthetic_data = synthetic_data.groupby(['Cell', 'timestep']).agg({
        'f': 'mean',
        'r': 'mean',
        'i': 'mean',
        'Temperature': 'first',
        'SOH': 'first'
    }).reset_index()
    
    # Pivot delle variabili sequenziali
    pivot_f = synthetic_data.pivot(index='Cell', columns='timestep', values='f')
    pivot_r = synthetic_data.pivot(index='Cell', columns='timestep', values='r')
    pivot_i = synthetic_data.pivot(index='Cell', columns='timestep', values='i')
    
    # Rinomina le colonne
    pivot_f.columns = [f"f_{i}" for i in pivot_f.columns]
    pivot_r.columns = [f"r_{i}" for i in pivot_r.columns]
    pivot_i.columns = [f"i_{i}" for i in pivot_i.columns]
    
    # Colonne statiche (Temperature e SOH)
    static_cols = synthetic_data[['Cell', 'Temperature', 'SOH']].drop_duplicates(subset='Cell').set_index('Cell')
    
    # Unisci i pivot
    df_concat = pd.concat([pivot_f, pivot_r, pivot_i], axis=1)
    
    # Riordina le colonne come f_0, r_0, i_0, ..., f_58, r_58, i_58
    ordered_cols = []
    for i in range(59):
        ordered_cols.extend([f"f_{i}", f"r_{i}", f"i_{i}"])
    
    # Assicurati che tutte le colonne esistano
    for col in ordered_cols:
        if col not in df_concat.columns:
            df_concat[col] = None
    
    df_concat = df_concat[ordered_cols]
    
    # Aggiungi Temperature e SOH in fondo
    df_concat = pd.concat([df_concat, static_cols[['Temperature', 'SOH']]], axis=1).reset_index()
    return df_concat

# === 1. Caricamento e trasformazione dati ===

data_wide = pd.read_csv('prova2.csv')

# Converto il dataset wide in formato long
records = []
base_time = datetime(2024, 1, 1)

for idx, row in data_wide.iterrows():
    for t in range(1, 60):  # r_1 to r_59 (59 timesteps)
        records.append({
            'Cell': idx,
            'timestep': t - 1,
            'timestamp': base_time + timedelta(seconds=t - 1),
            'f': row[f'f_{t}'],
            'r': row[f'r_{t}'],
            'i': row[f'i_{t}'],
            'Temperature': row['Temperature'],
            'SOH': row['SOH'],
        })

data_long = pd.DataFrame(records)
data_long['timestep'] = data_long['timestep'].astype(int)

data_long.to_csv('data_long.csv', index=False)

# === 2. Metadata ===

metadata = Metadata.detect_from_dataframe(data=data_long)
metadata.update_column(column_name='Cell', sdtype='id')
metadata.set_sequence_key(column_name='Cell')
metadata.set_sequence_index(column_name='timestep')
metadata.validate()

# === 3. Addestramento sintetizzatore ===

#synthesizer = PARSynthesizer(metadata=metadata, verbose=True, epochs=500)
#synthesizer.fit(data_long)
#synthesizer.save('synthesizer_prova2.pkl')
#print("✅ Synthesizer salvato.")
synthesizer = PARSynthesizer.load("synthesizer_prova2.pkl")

# === 4. Generazione e valutazione dati sintetici ===

while True:
    synthetic_data = synthesizer.sample(num_sequences=8, sequence_length=59)

    # Aggiungo il timestamp anche ai dati sintetici
    synthetic_data['timestamp'] = synthetic_data['timestep'].apply(
        lambda x: base_time + timedelta(seconds=int(x))
    )

    synthetic_data = convert_synthetic_long_to_wide(synthetic_data)

    try:
        score = InterRowMSAS.compute(
            real_data=(data_wide["Cell"], data_wide["SOH"]),
            synthetic_data=(synthetic_data["Cell"], synthetic_data["SOH"])
        )
    except Exception as e:
        print(f"⚠️ Errore durante la valutazione: {e}")
        score = 0.0

    print(f"📊 InterRowMSAS score: {score:.4f}")

    if score >= 0.75:
        synthetic_data.to_csv('synthetic_data.csv', index=False)
        print("💾 Dati sintetici salvati in synthetic_data.csv")
        break
