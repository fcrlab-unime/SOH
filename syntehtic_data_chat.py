import pandas as pd
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.column_pairs import InterRowMSAS
from datetime import datetime, timedelta

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
            'r': row[f'r_{t}'],
            'i': row[f'i_{t}'],
            'f': row[f'f_{t}'],
            'Temperature': row['Temperature'],
            'SOH': row['SOH'],
        })

data_long = pd.DataFrame(records)
data_long['timestep'] = data_long['timestep'].astype(int)

# === 2. Metadata ===

metadata = Metadata.detect_from_dataframe(data=data_long)
metadata.update_column(column_name='Cell', sdtype='id')
metadata.set_sequence_key(column_name='Cell')
metadata.set_sequence_index(column_name='timestep')
metadata.validate()

# === 3. Addestramento sintetizzatore ===

synthesizer = PARSynthesizer(metadata=metadata, verbose=True, epochs=500)
synthesizer.fit(data_long)
synthesizer.save('synthesizer_prova2.pkl')
print("✅ Synthesizer salvato.")

# === 4. Generazione e valutazione dati sintetici ===

while True:
    synthetic_data = synthesizer.sample(num_sequences=8, sequence_length=59)

    # Aggiungo il timestamp anche ai dati sintetici
    synthetic_data['timestamp'] = synthetic_data['timestep'].apply(
        lambda x: base_time + timedelta(seconds=int(x))
    )

    try:
        score = InterRowMSAS.compute(
            real_data=data_long,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_names=('Cell', 'SOH')
        )
    except Exception as e:
        print(f"⚠️ Errore durante la valutazione: {e}")
        score = 0.0

    print(f"📊 InterRowMSAS score: {score:.4f}")

    if score >= 0.75:
        synthetic_data.to_csv('synthetic_data.csv', index=False)
        print("💾 Dati sintetici salvati in synthetic_data.csv")
        break
