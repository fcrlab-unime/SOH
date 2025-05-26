from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

# Caricamento
df = pd.read_csv("original_dataset.csv")

# Costruisci le colonne nell'ordine desiderato
ordered_cols = ['Cell']
for i in range(1, 60):
    ordered_cols.extend([f'f_{i}', f'r_{i}', f'i_{i}'])
ordered_cols += ['Temperature', 'SOH']

# Estrai le colonne delle frequenze e costanti
freq_cols = [col for col in ordered_cols if col.startswith(('f_', 'r_', 'i_'))]
freqs = df[freq_cols].iloc[0]

# Rimuovi le frequenze dal dataset per l'addestramento
df_synth = df.drop(columns=freq_cols)

# Crea e configura i metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_synth)
metadata.update_column("Cell", sdtype="id")
metadata.set_sequence_key("Cell")

# Addestra il sintetizzatore
#synthesizer = PARSynthesizer(metadata=metadata, epochs=10000, verbose=True)
#synthesizer.fit(df_synth)
#synthesizer.save("synthesizer_no_f.pkl")

synthesizer = PARSynthesizer.load("synthesizer_no_f.pkl")

# Genera dati sintetici
synthetic_data = synthesizer.sample(num_sequences=8, sequence_length=625)

# Aggiungi le frequenze costanti
for col in freq_cols:
    synthetic_data[col] = freqs[col]

# Riordina le colonne secondo la lista specifica
synthetic_data = synthetic_data[ordered_cols]

# Salva il dataset sintetico
synthetic_data.to_csv("synthetic_eis_dataset.csv", index=False)
