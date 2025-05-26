from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

# Caricamento
df = pd.read_csv("orginal_dataset.csv")

# Estrai le colonne delle frequenze (quelle che iniziano con 'f_')
freq_cols = [col for col in df.columns if col.startswith('f_')]
freqs = df[freq_cols].iloc[0]  # prendi una sola riga: le frequenze sono costanti

# Rimuovi le frequenze dal dataset per l'addestramento del sintetizzatore
df_synth = df.drop(columns=freq_cols)

# Crea e configura i metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_synth)
metadata.set_sequence_key("Cell")

# Addestra il sintetizzatore
synthesizer = PARSynthesizer(metadata=metadata, epochs=10000, verbose=True)
synthesizer.fit(df_synth)
synthesizer.save("synthesizer_no_f.pkl")

# Genera dati sintetici
synthetic_data = synthesizer.sample(num_sequences=8, sequence_length=625)

# Aggiungi le frequenze come colonne a tutti i campioni generati
for col in freq_cols:
    synthetic_data[col] = freqs[col]

# Salva il dataset sintetico
synthetic_data.to_csv("synthetic_eis_dataset.csv", index=False)
