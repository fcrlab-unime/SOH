import pandas as pd
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.column_pairs import InterRowMSAS

# Caricamento dei dati
data = pd.read_csv('prova2.csv')

# Standardizzazione dei dati (escludendo la colonna "Cell" che è un ID)
#numeric_cols = data.select_dtypes(include=['number']).columns.difference(['Cell', "SOH"])
#scaler = StandardScaler()
#data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Creazione della metadata
metadata = Metadata.detect_from_dataframe(data=data)
metadata.update_column(column_name="Cell", sdtype="id")
metadata.set_sequence_key(column_name='Cell')
metadata.validate()

# Creazione e addestramento del sintetizzatore
#synthesizer = PARSynthesizer(metadata=metadata, verbose=True, epochs=7000)
#synthesizer.fit(data)

#synthesizer.save('my_synthesizer_gpu_7000.pkl')

synthesizer = PARSynthesizer.load('my_synthesizer_gpu_7000.pkl')


while True:
    # Generazione dei dati sintetici
    generated_data = synthesizer.sample(num_sequences=8, sequence_length=1250)

    # Applicazione della trasformazione inversa
    #generated_data[numeric_cols] = scaler.inverse_transform(generated_data[numeric_cols])

    result = InterRowMSAS.compute(
        real_data=(data["Cell"], data["SOH"]),
        synthetic_data=(generated_data["Cell"], generated_data["SOH"]),
    )

    print(result)
    if result >= 0.6:
        # Salvataggio dei dati sintetici
        generated_data.to_csv('synthetic_data_gpu.csv', index=False)
        break
