import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import joblib  # per salvare lo scaler

# Caricamento dati reali
real_data = pd.read_csv("Battery_RUL.csv")

# Separazione feature e target
target_column = "RUL"
feature_columns = [col for col in real_data.columns if col != target_column]

# Normalizzazione delle sole feature
scaler = StandardScaler()
normalized_features = scaler.fit_transform(real_data[feature_columns])

# Salvataggio dello scaler
joblib.dump(scaler, "scaler_rul.pkl")

# Creazione DataFrame normalizzato
normalized_data = pd.DataFrame(normalized_features, columns=feature_columns)
normalized_data[target_column] = real_data[target_column].values

# Metadata per SDV
metadata = Metadata.detect_from_dataframe(normalized_data)

# Addestramento CTGAN
synthesizer = CTGANSynthesizer(metadata, epochs=20000, verbose=True)
synthesizer.fit(normalized_data)

# Salvataggio sintetizzatore
synthesizer.save("ctgan_rul.pkl")

# Generazione dati sintetici
synthetic_data = synthesizer.sample(num_rows=len(real_data))

# Denormalizzazione delle feature sintetiche
synthetic_features = synthetic_data[feature_columns]
synthetic_features_denorm = scaler.inverse_transform(synthetic_features)

# Sostituzione delle feature con la versione denormalizzata
synthetic_data[feature_columns] = synthetic_features_denorm

# Salvataggio su CSV
synthetic_data.to_csv("synthetic_battery_rul.csv", index=False)
