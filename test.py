import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit
import matplotlib.pyplot as plt
from tqdm import tqdm

# Carica il dataset
df = pd.read_csv("original_dataset.csv")

# Candidati di circuiti e guess iniziali
circuit_candidates = [
    ('R0-p(R1,C1)', [0.01, 0.1, 1e-4]),
    ('R0-p(R1,CPE1)', [0.01, 0.1, 1e-5, 0.9]),
    ('R0-p(R1,CPE1)-W1', [0.01, 0.1, 1e-5, 0.9, 0.01]),
    ('R0-p(R1,C1)-p(R2,C2)', [0.01, 0.1, 1e-4, 0.05, 1e-5]),
    ('R0-p(R1,CPE1)-p(R2,CPE2)', [0.01, 0.1, 1e-5, 0.9, 0.05, 5e-6, 0.8]),
]

results = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Estrai frequenze e impedenze
    frequencies = np.array([row[col] for col in df.columns if col.startswith('f_')])
    Z_real = np.array([row[col] for col in df.columns if col.startswith('r_')])
    Z_imag = np.array([row[col] for col in df.columns if col.startswith('i_')])
    Z = Z_real + 1j * Z_imag

    # Filtra NaN e zero
    mask = (~np.isnan(frequencies)) & (~np.isnan(Z.real)) & (~np.isnan(Z.imag)) & (frequencies > 0)
    frequencies_clean = frequencies[mask]
    Z_clean = Z[mask]

    best_mse = np.inf
    best_fit = None

    for circuit_string, guess in circuit_candidates:
        try:
            circuit = CustomCircuit(initial_guess=guess, circuit=circuit_string)
            circuit.fit(frequencies_clean, Z_clean)
            Z_fit = circuit.predict(frequencies_clean)
            mse = np.mean(np.abs(Z_clean - Z_fit)**2)

            if mse < best_mse:
                best_mse = mse
                best_fit = {
                    "index": idx,
                    "circuit": circuit_string,
                    "mse": mse,
                    "parameters": circuit.parameters_,
                }
        except:
            continue

    if best_fit:
        results.append(best_fit)

# Converte i risultati in DataFrame
output_df = pd.DataFrame(results)
output_df[['param_' + str(i) for i in range(output_df['parameters'].str.len().max())]] = pd.DataFrame(output_df['parameters'].tolist(), index=output_df.index)
output_df.drop(columns='parameters', inplace=True)

# Salva su CSV
output_df.to_csv("fitted_parameters_all_curves.csv", index=False)

print("✅ Stima completata per tutte le curve. Risultati salvati in 'fitted_parameters_all_curves.csv'")
