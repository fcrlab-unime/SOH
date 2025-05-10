import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

real_data = pd.read_csv('original_dataset.csv')
synthetic_data = pd.read_csv('ctgan.csv')

row = synthetic_data.iloc[1]

# Estrai le colonne con prefisso r_ (X) e i_ (Y)
x_values = [row[col] for col in synthetic_data.columns if col.startswith('r_')]
y_values = [row[col] for col in synthetic_data.columns if col.startswith('i_')]

if len(x_values) > 1:  # Assicura di avere abbastanza punti
    interp_func = interp1d(x_values, y_values, kind='cubic', fill_value='extrapolate')
    x_interp = np.linspace(min(x_values), max(x_values), 100)
    y_interp = interp_func(x_interp)
    plt.plot(x_interp, y_interp, label='Interpolazione Cubica', color='r')
    plt.scatter(x_values, y_values, color='b', label='Dati Originali')
else:
    plt.scatter(x_values, y_values, label='Dati Originali')

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='b', label='r vs i')
plt.xlabel('Valori r_')
plt.ylabel('Valori i_')
plt.title('Grafico r_ vs i_')
plt.legend()
plt.grid()
plt.show()





