import pandas as pd
import os

full_dataset = pd.DataFrame()
real_dataset = pd.read_csv("original_dataset.csv")
real_dataset.insert(0, "Battery", 1)
synthetic_dataset = pd.read_csv("dataset_sintetico_1.csv")
synthetic_dataset.insert(0, "Battery", 2)

full_dataset = pd.concat([real_dataset, synthetic_dataset], axis=0)
full_dataset.to_csv("2_batteries.csv", index=False)

"""i = 1

for f in os.listdir(os.getcwd()):
    if "dataset" in f and f.endswith(".csv"):
        df = pd.read_csv(f)
        df.insert(0, "Battery", i)
        i += 1
        full_dataset = pd.concat([full_dataset, df])

full_dataset.to_csv("full_dataset.csv", index=False)"""



"""features = pd.read_csv("features.csv", header=None)
labels = pd.read_csv("labels.csv", header=None)

# Cast della prima colonna delle features a int
features[0] = features[0].astype(int)

# Moltiplica la colonna delle labels per 100
labels[0] = (labels[0] * 100).astype(int)

# Combina i due dataset
combined = pd.concat([features, labels], axis=1)

# Genera gli headers
num_features = features.shape[1]  # Numero di colonne nelle features
headers = ["Cell"] + [f"f_{i}" if i % 3 == 0 else f"r_{i}" if i % 3 == 1 else f"i_{i}" for i in range(1, num_features)] + ["SOH"]

# Applica gli headers al dataset
combined.columns = headers

# Salva il dataset combinato in un nuovo file CSV
combined.to_csv("feature_selection_dataset.csv", index=False)"""


