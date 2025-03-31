import pandas as pd
import os

full_dataset = pd.DataFrame()

for f in os.listdir(os.getcwd()):
    if f.endswith(".csv"):
        df = pd.read_csv(f)
        full_dataset = pd.concat([full_dataset, df])

full_dataset.to_csv("full_dataset.csv", index=False)