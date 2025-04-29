from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
import pandas as pd
import numpy as np

df = pd.read_csv("original_dataset.csv")

metadata = Metadata.detect_from_dataframe(df)


synthesizer = TVAESynthesizer(metadata, verbose=True, epochs=5000)

synthesizer.fit(df)

synthesizer.save("tvae_model.pkl")

synthetic_data = synthesizer.sample(num_rows=10000)

synthetic_data.to_csv("synthetic_data.csv", index=False)