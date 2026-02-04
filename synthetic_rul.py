import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

df = pd.read_csv("Battery_RUL_id.csv")
metadata = Metadata.detect_from_dataframe(data=df)

metadata.set_sequence_key(column_name='Battery_ID')
metadata.set_sequence_index(column_name='Cycle_Index')

synthesizer = PARSynthesizer(metadata, verbose=True)
synthesizer.fit(df)

synthetic_data = synthesizer.sample(num_sequences=1)

synthetic_data.to_csv('synthetic_data.csv', index=False)