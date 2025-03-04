import pandas as pd
import sdv 
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata


data = pd.read_csv('prova2.csv')

data = data.iloc[:len(data) //2]

metadata = Metadata.detect_from_dataframe(data=data)

metadata.update_column(column_name="Cell", sdtype="id")

metadata.set_sequence_key(column_name='Cell')

metadata.validate()

synthesizer = PARSynthesizer(metadata, verbose=True)

synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_sequences=100, verbose=True)

synthetic_data.to_csv('synthetic_data.csv', index=False)





