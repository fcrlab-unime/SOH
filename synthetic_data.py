import pandas as pd
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.timeseries import LSTMDetection


data = pd.read_csv('prova.csv')

metadata = Metadata.detect_from_dataframe(data=data)

metadata.update_column(column_name="Cell", sdtype="id")

metadata.set_sequence_key(column_name='Cell')

metadata.validate()

synthetizer = PARSynthesizer(metadata=metadata, epochs=230, verbose=True)

synthetizer.fit(data)

synthetizer.save('my_synthesizer_230.pkl')
