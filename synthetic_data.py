import pandas as pd
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.column_pairs import InterRowMSAS


data = pd.read_csv('original_dataset.csv')

metadata = Metadata.detect_from_dataframe(data=data)

metadata.update_column(column_name="Cell", sdtype="id")

metadata.set_sequence_key(column_name='Cell')

metadata.validate()

synthetizer = PARSynthesizer(metadata=metadata, epochs=20000, verbose=True)

synthetizer.fit(data)

synthetizer.save('parsynthesizer_20000.pkl')

while True:

    synthetic_data = synthetizer.sample(num_sequences=8)

    result = InterRowMSAS.compute(
        real_data=(data["Cell"], data["178"]),
        synthetic_data=(synthetic_data["Cell"], synthetic_data["178"]),
    )

    print(result)

    if result >= 0.7:
        synthetic_data.to_csv('synthetic_data.csv', index=False)
        break



