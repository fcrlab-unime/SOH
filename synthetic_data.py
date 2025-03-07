import pandas as pd
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.timeseries import LSTMDetection


data = pd.read_csv('prova2.csv')

metadata = Metadata.detect_from_dataframe(data=data)

metadata.update_column(column_name="Cell", sdtype="id")

metadata.set_sequence_key(column_name='Cell')

metadata.validate()

epochs_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

results_file = 'experiment_results.csv'

with open(results_file, 'w') as f:
    f.write('epochs,result\n')

for epochs in epochs_list:
    print(f"Running PARSynthesizer with {epochs} epochs...")
    
    synthesizer = PARSynthesizer(metadata, epochs=epochs)
    
    synthesizer.fit(data)
    
    synthesizer.save(filepath=f'my_synthesizer_{epochs}.pkl')
    
    synthetic_data = synthesizer.sample(num_sequences=100)
    
    synthetic_data.to_csv(f'synthetic_data_{epochs}.csv', index=False)
    
    result = LSTMDetection.compute(
        real_data=data,
        synthetic_data=synthetic_data,
        sequence_key='Cell',
    )
    
    with open(results_file, 'a') as f:
        f.write(f'{epochs},{result}\n')
    
    print(f"Result for {epochs} epochs: {result}")

print(f"Final results saved to {results_file}")


