import pandas as pd
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.column_pairs import InterRowMSAS
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport
from sdv.evaluation.single_table import get_column_plot



data = pd.read_csv('prova2.csv')

metadata = Metadata.detect_from_dataframe(data=data)

synthetic_data = pd.read_csv('synthetic_data.csv')

"""fig = get_column_plot(
    real_data=data,
    synthetic_data=synthetic_data,
    column_name='SOH',
    metadata=metadata
)
    
fig.show()"""


report = QualityReport()
diagnostic = DiagnosticReport()

report.generate(data, synthetic_data, metadata.to_dict()["tables"]["table"])
diagnostic.generate(data, synthetic_data, metadata.to_dict()["tables"]["table"])

result = InterRowMSAS.compute(
        real_data=(data["Cell"], data["SOH"]),
        synthetic_data=(synthetic_data["Cell"], synthetic_data["SOH"]),
    )

print(result)


"""synthetizer = PARSynthesizer.load('my_synthesizer_230.pkl')
i = 1

while True:

    synthetic_data = synthetizer.sample(num_sequences=8, sequence_length=1250)

    result = InterRowMSAS.compute(
        real_data=(data["Cell"], data["SOH"]),
        synthetic_data=(synthetic_data["Cell"], synthetic_data["SOH"]),
    )

    print(result)

    if result >= 0.75:
        synthetic_data.insert(0, 'Battery', i)
        synthetic_data.to_csv(f'synthetic_data_{i}.csv', index=False)
        i+=1
    
    if i ==5:
        break

"""

    

