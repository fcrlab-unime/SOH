import pandas as pd
import os
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from sdmetrics.column_pairs import InterRowMSAS
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport



data = pd.read_csv('prova2.csv')

metadata = Metadata.detect_from_dataframe(data=data)


synthetic_data = pd.read_csv('synthetic_data_228.csv')

report = QualityReport()
diagnostic = DiagnosticReport()

report.generate(data, synthetic_data, metadata.to_dict()["tables"]["table"])
diagnostic.generate(data, synthetic_data, metadata.to_dict()["tables"]["table"])


"""synthetizer = PARSynthesizer.load('my_synthesizer_228.pkl')


while True:

    synthetic_data = synthetizer.sample(num_sequences=80)

    result = InterRowMSAS.compute(
        real_data=(data["Cell"], data["178"]),
        synthetic_data=(synthetic_data["Cell"], synthetic_data["178"]),
    )

    print(result)

    if result >= 0.7:
        synthetic_data.to_csv('synthetic_data_228.csv', index=False)
        break
"""

