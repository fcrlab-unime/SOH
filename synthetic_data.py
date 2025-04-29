from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
import pandas as pd
from sdmetrics.reports.single_table import QualityReport

df = pd.read_csv("original_dataset.csv")
report = QualityReport()

metadata = Metadata.detect_from_dataframe(df)


#synthesizer = TVAESynthesizer(metadata, verbose=True, epochs=5000)

#synthesizer.fit(df)

#synthesizer.save("tvae_model.pkl")
synthesizer = TVAESynthesizer.load("tvae_model.pkl")

synthetic_data = synthesizer.sample(num_rows=10000)

report.generate(df, synthetic_data, metadata.to_dict()["tables"]["table"])
if report.get_score >= 0.9:
    synthetic_data.to_csv("synthetic_data.csv", index=False)