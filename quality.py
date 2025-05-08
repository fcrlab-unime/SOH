import pandas as pd
from sdmetrics.single_table import TableStructure
from sdmetrics.column_pairs import StatisticMSAS
from sdmetrics.single_column import StatisticSimilarity
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import Metadata

real_data = pd.read_csv('original_dataset.csv')
synthetic_data = pd.read_csv('synthetic_data.csv')
metadata = Metadata.detect_from_dataframe(real_data)

quality = QualityReport()

print(StatisticMSAS.compute(
    real_data=(real_data["Cell"], real_data["SOH"]),
    synthetic_data=(synthetic_data["Cell"], synthetic_data["SOH"]),
    statistic="median"
))

print(StatisticSimilarity.compute(
    real_data=real_data["SOH"],
    synthetic_data=synthetic_data["SOH"],
    statistic="median"
))


quality.generate(real_data, synthetic_data, metadata.to_dict()["tables"]["table"])