from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
import pandas as pd
from sdmetrics.column_pairs import StatisticMSAS

df = pd.read_csv("original_dataset.csv")

metadata = Metadata.detect_from_dataframe(df)


#synthesizer = TVAESynthesizer(metadata, verbose=True, epochs=5000)

#synthesizer.fit(df)

#synthesizer.save("tvae_model.pkl")
synthesizer = TVAESynthesizer.load("tvae_model.pkl")

while True:
    synthetic_data = synthesizer.sample(num_rows=10000)

    result = StatisticMSAS.compute(
        real_data=(df["Cell"], df["SOH"]),
        synthetic_data=(synthetic_data["Cell"], synthetic_data["SOH"]),
        statistic="median"
    )

    if result >= 0.6:
        synthetic_data.to_csv("synthetic_data.csv", index=False)

