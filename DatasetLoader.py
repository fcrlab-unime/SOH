import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset

class DatasetLoader():

    def __init__(self, filename: str, num_partitions: int, partition_id: int):
        self.filename = filename
        self.num_partitions = num_partitions
        self.partition_id = partition_id
    
    def get_soh(self):
        return self.filename.split('_')[1].split("SOH")[0]
    
    def get_temperature(self):
        return self.filename.split('_')[2].split("degC")[0]
    
    def load_dataset(self):

        df = pd.read_excel(self.filename)

        df.columns = ["fre", "re", "im"]

        df["tem"] = self.get_temperature()

        df["soh"] = self.get_soh()

        dataset = Dataset.from_pandas(df)

        partition = dataset.shard(self.num_partitions, index=self.partition_id)

        train, test = partition.train_test_split(test_size=0.2)

        train, val = train.train_test_split(test_size=0.2)

        trainloader = DataLoader(train, batch_size=32)

        valloader = DataLoader(val, batch_size=32)

        testloader = DataLoader(test, batch_size=32)

        return trainloader, valloader, testloader


    def debug(self):
        df = pd.read_excel(self.filename)

        df.columns = ["fre", "re", "im"]

        df["tem"] = self.get_temperature()

        df["soh"] = self.get_soh()
        
        print(df.head())