import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
import os

class DatasetLoader():

    def __init__(self, path: str, num_partitions: int, partition_id: int):
        self.path = path
        self.num_partitions = num_partitions
        self.partition_id = partition_id
    
    def get_soh(self, filename: str):
        return filename.split('_')[1].split("SOH")[0]
    
    def get_temperature(self, filename: str):
        return filename.split('_')[2].split("degC")[0]
    
    def load_dataset(self):

        dataset = pd.DataFrame()
        for filename in os.listdir(self.path):
            f = os.path.join(self.path, filename)
            df = pd.read_excel(f)

            values = df.values.flatten()

            triplets = values.reshape(-1, 3)
            formatted_triplets = [f"[{v1}, {v2}, {v3}]" for v1, v2, v3 in triplets]


            reshaped_df = pd.DataFrame(formatted_triplets)

            tem = self.get_temperature(f)
            soh = self.get_soh(f)
            reshaped_df.loc[len(reshaped_df)] = tem
            reshaped_df.loc[len(reshaped_df)] = soh

            reshaped_df = reshaped_df.transpose()

            dataset = pd.concat([dataset, reshaped_df], axis=0, ignore_index=True)
  

        dataset = Dataset.from_pandas(df)

        partition = dataset.shard(self.num_partitions, index=self.partition_id)

        train_test = partition.train_test_split(test_size=0.2)

        train = train_test["train"]

        test = train_test["test"]

        train, val = train.train_test_split(test_size=0.2)

        trainloader = DataLoader(train, batch_size=32)

        valloader = DataLoader(val, batch_size=32)

        testloader = DataLoader(test, batch_size=32)

        return trainloader, valloader, testloader


    def debug(self):
        df = pd.read_excel("dataset/Cell05_45SOH_25degC_30SOC_4572.xlsx")

        values = df.values.flatten()

        triplets = values.reshape(-1, 3)
        formatted_triplets = [f"[{v1}, {v2}, {v3}]" for v1, v2, v3 in triplets]


        reshaped_df = pd.DataFrame(formatted_triplets)

        tem = self.get_temperature("dataset/Cell05_45SOH_25degC_30SOC_4572.xlsx")
        soh = self.get_soh("dataset/Cell05_45SOH_25degC_30SOC_4572.xlsx")
        reshaped_df.loc[len(reshaped_df)] = tem
        reshaped_df.loc[len(reshaped_df)] = soh

        reshaped_df = reshaped_df.transpose()
        
        
        
        print(reshaped_df)