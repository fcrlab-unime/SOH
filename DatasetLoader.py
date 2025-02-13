import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
import torch
import os
import ast

class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample_values = self.dataframe.iloc[idx, :-1].values  
        label = self.dataframe.iloc[idx, -1]

        processed_sample = []
        print(f"Sample values: {sample_values}")
        for s in sample_values[:len(sample_values)-1]:
            print(f"s: {s}")
            if isinstance(s, str):  
                try:
                    processed_sample.extend(map(float, ast.literal_eval(s)))
                except (ValueError, SyntaxError) as e:
                    print(f"Errore nella conversione: {s} -> {e}")
                    processed_sample.extend([0.0, 0.0, 0.0])
            elif isinstance(s, (int, float)):  
                processed_sample.append(float(s))

        penultima_colonna = float(sample_values[-1])
        processed_sample.append(penultima_colonna)

        sample_tensor = torch.tensor(processed_sample, dtype=torch.float32)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        return sample_tensor, label_tensor

class DatasetLoader():

    def __init__(self, path: str, write_file: bool, read_file: bool):
        self.write_file = write_file
        self.read_file = read_file
        self.path = path
    
    def get_soh(self, filename: str):
        return filename.split('_')[1].split("SOH")[0]
    
    def get_temperature(self, filename: str):
        return filename.split('_')[2].split("degC")[0]
    
    def format_dataset(self) -> pd.DataFrame:
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

        if self.write_file:
            self.write_dataset(dataset)
        return dataset
    
    def write_dataset(self, dataset=None):
        if self.write_file and not self.read_file:
            filename = str(input("Input the filename for the dataset:\n"))
            if dataset == None:
                self.format_dataset().to_excel(filename+".xlsx", index=False)
            else:
                dataset.to_excel(filename+".xlsx", index=False)
    
    def load_dataset(self, num_partitions: int, partition_id: int):
        
        if not self.read_file:
            dataset = CustomDataset(self.format_dataset())
        else:
            filename = str(input("Input the filename of the dataset to read from .xlsx:\n"))
            dataset = pd.read_excel(filename+".xlsx")
            dataset = CustomDataset(dataset)

        partition_size = len(dataset) // num_partitions
        remainder = len(dataset) % num_partitions

        partition_sizes = [partition_size + 1 if i < remainder else partition_size for i in range(num_partitions)]

        partitions = random_split(dataset, partition_sizes)
        partition = partitions[partition_id]

        test_size = int(0.2 * len(partition))
        train_size = len(partition) - test_size
        train_dataset, test_dataset = random_split(partition, [train_size, test_size])

        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        trainloader = DataLoader(train_dataset, shuffle=True, batch_size=32,drop_last=True)
        valloader = DataLoader(val_dataset, shuffle=True, batch_size=32,drop_last=True)
        testloader = DataLoader(test_dataset, shuffle=False)

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
        
        dataset = CustomDataset(reshaped_df)
        sample, label = dataset[0]
        print("Sample shape:", sample.shape)
        print("Sample:", sample)
        print("Label:", label)

        