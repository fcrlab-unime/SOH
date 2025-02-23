import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
import torch
import os
import ast
import numpy as np


class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        self.dataframe.iloc[:, :-1] = self.dataframe.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0)

        self.mean = self.dataframe.iloc[:, :-1].mean()
        self.std = self.dataframe.iloc[:, :-1].std()

        self.std.replace(0, 1, inplace=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, -1]

        sample_values = ((self.dataframe.iloc[idx, :-1] - self.mean) / self.std).values.astype(np.float32)

        sample_tensor = torch.from_numpy(sample_values)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        return sample_tensor, label_tensor



class DatasetLoader():

    def __init__(self, path: str, write_file: bool, read_file: bool, dataset_filename: str = None):
        """
        Class to load the dataset

        Args:
            path (str): Path to the dataset folder.
            write_file (bool): Whether to write the formatted dataset to a file.
            read_file (bool): Whether to read the formatted dataset from a file.
            dataset_filename (str): Name of the dataset file to read or write. Defaults to None.
        """
        self.write_file = write_file
        self.read_file = read_file
        self.path = path
        self.datase_filename = dataset_filename
    
    def get_soh(self, filename: str):
        """
        Return the SOH
        """
        return filename.split('_')[1].split("SOH")[0]
    
    def get_temperature(self, filename: str):
        """
        Return the temperature
        """
        return filename.split('_')[2].split("degC")[0]
    
    def format_dataset(self) -> pd.DataFrame:
        """
        Formats the dataset where each row is a single measurement.
        """
        dataset = pd.DataFrame()
        
        for filename in os.listdir(self.path):
            f = os.path.join(self.path, filename)
            df = pd.read_excel(f)

            if df.isna().sum().sum() > 0:
                print(f"Warning: Ignoring file {filename} due to NaN values in the original file.")
                continue

            values = df.values.flatten()

            df = pd.DataFrame(values)

            if df.isna().sum().sum() > 0:
                print(f"Warning: Ignoring file {filename} due to NaN values after flattening.")
                continue

            tem = self.get_temperature(f)
            soh = self.get_soh(f)

            if pd.isna(tem) or pd.isna(soh):
                print(f"Warning: Ignoring file {filename} due to NaN in temperature or SOH.")
                continue

            df.loc[len(df)] = tem
            df.loc[len(df)] = soh

            df = df.transpose()

            if df.isna().sum().sum() > 0:
                print(f"Warning: Ignoring file {filename} due to NaN values before concatenation.")
                continue


            dataset = pd.concat([dataset, df], axis=0, ignore_index=True)

        dataset = dataset.dropna(axis=0, how="any").reset_index(drop=True)

        if self.write_file:
            self.write_dataset(dataset)
    
        return dataset
    
    def write_dataset(self, dataset=None):
        if self.write_file and not self.read_file and self.datase_filename is not None:
            if dataset is None:
                dataset = self.format_dataset()

        with open(self.datase_filename, mode='w', newline='') as f:
            dataset.to_csv(f, index=False)
    
    def load_dataset(self, num_partitions: int, partition_id: int):
        """
        Load the dataset from file and partition it

        Returns:
            DataLoader instances
        """
        if not self.read_file:
            dataset = CustomDataset(self.format_dataset())
        else:
            dataset = pd.read_csv(self.datase_filename)
            dataset = dataset.dropna(axis=0, how="any").reset_index(drop=True)
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

        trainloader = DataLoader(train_dataset, shuffle=True, batch_size=128,drop_last=True)
        valloader = DataLoader(val_dataset, shuffle=True, batch_size=128,drop_last=True)
        testloader = DataLoader(test_dataset, shuffle=False)

        return trainloader, valloader, testloader


        