from DatasetLoader import DatasetLoader
from CCN1D import CCN1D
from FlowerClient import FlowerClient

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy, FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
from datasets import load_dataset, Dataset
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.utils.data as data

from FlowerClient import test, train

import torch

DEVICE = torch.device('cpu')

NUM_PARTITIONS = 2

class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
        """
        Class to transform the Pandas dataframe to Pytorch dataset. Also standardizes the data.
        """
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


#dataset = DatasetLoader("dataset/", dataset_filename="prova.csv")
#partitioner = GroupedNaturalIdPartitioner(partition_by="Cell", group_size=4, sort_unique_ids=True)
#fds = FederatedDataset()
#df = pd.read_csv("prova.csv")
#dataset = Dataset.from_pandas(df)
#partitioner.dataset = dataset
#print(partitioner.num_partitions)
#partition = partitioner.load_partition(1)

#print(len(partition))

df = pd.read_csv("prova.csv")
#dataset = Dataset.from_pandas(df)
dataset = CustomDataset(df)


train_size = int(0.8 * len(dataset))  # 80% per il training
test_size = len(dataset) - train_size  # 20% per il test
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_channels = 179
hidden_channels = 2048
num_layers = 7  
network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)

print("Numero di hidden channels: ", hidden_channels)
print("Numero di layer: ", num_layers)

num_params = sum(p.numel() for p in network.parameters())
print(f"Numero totale di parametri: {num_params}")


memory_bytes = sum(p.element_size() * p.numel() for p in network.parameters())
memory_mb = memory_bytes / (1024 ** 2)  # Converti in MB
print(f"Memoria occupata dal modello: {memory_mb:.2f} MB")

for i in range(4):
    train(network, trainloader, epochs=15)
    print(test(network, testloader))


