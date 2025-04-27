from DatasetLoader import DatasetLoader
from CCN1D import CCN1D
from FlowerClient import FlowerClient
from Transformer import Transformer

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
import torch.onnx

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


df = pd.read_csv("full_dataset_sintetico.csv")
df_test = pd.read_csv("original_dataset.csv")

test_dataset_custom = CustomDataset(df_test)
testloader_global = DataLoader(test_dataset_custom, shuffle=False)

dataset = CustomDataset(df)


train_size = int(0.8 * len(dataset))  # 80% per il training
test_size = len(dataset) - train_size  # 20% per il test
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_channels = 180
hidden_channels = 256
num_layers = 5

t = torch.empty(64, 179, 1)
network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)
#network = Transformer(t.shape, embed_size=8, output_size=1, num_layers=8, forward_expansion=1, heads=2, dropout = 0.1)

for i in range(4):
    train(network, trainloader, epochs=15)
    print(test(network, testloader_global))

