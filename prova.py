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

from FlowerClient import test, train

import torch

DEVICE = torch.device('cpu')


dataset = DatasetLoader("dataset/", dataset_filename="prova2.csv")
trainloader, valloader, _ = dataset.load_dataset(2, 1)

"""
input_channels = 178
hidden_channels = 256
num_layers = 5  
network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)

num_params = sum(p.numel() for p in network.parameters())
print(f"Numero totale di parametri: {num_params}")


memory_bytes = sum(p.element_size() * p.numel() for p in network.parameters())
memory_mb = memory_bytes / (1024 ** 2)  # Converti in MB
print(f"Memoria occupata dal modello: {memory_mb:.2f} MB")

torch.cuda.memory_allocated() / (1024 ** 2)  # Memoria attualmente allocata in MB
torch.cuda.max_memory_allocated() / (1024 ** 2)  # Picco massimo di memoria allocata
"""

