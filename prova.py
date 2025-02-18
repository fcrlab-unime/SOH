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


dataset = DatasetLoader("dataset/", write_file=True, read_file=False, dataset_filename="prova3.xlsx")
trainloader, valloader, _ = dataset.load_dataset(2, 1)


input_channels = 178
hidden_channels = 32
num_layers = 3  
network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)

train(network, trainloader, 50)