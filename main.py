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

import torch

DEVICE = torch.device('cpu')

NUM_PARTITIONS = 2

dataset = DatasetLoader("dataset/", 2, 1)


trainloader, valloader, _ = dataset.load_dataset()


def client_fn(context: Context) -> Client:
    input_channels = 1
    hidden_channels = 16
    num_layers = 3  
    net = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    net = net.to(DEVICE)
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()

def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=10) #se non viene passata la strategi usa FedAvg

    return ServerAppComponents(config=config)


client = ClientApp(client_fn=client_fn)

server = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
)

