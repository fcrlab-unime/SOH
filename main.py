from DatasetLoader import DatasetLoader
from CCN1D import CCN1D
from FlowerClient import FlowerClient
from typing import Dict, Callable, List, Tuple

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

dataset = DatasetLoader("dataset/", write_file=False, read_file=True, dataset_filename="prova3.csv")


def client_fn(context: Context) -> Client:
    input_channels = 178
    hidden_channels = 256
    num_layers = 5
    network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    network = network.to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, testloader = dataset.load_dataset(num_partitions, partition_id)
    return FlowerClient(partition_id, network, trainloader, valloader).to_client()

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    "ritorna una funzione con le configurazioni per il training"

    def fit_config(server_round: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(64),
        }

        return config

    return fit_config

def aggregate_metrics(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Aggrega i valori di r2_score dai client
    r2_scores = [r["mean_r2"] for _, r in results if "mean_r2" in r]
    return {"r2_score": r2_scores}

def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=15) #se non viene passata la strategi usa FedAvg

    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=get_on_fit_config_fn(),
    )

    return ServerAppComponents(strategy=strategy, config=config)

client = ClientApp(client_fn=client_fn)

server = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
)
