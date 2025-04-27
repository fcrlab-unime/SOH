from DatasetLoader import DatasetLoader
from CCN1D import CCN1D
from FlowerClient import FlowerClient
from typing import Dict, Callable, List, Tuple
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data

import pandas as pd
import numpy as np
from datasets import Dataset

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy, FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner

import torch

class CustomDataset(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        """
        Class to transform the Pandas dataframe to Pytorch dataset. Also standardizes the data.
        """
        self.dataframe = dataframe
        if "Battery" in self.dataframe.columns:
            self.dataframe = self.dataframe.drop(labels=["Battery"], axis=1)
        if "Cell" in self.dataframe.columns:
            self.dataframe = self.dataframe.drop(labels=[ "Cell"], axis=1)
        
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



def data_loader(partition: Dataset):
    partition = CustomDataset(partition.to_pandas())
    test_size = int(0.2 * len(partition))
    train_size = len(partition) - test_size
    train_dataset, test_dataset = random_split(partition, [train_size, test_size])

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=32, drop_last=True)
    valloader = DataLoader(val_dataset, shuffle=True, batch_size=32, drop_last=True)
    testloader = DataLoader(test_dataset, shuffle=False)

    return trainloader, valloader, testloader

DEVICE = torch.device('cpu')

NUM_PARTITIONS = 5


#dataset = DatasetLoader("dataset/", dataset_filename="full_dataset.csv")

dataset = pd.read_csv("full_dataset_sintetico.csv")
test_dataset = pd.read_csv("original_dataset.csv")

partitioner = PathologicalPartitioner(num_classes_per_partition=1, partition_by="Battery", num_partitions=NUM_PARTITIONS, class_assignment_mode="first-deterministic")
partitioner.dataset = Dataset.from_pandas(dataset)

test_dataset_custom = CustomDataset(test_dataset)
testloader_global = DataLoader(test_dataset_custom, shuffle=False)



def client_fn(context: Context) -> Client:
    input_channels = 178
    hidden_channels = 256
    num_layers = 5
    network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    network = network.to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = data_loader(partitioner.load_partition(partition_id))

    return FlowerClient(partition_id, network, trainloader, testloader_global).to_client()


def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    "ritorna una funzione con le configurazioni per il training"

    def fit_config(server_round: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
        }

        return config

    return fit_config

def aggregate_metrics(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Aggrega i valori di r2_score dai client
    r2_scores = [r["mean_r2"] for _, r in results if "mean_r2" in r]
    return {"r2_score": r2_scores}

def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=15)

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
