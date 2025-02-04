from DatasetLoader import DatasetLoader
from CCN1D import CCN1D

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy, FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset


dataset = DatasetLoader("dataset/Cell01_44SOH_25degC_10SOC_4406.xlsx", 2, 1)

dataset.debug()

