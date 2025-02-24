import flwr
from flwr.client import Client, ClientApp, NumPyClient

import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics import R2Score
import torch.optim as optim
from sklearn.metrics import r2_score

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable

DEVICE = torch.device('cpu')

def get_parameters(net) -> List[np.ndarray]:
    """
    Function to get the parameters of the network
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

"""def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)"""

def set_parameters(net, parameters):
    """
    Function to set the parameters of the network
    """
    state_dict = {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)}
    net.load_state_dict(state_dict, strict=False) 

def train(net, trainloader, epochs: int):
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    net.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        all_labels = []
        all_preds = []
        total_samples = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.unsqueeze(2)

            if torch.isnan(inputs).any():
                print("Warning: NaN detected in input!")
                continue

            optimizer.zero_grad()
            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            
            #total_loss += loss.item() * inputs.size(0)
            total_loss += loss.item()
            total_samples += inputs.size(0)

            loss.backward()
            optimizer.step()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(y_pred.detach().cpu().numpy())

        avg_loss = total_loss / total_samples
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        avg_r2 = r2_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Average R2: {avg_r2:.4f}")


def test(net, testloader):
    net.eval()
    
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    all_labels = []
    all_preds = []
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.unsqueeze(2)

            if torch.isnan(inputs).any():
                print("Warning: NaN detected in input!")
                continue

            y_pred = net(inputs)

            loss = criterion(y_pred, labels)
            #total_loss += loss.item() * inputs.size(0)
            total_loss+= loss.item()
            total_samples += inputs.size(0)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(y_pred.cpu().numpy())

    avg_loss = total_loss / total_samples
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    avg_r2 = r2_score(all_labels, all_preds)
    
    return avg_loss, avg_r2

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=5)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, mean_r2 = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"mean_r2": float(mean_r2)}