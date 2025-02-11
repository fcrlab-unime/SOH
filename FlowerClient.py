import flwr
from flwr.client import Client, ClientApp, NumPyClient

import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics import R2Score
import torch.optim as optim

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable

DEVICE = torch.device('cpu')

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int):
    #Addestra la rete sul training set
    criterion = nn.MSELoss()
    r2_score_metric = R2Score().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    r2_total = 0.0
    for epoch in range(epochs):
        total_loss, epoch_loss = 0.0, 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()
            y_pred = net(inputs)
            print(y_pred)
            loss = criterion(y_pred, labels)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            r2_score_metric.update(y_pred, labels)
            
        r2_score_metric.compute()

    epoch_loss /= len(trainloader.dataset)
    avg_loss = total_loss / len(trainloader)
    avg_r2 = r2_total / len(trainloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Average R2: {avg_r2:.4f}")


def test(net, testloader):
    net.eval()
    
    criterion = nn.MSELoss()
    r2_score_metric = R2Score().to(DEVICE)
    total_loss = 0.0
    r2_total = 0.0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels = labels.unsqueeze(1)

            y_pred = net(inputs)

            loss = criterion(y_pred, labels)
            total_loss += loss.item()
            
            r2_total += r2_score_metric(y_pred, labels).item()
    
    avg_loss = total_loss / len(testloader)
    avg_r2 = r2_total / len(testloader)
    loss /= len(testloader.dataset)
    
    return loss, avg_r2

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
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}