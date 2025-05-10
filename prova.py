from DatasetLoader import DatasetLoader
from CCN1D import CCN1D
from Transformer import Transformer

from datasets import load_dataset, Dataset
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.utils.data as data
import torch.onnx
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim


import torch

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
            #labels = labels.unsqueeze(1)

            if torch.isnan(inputs).any():
                print("Warning: NaN detected in input!")
                continue

            y_pred = net(inputs)

            loss = criterion(y_pred, labels)

            total_loss+= loss.item()
            total_samples += inputs.size(0)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(y_pred.cpu().numpy())

    avg_loss = total_loss / total_samples
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    avg_r2 = r2_score(all_labels, all_preds)
    
    return avg_loss, avg_r2

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
            #labels = labels.unsqueeze(1)

            if torch.isnan(inputs).any():
                print("Warning: NaN detected in input!")
                continue

            optimizer.zero_grad()
            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            
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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



NUM_PARTITIONS = 2

class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
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


df = pd.read_csv("synthetic_dataset_knn.csv")
df_test = pd.read_csv("original_dataset.csv")

test_dataset_custom = CustomDataset(df_test)
testloader_global = DataLoader(test_dataset_custom, batch_size=len(test_dataset_custom),shuffle=False)

dataset = CustomDataset(df)


train_size = int(0.8 * len(dataset))  # 80% per il training
test_size = len(dataset) - train_size  # 20% per il test
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_channels = 178
hidden_channels = 1024
num_layers = 8

t = torch.empty(64, 178, 1)
network = CCN1D(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers)
#network = Transformer(t.shape, embed_size=8, output_size=1, num_layers=8, forward_expansion=1, heads=2, dropout = 0.1)
network.to(DEVICE)
for i in range(4):
    train(network, trainloader, epochs=15)
    print(test(network, testloader_global))

