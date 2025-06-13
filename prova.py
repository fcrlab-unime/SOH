import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from DatasetLoader import DatasetLoader
from CCN1D import CCN1D
from Transformer import Transformer

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
TRAIN_RATIO = 0.8


# Dataset
class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop(columns=[col for col in ["Battery", "Cell"] if col in dataframe.columns])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx, :-1].values.astype(np.float32)
        label = float(self.dataframe.iloc[idx, -1])
        return torch.tensor(features), torch.tensor(label, dtype=torch.float32)


# Train
def train(model, dataloader, epochs):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        preds, labels = [], []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = inputs.unsqueeze(2)
            #targets = targets.unsqueeze(1)

            if torch.isnan(inputs).any():
                print("NaN in input, skipping batch.")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds.append(outputs.detach().cpu().numpy())
            labels.append(targets.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        avg_loss = total_loss / len(dataloader.dataset)
        avg_r2 = r2_score(labels, preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | R2: {avg_r2:.4f}")


# Test
def test(model, dataloader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    preds, labels = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = inputs.unsqueeze(2)
            #targets = targets.unsqueeze(1)

            if torch.isnan(inputs).any():
                print("NaN in input, skipping batch.")
                continue

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_r2 = r2_score(labels, preds)

    return avg_loss, avg_r2


# Main
if __name__ == "__main__":
    df_train = pd.read_csv("synthetic_battery_rul.csv")
    df_test = pd.read_csv("Battery_RUL.csv")

    columns_to_drop = [col for col in ["Battery", "Cell"] if col in df_train.columns]
    X_train = df_train.drop(columns=columns_to_drop + ["RUL"], axis=1)
    X_test = df_test.drop(columns=columns_to_drop + ["RUL"], axis=1)
    y_train = df_train["RUL"]
    y_test = df_test["RUL"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    df_train_scaled["RUL"] = y_train.values

    df_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    df_test_scaled["RUL"] = y_test.values

    full_dataset = CustomDataset(df_train_scaled)
    test_dataset = CustomDataset(df_test_scaled)

    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CCN1D(input_channels=8, hidden_channels=512, num_layers=4, dropout=0.3)
    #model = Transformer((BATCH_SIZE, 178, 1), embed_size=8, output_size=1, num_layers=8, forward_expansion=1, heads=2, dropout=0.1)
    model.to(DEVICE)

    for i in range(4):
        print(f"\n--- Training Round {i+1} ---")
        train(model, trainloader, EPOCHS)
        loss, r2 = test(model, testloader)
        print(f"Test Loss: {loss:.4f} | Test R2: {r2:.4f}")
