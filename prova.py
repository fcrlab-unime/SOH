import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# --- Classi Modello ---
from CCN1D import CCN1D 

# --- Configurazione ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
CSV_PATH = "dataset_sorted.csv"
TARGET_COL = "soh"

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# --- Dataset ---
class CustomDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- Funzioni di Training & Test ---
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        inputs = inputs.unsqueeze(2) 

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = inputs.unsqueeze(2)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_r2 = r2_score(labels, preds)
    return avg_loss, avg_r2

# --- Main ---
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # 1. Pulizia Colonne e NaN
    cols_to_remove = [c for c in df.columns if c.lower().startswith(('f_', 'soc', 'id'))]
    df = df.drop(columns=cols_to_remove)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 2. Stratificazione per Regressione
    # Creiamo dei bins basati sui decili del target per lo split stratificato
    y_binned = pd.qcut(df[TARGET_COL], q=10, labels=False, duplicates='drop')

    # 3. Preprocessing
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 4. Stratified Split (80% train, 10% val, 10% test)
    # Primo split: Train vs (Val + Test)
    idx_train, idx_temp = train_test_split(
        np.arange(len(df)), 
        test_size=0.2, 
        random_state=42, 
        stratify=y_binned
    )
    
    # Secondo split: Val vs Test
    y_binned_temp = y_binned.iloc[idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp, 
        test_size=0.5, 
        random_state=42, 
        stratify=y_binned_temp
    )

    full_dataset = CustomDataset(X_scaled, y_scaled)
    train_ds = Subset(full_dataset, idx_train)
    val_ds = Subset(full_dataset, idx_val)
    test_ds = Subset(full_dataset, idx_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 5. Training
    num_features = X.shape[1]
    model = CCN1D(input_channels=num_features, hidden_channels=1024, num_layers=6, dropout=0.2).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=PATIENCE, path='best_model.pth')

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_r2 = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val R2: {val_r2:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggerato.")
            break

    # 6. Export ONNX
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    dummy_input = torch.randn(1, num_features, 1).to(DEVICE)
    torch.onnx.export(model, dummy_input, "model_output.onnx", 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Modello esportato in ONNX.")