import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
import torch
import os
import ast

class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample_values = self.dataframe.iloc[idx, :-1].values  # Tutte le colonne tranne l'ultima (etichetta)
        label = self.dataframe.iloc[idx, -1]  # L'ultima colonna è l'etichetta

        processed_sample = []
        for s in sample_values[:-1]:  # Prendi tutti tranne la penultima colonna
            if isinstance(s, str):  
                try:
                    processed_sample.extend(map(float, ast.literal_eval(s)))  # Appiattisci la lista
                except (ValueError, SyntaxError) as e:
                    print(f"Errore nella conversione: {s} -> {e}")
                    processed_sample.extend([0.0, 0.0, 0.0])  # Valore predefinito in caso di errore
            elif isinstance(s, (int, float)):  
                processed_sample.append(float(s))  # Se è un numero, aggiungilo direttamente
            else:
                print(f"Tipo sconosciuto: {s} ({type(s)})")
                processed_sample.extend([0.0, 0.0, 0.0])

        # Aggiungi la penultima colonna (che è un valore singolo) al sample
        penultima_colonna = float(sample_values[-1])
        processed_sample.append(penultima_colonna)

        # Converti in tensore PyTorch
        sample_tensor = torch.tensor(processed_sample, dtype=torch.float32)
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return sample_tensor, label_tensor

class DatasetLoader():

    def __init__(self, path: str):
        self.path = path
    
    def get_soh(self, filename: str):
        return filename.split('_')[1].split("SOH")[0]
    
    def get_temperature(self, filename: str):
        return filename.split('_')[2].split("degC")[0]
    
    def load_dataset(self, num_partitions: int, partition_id: int):
        dataset = pd.DataFrame()
        for filename in os.listdir(self.path):
            f = os.path.join(self.path, filename)
            df = pd.read_excel(f)

            values = df.values.flatten()

            triplets = values.reshape(-1, 3)
            formatted_triplets = [f"[{v1}, {v2}, {v3}]" for v1, v2, v3 in triplets]


            reshaped_df = pd.DataFrame(formatted_triplets)

            tem = self.get_temperature(f)
            soh = self.get_soh(f)
            reshaped_df.loc[len(reshaped_df)] = tem
            reshaped_df.loc[len(reshaped_df)] = soh

            reshaped_df = reshaped_df.transpose()

            dataset = pd.concat([dataset, reshaped_df], axis=0, ignore_index=True)
        
        dataset = CustomDataset(dataset)

        partition_size = len(dataset) // num_partitions
        remainder = len(dataset) % num_partitions

        # Crea una lista delle dimensioni delle partizioni
        partition_sizes = [partition_size + 1 if i < remainder else partition_size for i in range(num_partitions)]

        partitions = random_split(dataset, partition_sizes)
        partition = partitions[partition_id]

        # Suddividi il dataset in train e test (80-20)
        test_size = int(0.2 * len(partition))
        train_size = len(partition) - test_size
        train_dataset, test_dataset = random_split(partition, [train_size, test_size])

        # Ulteriore suddivisione del train set in train e validation (80-20)
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Crea i DataLoader
        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return trainloader, valloader, testloader


    def debug(self):
        df = pd.read_excel("dataset/Cell05_45SOH_25degC_30SOC_4572.xlsx")

        values = df.values.flatten()

        triplets = values.reshape(-1, 3)
        formatted_triplets = [f"[{v1}, {v2}, {v3}]" for v1, v2, v3 in triplets]


        reshaped_df = pd.DataFrame(formatted_triplets)

        tem = self.get_temperature("dataset/Cell05_45SOH_25degC_30SOC_4572.xlsx")
        soh = self.get_soh("dataset/Cell05_45SOH_25degC_30SOC_4572.xlsx")
        reshaped_df.loc[len(reshaped_df)] = tem
        reshaped_df.loc[len(reshaped_df)] = soh

        reshaped_df = reshaped_df.transpose()
        
        dataset = CustomDataset(reshaped_df)
        sample, label = dataset[0]
        print("Sample shape:", sample.shape)
        print("Sample:", sample)
        print("Label:", label)

        