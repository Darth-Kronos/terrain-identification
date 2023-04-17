import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# train_data_root = "./data/train_data.csv"
train_data_root = "/Users/purushothamanyadav/Documents/NCSU/Spring23/NN/Project/ProjC/terrain-identification/data/train_data.csv"


BATCH_SIZE = 128

class dataset(Dataset):
    def __init__(self, data, test=False,scaler=None) -> None:
        self.data = data
        self.test = test
        if self.test:
            self.X = self.data.iloc[:, :-1]
        else:
            self.X = self.data.iloc[:, :-2]
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        if not self.test:
            self.Y = self.data["label"]
        
    def __getitem__(self, index):
        feature = self.X[index, :]
        if not self.test:
            label = self.Y[index]
            return torch.from_numpy(feature).float(), label
        return torch.from_numpy(feature).float()

    def __len__(self):
        return len(self.data)

data = pd.read_csv(train_data_root)

train_data, val_data = train_test_split(data, test_size=0.2)

train_data = train_data.reset_index()
val_data = val_data.reset_index()

train_data = train_data.drop(["index"], axis=1)
val_data = val_data.drop(["index"], axis=1)

train_dataset = dataset(train_data)
val_dataset = dataset(val_data, scaler = train_dataset.scaler)

ys = train_dataset.Y.tolist()
counts = Counter(ys)
weights = np.array([1.0 / counts[_y] for _y in ys])
sample_weights = torch.from_numpy(weights).float()

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, drop_last=True, sampler=sampler
)

val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, drop_last=True
)
