import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

train_data = "./data/merged_data.csv"

BATCH_SIZE = 128


class SubjectDataset(Dataset):
    def __init__(self, path, scaler=None) -> None:
        self.data = pd.read_csv(path)
        self.X = self.data.iloc[:, :-2]
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        self.Y = self.data["label"]

    def __getitem__(self, index):
        feature = self.X[index, :]
        label = self.Y[index]
        return torch.from_numpy(feature).float(), label

    def __len__(self):
        return len(self.data)


train_dataset = SubjectDataset(train_data)
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
