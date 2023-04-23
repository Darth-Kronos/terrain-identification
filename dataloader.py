import pandas as pd
import numpy as np
import os

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# train_data_root = "./data/train_data.csv"
train_data_root = "/Users/purushothamanyadav/Documents/NCSU/Spring23/NN/Project/ProjC/terrain-identification/data/TrainingData/filtered_120_60"

SUBJECT_X_TEMPLATE = "subject_{}_{}__x.csv"
SUBJECT_X_TIME_TEMPLATE = "subject_{}_{}__x_time.csv"
SUBJECT_Y_TEMPLATE = "subject_{}_{}__y.csv"
SUBJECT_Y_TIME_TEMPLATE = "subject_{}_{}__y_time.csv"

X_HEADER = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
Y_HEADER = ["label"]

BATCH_SIZE = 128

def parse_uid(uid):
    subject_id, session_num = uid.split("_")
    # return int(subject_id), int(session_num)
    return subject_id, session_num

class SubjectDataset(Dataset):

    def __init__(self, datapath, ids, scaler=None, test=False):
        
        self.ids = ids
        self.datapath = datapath
        self.data_files = {uid: os.path.join(self.datapath, SUBJECT_X_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        self.scaler = scaler
        timesteps = None
        self.X = []
        self.Y = []
        for uid in ids:
            # print(f"Converting uid {uid}")
            data_path = os.path.join(self.datapath, SUBJECT_X_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1]))

            data = pd.read_csv(data_path)

            if timesteps is None:
                _sample = data[data["timestamp"] == 0]
                timesteps = len(_sample)
            
            if len(self.X) == 0:
                self.X = data[X_HEADER].values

            else:
                self.X = np.vstack([self.X, data[X_HEADER].values])

            if not test:
                temp_label = np.int_(data['label'].values.reshape(-1, timesteps))
                if len(self.Y) == 0:
                    self.Y = stats.mode(temp_label, axis=1, keepdims=True)[0]
                else:
                    self.Y = np.vstack([self.Y, stats.mode(temp_label, axis=1, keepdims=True)[0]])

        if self.scaler == None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
        
        self.X = self.scaler.transform(self.X)
        
        self.X = self.X.reshape(-1, timesteps, len(X_HEADER))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        
        inputs = self.X[index]
        labels = self.Y[index]

        return torch.from_numpy(inputs), torch.from_numpy(labels).squeeze()


split_ids = {'train': ['005_02',
  '001_06',
  '003_02',
  '001_05',
  '002_02',
  '003_01',
  '003_03',
  '005_01',
  '001_07',
  '002_05',
  '004_02',
  '002_03',
  '001_02',
  '002_04',
  '001_03',
  '004_01',
  '005_03',
  '006_01',
  '006_02',
  '007_01',
  '007_03',
  '007_04',
  
  ],
 'val': ['001_08', '002_01', '001_01', '001_04', '006_03','007_02','008_01'],
}
train_data_path = '/Users/purushothamanyadav/Documents/NCSU/Spring23/NN/Project/ProjC/terrain-identification/data/TrainingData/filtered_120_60'
train_dataset = SubjectDataset(
    train_data_path, 
    split_ids["train"]
)
ys = train_dataset.Y.T.tolist()
counts = Counter(ys[0])
weights = np.array([1./counts[_y] for _y in ys[0]])
sample_weights = torch.from_numpy(weights).float()
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
train_iterations = (len(train_dataset) // BATCH_SIZE) + ((len(train_dataset) % BATCH_SIZE) != 0)

val_dataset = SubjectDataset(
    train_data_path, 
    split_ids["val"],
    scaler=train_dataset.scaler
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


"""
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
    train_dataset, BATCH_SIZE=BATCH_SIZE, drop_last=True, sampler=sampler
)

val_dataloader = DataLoader(
    val_dataset, BATCH_SIZE=BATCH_SIZE, drop_last=True
)
"""