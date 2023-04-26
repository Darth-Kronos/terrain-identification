import pandas as pd
import numpy as np
import os

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# train_data_root = "./data/train_data.csv"
train_data_root = "terrain-identification/data/train_data.csv"

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

    def __init__(self, datapath: str, ids: list):
        
        self.ids = ids
        self.datapath = datapath
        self.y_files = {uid: os.path.join(self.datapath, SUBJECT_Y_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        self.x_files = {uid: os.path.join(self.datapath, SUBJECT_X_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        
        # Generate a list of samples and determine the number of datapoints in the dataset 
        # and build up the cache
        self.build_cache_and_datalen()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        
        inputs = self.X[index]
        labels = np.array([self.y[index]])

        return torch.from_numpy(inputs), torch.from_numpy(labels)

    def build_cache_and_datalen(self):

        num_samples = 0
        timesteps = None

        X_list = []
        y_list = []

        for uid, y_file in self.y_files.items():
            # print(f"Converting uid {uid}")
            y = pd.read_csv(y_file)
            n_samples = len(y)
            num_samples += n_samples

            x_file = self.x_files[uid]
            X_dataframe = pd.read_csv(x_file)
            if timesteps is None:
                _sample = X_dataframe[X_dataframe["timestamp"] == 0]
                timesteps = len(_sample)

            # Convert to numpy
            X = self.dataframe_to_numpy(X_dataframe, timesteps, y)

            X_list.append(X)
            y_list.append(y["label"].values)

        self.X = np.concatenate(X_list, axis=0).astype(np.float32)
        self.y = np.concatenate(y_list, axis=0).astype(int)
        assert self.X.shape[0] == self.y.shape[0]
        
        self.num_samples = self.X.shape[0]

    def dataframe_to_numpy(self, df, timesteps, y_df):
        """Convert from pandas to numpy for faster access
        """
        len_array = int(len(df) / timesteps)
        assert len_array == len(y_df)

        values = df[X_HEADER].values
        X = values.reshape(len_array, timesteps, len(X_HEADER))
        
        return np.transpose(X, axes=(0, 2, 1)).copy()

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
train_data_path = 'terrain-identification/data/TrainingData/filtered_window_1'
train_dataset = SubjectDataset(
    train_data_path, 
    split_ids["train"]
)
ys = train_dataset.y.tolist()
counts = Counter(ys)
weights = np.array([1./counts[_y] for _y in ys])
sample_weights = torch.from_numpy(weights).float()
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
train_iterations = (len(train_dataset) // BATCH_SIZE) + ((len(train_dataset) % BATCH_SIZE) != 0)

val_dataset = SubjectDataset(
    train_data_path, 
    split_ids["val"]
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_iterations = (len(val_dataset) // BATCH_SIZE) + ((len(val_dataset) % BATCH_SIZE) != 0)
