import pandas as pd
import numpy as np
import glob
import os



data_root = 'data/TrainingData/'
X_TIME = 'subject_{}_{}__x_time.csv'
X_DATA = 'subject_{}_{}__x.csv'

Y_TIME = 'subject_{}_{}__y_time.csv'
Y_DATA = 'subject_{}_{}__y.csv'

def downsample(self, x, y, win_len=0.1):
        agg_x = pd.DataFrame()
        for center in y['time']:
            window_start = center - win_len/2
            window_end = center + win_len/2
            windowed_signal = x[(x["time"] > window_start) & (x["time"] < window_end)][['accr_x', 'accr_y','accr_z','gyr_x','gyr_y','gyr_z']]
            aggregate_data = windowed_signal.mean().to_frame().T
            aggregate_data['time'] =  center
            agg_x = pd.concat([agg_x, aggregate_data], axis=0)
        
        return pd.concat([agg_x.reset_index(drop=True), y['label']], axis=1)

def preprocess():
    files_x_data = sorted(glob.glob(data_root+X_DATA.format('*', '*')))
    files_x_time = sorted(glob.glob(data_root+X_TIME.format('*', '*')))

    files_y_data = sorted(glob.glob(data_root+Y_DATA.format('*', '*')))
    files_y_time = sorted(glob.glob(data_root+Y_TIME.format('*', '*')))

    merged_data = pd.DataFrame()

    for i in range(len(files_x_data)):
        y_data = pd.read_csv(files_y_data[i], names=['label'])
        y_time = pd.read_csv(files_y_time[i], names=['time'])

        x_data = pd.read_csv(files_x_data[i], names=['accr_x', 'accr_y','accr_z','gyr_x','gyr_y','gyr_z'])
        x_time = pd.read_csv(files_x_time[i], names=['time'])

        x_data = pd.concat([x_data, x_time], axis=1)
        y_data = pd.concat([y_data, y_time], axis=1)
        
        dwn_smp = downsample(x_data, y_data)
        
        merged_data = pd.concat([merged_data, dwn_smp], axis=0).reset_index(drop=True)

    return merged_data

def preprocessor(reload=False):
    if reload:
        merged_data = preprocess()
        merged_data.to_csv('data/merged_data.csv', index=False)
    else:
        if os.path.exists('data/merged_data.csv'):
            merged_data = pd.read_csv('data/merged_data.csv')
        else:
            merged_data = preprocess()
            merged_data.to_csv('data/merged_data.csv', index=False)
    
    return merged_data



    