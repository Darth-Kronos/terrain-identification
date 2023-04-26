import pandas as pd
import numpy as np
import glob
import os
import os
import sys
import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


data_root = "terrain-identification/data/TestData/"
# data_root = "data/TestData/"
X_TIME = "subject_{}_{}__x_time.csv"
X_DATA = "subject_{}_{}__x.csv"

Y_TIME = "subject_{}_{}__y_time.csv"
Y_DATA = "subject_{}_{}__y.csv"

SAMPLING_RATE_X = 40 # Hz
SAMPLING_RATE_Y = 10 # Hz

X_HEADER = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
Y_HEADER = ["label"]

fs=40
cutoff=10
l_filter = 155

def median_filter(data, f_size):
	lgth, num_signal=data.shape
	f_data=np.zeros([lgth, num_signal])
	for i in range(num_signal):
		f_data[:,i]=signal.medfilt(data[:,i], f_size)
	return f_data

def freq_filter(data, f_size, cutoff):
	lgth, num_signal=data.shape
	f_data=np.zeros([lgth, num_signal])
	lpf=signal.firwin(f_size, cutoff, window='hamming')
	for i in range(num_signal):
		f_data[:,i]=signal.convolve(data[:,i], lpf, mode='same')
	return f_data


def get_windowed_data(x: pd.DataFrame, 
                        interval: float, 
                        y=None, 
                        window_type: str="centered") -> pd.DataFrame:
    
   
    assert window_type in ["centered", "trailing"]
    if y is None:
        windowed_data = get_windows_without_labels(x, interval, window_type)
    else:
        windowed_data = get_windows_with_labels(x, y, interval, window_type)
    return windowed_data

def get_windows_without_labels(
                                x: pd.DataFrame, 
                                interval: float, 
                                window_type: str="centered") -> np.array:

    centers = x["time"]
    windowed_data = trunc_centered(x, centers, interval)

    return windowed_data

def get_windows_with_labels(
                            x: pd.DataFrame, 
                            y: pd.DataFrame, 
                            interval: float, 
                            window_type: str="centered") -> pd.DataFrame:

    
    centers = y["time"]
    windowed_signal = trunc_centered(x, centers, interval)
    
    return windowed_signal

def trunc_centered(x: pd.DataFrame, centers: pd.Series, interval: float) -> pd.DataFrame:
    windowed_signals = []
    for i, center in enumerate(centers):
        center = float(center)

        
        window_start = center - interval / 2
        window_end = center + interval / 2

        windowed_signal = retrieve_window(x, window_start, window_end, interval)
        windowed_signal["timestamp"] = [i for _ in range(len(windowed_signal))]
        windowed_signals.append(windowed_signal)
    return pd.concat(windowed_signals, axis=0).reset_index(drop=True)


def retrieve_window(x: pd.DataFrame, 
                    window_start: float, 
                    window_end: float, 
                    interval: int) -> pd.DataFrame:

    
    expected_num_samples = interval * SAMPLING_RATE_X
    windowed_signal = x[(x["time"] > window_start) & (x["time"] < window_end)][X_HEADER]

    if window_start < x.loc[0, "time"]:
        # Padding at the start of the signal
        num_required_samples = expected_num_samples - len(windowed_signal)
        repeated_sample = x.head(1)[X_HEADER]
        padding_df = pd.DataFrame(np.repeat(repeated_sample.values, num_required_samples, axis=0))
        padding_df.columns = X_HEADER
        # Pad at the begining
        windowed_signal = pd.concat([padding_df, windowed_signal], axis=0)
        windowed_signal = windowed_signal.reset_index(drop=True)

    if window_end > x.loc[len(x)-1, "time"]:
        # Padding at the end of the signal
        num_required_samples = expected_num_samples - len(windowed_signal)
        repeated_sample = x.tail(1)[X_HEADER]
        padding_df = pd.DataFrame(np.repeat(repeated_sample.values, num_required_samples, axis=0))
        padding_df.columns = X_HEADER
        # Pad at the end
        windowed_signal = pd.concat([windowed_signal, padding_df], axis=0)
        windowed_signal = windowed_signal.reset_index(drop=True)

    assert expected_num_samples == len(windowed_signal)
    return windowed_signal


def downsample(x, y, test, win_len=0.1):
    agg_x = pd.DataFrame()
    for center in y["time"]:
        window_start = center - win_len / 2
        window_end = center + win_len / 2
        windowed_signal = x[(x["time"] > window_start) & (x["time"] < window_end)][
            ["accr_x", "accr_y", "accr_z", "gyr_x", "gyr_y", "gyr_z"]
        ]
        aggregate_data = windowed_signal.mean().to_frame().T
        aggregate_data["time"] = center
        agg_x = pd.concat([agg_x, aggregate_data], axis=0)
    
    if test:
        return agg_x.reset_index(drop=True)
    
    return pd.concat([agg_x.reset_index(drop=True), y["label"]], axis=1)


def preprocess(interval, test):
    save_path = data_root + f"filtered_window_{interval}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files_x_data = sorted(glob.glob(data_root + X_DATA.format("*", "*")))
    files_x_time = sorted(glob.glob(data_root + X_TIME.format("*", "*")))
    
    if not test:
        files_y_data = sorted(glob.glob(data_root + Y_DATA.format("*", "*")))

    files_y_time = sorted(glob.glob(data_root + Y_TIME.format("*", "*")))

    merged_data = pd.DataFrame()

    for i in tqdm(range(len(files_x_data)), total=len(files_x_data)):
        if not test:
            y_data = pd.read_csv(files_y_data[i], names=["label"])
        else:
            y_data = pd.DataFrame()
        y_time = pd.read_csv(files_y_time[i], names=["time"])

        x_data = pd.read_csv(
            files_x_data[i],
            names=X_HEADER,
        )
        median_data=median_filter(x_data.values, l_filter)
        lpf_data=freq_filter(x_data.values, l_filter, cutoff/fs)
        comb_data=freq_filter(median_data, l_filter, cutoff/fs)
        filtered_data = pd.DataFrame(comb_data, columns=X_HEADER)
        filtered_data['session'] = files_x_data[i].split('/')[-1]
        # merged_data = pd.concat([merged_data, filtered_data], axis=0)
        x_time = pd.read_csv(files_x_time[i], names=["time"])

        x_data = pd.concat([filtered_data, x_time], axis=1)
        y_data = pd.concat([y_data, y_time], axis=1)
        y_data = y_data.reset_index(drop=False).rename(columns={"index": "timestamp"})

        windowed_data = get_windowed_data(x_data, interval, y_data)
        windowed_data = windowed_data.merge(y_data, on = 'timestamp', how = 'inner')
        if not test:
            y_path = os.path.join(save_path, files_y_data[i].split('/')[-1])

            y_data[['timestamp', 'label']].to_csv(y_path, index = False)

            windowed_data = windowed_data.drop('label', axis = 1)
        
        x_path = os.path.join(save_path,files_x_data[i].split('/')[-1])
        windowed_data.to_csv(x_path, index = False)
    # merged_data.to_csv('test_filt_merg.csv', index=False)


preprocess(interval=1, test=True)