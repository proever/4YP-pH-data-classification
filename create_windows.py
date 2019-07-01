import argparse
import os
import glob
import pandas as pd
import math
import numpy as np

import data_utils


def get_data_and_labels(path_to_experiment, window_length, data_steps):

    # print('reading', path_to_experiment)
    # exp_data = pd.read_excel(io=path_to_experiment, sheet_name='Sheet1')
    exp_data = np.genfromtxt(path_to_experiment, delimiter=',', names=True)

    max_index = len(exp_data['data'])

    num_windows = math.floor((max_index - window_length ) / data_steps)

    # from https://stackoverflow.com/a/42258242
    indexer = np.arange(window_length)[None, :] + data_steps*np.arange(num_windows)[:, None]

    # print('num samples', max_index)
    # print('num_windows', num_windows)

    windowed_data = exp_data['data'][indexer]
    windowed_labels = exp_data['labels'][indexer]

    # windowed_data = (2 * (windowed_data - np.amin(windowed_data, axis=1)[:, np.newaxis]) / (np.amax(windowed_data, axis=1)[:, np.newaxis] - np.amin(windowed_data, axis=1)[:, np.newaxis])) - 1
    windowed_labels = np.amax(windowed_labels, axis=1)

    return (windowed_data, windowed_labels)


def get_all_data_and_labels(path_to_experiments, window_duration, window_every):
    
    sampling_rate = 5 # Hz

    window_length = window_duration * sampling_rate
    data_steps = window_every * sampling_rate

    windows = []
    labels = []

    for f in glob.glob(path_to_experiments + '*.csv'):
        windowed_data, windowed_labels = get_data_and_labels(f, window_length, data_steps)
        
        windows.append(windowed_data)
        labels.append(windowed_labels)

    windows = np.vstack(windows)
    labels = np.concatenate(labels)

    return(windows, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create windows from labeled data')
    parser.add_argument('path_to_experiment_and_labels', type=str, help='path to labeled experiment data')
    parser.add_argument('output_folder', type=str, help='name of output folder')

    args = parser.parse_args()
    path_to_experiment_and_labels = args.path_to_experiment_and_labels
    output_folder = args.output_folder

    exp_fn = os.path.split(path_to_experiment_and_labels)[1]

    exp_name = exp_fn.strip('.xlsx')

    window_size = 60 # seconds
    step_size = 5  # seconds

    sampling_rate = 5 # Hz

    window_length = int(window_size * sampling_rate)
    data_steps = int(step_size * sampling_rate)

    windowed_data, windowed_labels = get_data_and_labels(path_to_experiment_and_labels, window_length, data_steps)

    output_path = os.path.join(output_folder, exp_name)

    np.savez_compressed(output_path, windows=windowed_data, labels=windowed_labels)

