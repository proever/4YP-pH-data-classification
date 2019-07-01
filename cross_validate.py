import argparse
import pandas as pd
import math
import numpy as np
import keras

from data_utils import scale_windows
 

def get_windowed_data(path_to_experiment, window_length, data_steps):

    print('reading', path_to_experiment)
    exp_data = pd.read_excel(io=path_to_experiment, sheet_name='Sheet1')

    print(exp_data.columns)

    if 'LDR' in exp_data:
        data_code = 'LDR'

    else:
        data_code = 'data'

    max_index = len(exp_data[data_code])

    num_windows = math.floor((max_index - window_length ) / data_steps)
    indexer = np.arange(window_length)[None, :] + data_steps*np.arange(num_windows)[:, None]

    print('num samples', max_index)
    print('num_windows', num_windows)

    windowed_data = exp_data[data_code].values[indexer]
    # windowed_data = (2 * (windowed_data - np.amin(windowed_data, axis=1)[:, np.newaxis]) / (np.amax(windowed_data, axis=1)[:, np.newaxis] - np.amin(windowed_data, axis=1)[:, np.newaxis])) - 1

    return windowed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create windows from unlabeled data')
    parser.add_argument('path_to_experiment', type=str, help='path to unlabeled experiment data')
    parser.add_argument('path_to_model', type=str, help='path to model to evaluate')
    parser.add_argument('--make_csv', dest='make_csv', action='store_true')
    parser.set_defaults(make_csv=False)

    args = parser.parse_args()
    path_to_experiment = args.path_to_experiment
    path_to_model = args.path_to_model
    make_csv = args.make_csv

    data = get_windowed_data(path_to_experiment, 50, 1)
    data = scale_windows(data, norm_lower=-1)

    if make_csv: np.savetxt('results/windowed_data.csv', data, delimiter=',')

    data = np.expand_dims(data, axis=-1)

    model = keras.models.load_model(path_to_model)
    out = model.predict(data)
    np.savetxt('results/test.txt', out)
