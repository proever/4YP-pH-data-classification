import pandas as pd
import numpy as np
import datetime


def get_data_and_labels():
    data_dir = './raw_data/'

    exp_log_fn = 'experiment_start_stop_times.xls'

    exp_info = pd.read_excel(data_dir + exp_log_fn)

    info_columns = list(exp_info)

    all_raw_data = []
    all_labels = []

    for i, exp_id in enumerate(exp_info[info_columns[0]]):
        dose_starts = exp_info[info_columns[1]][0]
        dose_stops = exp_info[info_columns[2]][0]

        dose_starts = dose_starts[1:-1]
        dose_stops = dose_stops[1:-1]

        dose_starts = dose_starts + ' ' + exp_info[info_columns[3]][0]
        dose_stops = dose_stops + ' ' + exp_info[info_columns[4]][0]

        dose_starts = dose_starts.split(' ')
        dose_stops = dose_stops.split(' ')

        exp_data_fn = data_dir + exp_id + '.txt'

        exp_data = pd.read_csv(exp_data_fn, delimiter='\t')

        columns = list(exp_data)

        sampling_rate = 5
        offset = 15

        zero_time = datetime.datetime.strptime('00:00:00', '%H:%M:%S')

        exp_data_points = len(exp_data[columns[0]])

        labels = np.zeros([1, exp_data_points])

        for i, start in enumerate(dose_starts):
            stop = dose_stops[i]

            if i == 3:
                start = start.replace('60', '01', 1)
                stop = stop.replace('60', '01', 1)

            start = datetime.datetime.strptime(start, '%H:%M:%S') - zero_time
            stop = datetime.datetime.strptime(stop, '%H:%M:%S') - zero_time

            start_index = np.where(
                exp_data[columns[0]] == start.total_seconds())[0][0]
            stop_index = np.where(
                exp_data[columns[0]] == stop.total_seconds())[0][0]

            os = offset*sampling_rate

            start_index += os
            stop_index += os

            labels[0][start_index:stop_index] = 1

        all_labels.append(labels[0][:])

        ch0 = exp_data[columns[1]]

        all_raw_data.append(ch0.values)

    all_raw_data_in_windows = np.zeros((393890, 150))
    all_labels_for_windows = np.zeros((393890, 1))

    window_duration = 30
    samples_per_window = window_duration * sampling_rate

    for exp_number, raw_data in enumerate(all_raw_data):
        num_rolling_windows = len(raw_data) - samples_per_window

        current_labels = all_labels[exp_number]

        for i in range(num_rolling_windows):
            rolling_window = raw_data[i:i+samples_per_window]
            rolling_window_label = int(
                1 in current_labels[i:i+samples_per_window])

            all_raw_data_in_windows[i] = rolling_window
            all_labels_for_windows[i][0] = rolling_window_label

    return (all_raw_data_in_windows, all_labels_for_windows)
