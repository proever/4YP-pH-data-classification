import argparse
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math

import data_utils

parser = argparse.ArgumentParser(description='display data and labels')
parser.add_argument('path_to_experiment', type=str, help='path to experiment raw data')
parser.add_argument('path_to_start_stop_times', type=str, help='path to experiment start/stop times')
parser.add_argument('output_folder', type=str, help='name of output folder')
parser.add_argument('--noldr', dest='ldr', action='store_false')
parser.set_defaults(ldr=True)

args = parser.parse_args()

path_to_experiment = args.path_to_experiment
path_to_start_stop_times = args.path_to_start_stop_times
output_folder = args.output_folder
ldr = args.ldr

exp_id = os.path.split(path_to_experiment)[1].strip('.tsv.xlsx')

print('reading', exp_id, 'from', path_to_experiment)
exp_data = pd.read_excel(io=path_to_experiment, sheet_name='Sheet1')

start_stop_times = pd.read_excel(io=path_to_start_stop_times, sheet_name='Sheet1')

start_stop_times.set_index('exp_id', inplace=True)

exp_times = start_stop_times.loc[exp_id, :]

dose_starts = exp_times['CCKSSTART'][1:-1]
dose_stops = exp_times['CCKSSTOP'][1:-1]

dose_starts = dose_starts + ' ' + exp_times['CCK_HIGHdoseSTART'].replace('60', '01', 1)
dose_stops = dose_stops + ' ' + exp_times['CCK_HIGHdoseSTOP'].replace('60', '01', 1)

dose_starts = dose_starts.split(' ')
dose_stops = dose_stops.split(' ')

response_duration=120
offset=15

sampling_rate = 5

exp_duration = max(exp_data['Time / s'])

output = pd.DataFrame()

if ldr:
    time_code = 't_out'
    time = exp_data[time_code]
    max_index = data_utils.find_nearest(time, exp_duration)
    # output[time_code] = time[1:max_index+1]
    data_code = 'LDR'

else:
    time_code = 'Time / s'
    time = exp_data[time_code]
    max_index = data_utils.find_nearest(time, exp_duration)
    data_code = '[chan  0]'

output[time_code] = time[:max_index]

data = exp_data[data_code]

exp_duration = max(exp_data['Time / s'])

max_index = data_utils.find_nearest(time, exp_duration)

zero_time = datetime.datetime.strptime('00:00:00', '%H:%M:%S')

# data_scaled = (data[:max_index] - min(data[:max_index])) / (max(data[:max_index]) - min(data[:max_index]))
# exp_data['LDR_scaled'] = data_scaled

labels = np.zeros(int(max_index))

for i, start in enumerate(dose_starts):
    if i != 0:
        stop = dose_stops[i]

        start = datetime.datetime.strptime(start, '%H:%M:%S') - zero_time
        stop = datetime.datetime.strptime(stop, '%H:%M:%S') - zero_time

        start_index = data_utils.find_nearest(exp_data[time_code], stop.total_seconds())

        # start_index = np.where(
        #     exp_data[time_code] == start.total_seconds())[0][0]

        if response_duration != -1:
            stop_index = start_index + (response_duration * sampling_rate)

        else:
            stop_index = data_utils.find_nearest(exp_data[time_code], stop.total_seconds())
            # stop_index = np.where(
            #     exp_data[time_code] == stop.total_seconds())[0][0]

        off = offset * sampling_rate

        labels[start_index + off:stop_index + off] = 1

output['data'] = data
output['labels'] = labels.tolist()

no_outliers_folder = os.path.join(output_folder, 'no_outliers')
os.makedirs(no_outliers_folder, exist_ok=True)

output_path = os.path.join(output_folder, exp_id + '.xlsx')
no_outliers_output_path = os.path.join(no_outliers_folder, exp_id + '.xlsx')

no_outliers_output = pd.DataFrame()
no_outliers_data = (data[:max_index] - min(data[:max_index])) / (max(data[:max_index]) - min(data[:max_index]))

outlier_mask = data_utils.reject_outliers(no_outliers_data.values)

no_outliers_output[time_code] = output[time_code][outlier_mask].values
no_outliers_output['data'] = no_outliers_data[outlier_mask].values
no_outliers_output['labels'] = output['labels'][outlier_mask].values

print(output_path)

output.to_excel(output_path)
no_outliers_output.to_excel(no_outliers_output_path)
