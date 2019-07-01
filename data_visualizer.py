import argparse
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description='display data and labels')
parser.add_argument('path_to_experiment', type=str, help='path to experiment raw data')
parser.add_argument('path_to_start_stop_times', type=str, help='path to experiment start/stop times')

args = parser.parse_args()

path_to_experiment = args.path_to_experiment
path_to_start_stop_times = args.path_to_start_stop_times

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

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def reject_outliers(data, m = 5.189):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    # data[s>m] = mdev
    return s<m


time_code = 't_out'
time = exp_data[time_code]
data = exp_data['LDR']

exp_duration = max(exp_data['Time / s'])

max_index = find_nearest(time, exp_duration)

zero_time = datetime.datetime.strptime('00:00:00', '%H:%M:%S')
plt.plot(time[:max_index], exp_data['[chan  0]'][:max_index])
plt.plot(time[:max_index], data[:max_index])
plt.title("Experiment ID: " + exp_id)
plt.legend(['original data', 'LDR data'])
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.figure()

data_scaled = (data[:max_index] - min(data[:max_index])) / (max(data[:max_index]) - min(data[:max_index]))
exp_data['LDR_scaled'] = data_scaled

# ax = exp_data.plot(x=time_code, y='LDR')
plt.plot(time[:max_index], data_scaled[:max_index])

no_outliers_mask = reject_outliers(data_scaled.values)
plt.plot(time[:max_index][no_outliers_mask], data_scaled[:max_index][no_outliers_mask])

for i, start in enumerate(dose_starts):
    stop = dose_stops[i]

    start = datetime.datetime.strptime(start, '%H:%M:%S') - zero_time
    stop = datetime.datetime.strptime(stop, '%H:%M:%S') - zero_time

    start_index = find_nearest(exp_data[time_code], start.total_seconds())

    # print(start_index)

    # start_index = np.where(
    #     exp_data[time_code] == start.total_seconds())[0][0]

    if response_duration != -1:
        stop_index = start_index + (response_duration * sampling_rate)

    else:
        stop_index = find_nearest(exp_data[time_code], stop.total_seconds())
        # stop_index = np.where(
        #     exp_data[time_code] == stop.total_seconds())[0][0]

    time_snippet = exp_data.loc[(exp_data.index > start_index + offset*sampling_rate) & (exp_data.index < stop_index + offset*sampling_rate), time_code]
    data_snippet = exp_data.loc[(exp_data.index > start_index + offset*sampling_rate) & (exp_data.index < stop_index + offset*sampling_rate), 'LDR_scaled']

    plt.plot(time_snippet, data_snippet, color='C2')
    # exp_data.loc[(exp_data.index > start_index + offset*sampling_rate) & (exp_data.index < stop_index + offset*sampling_rate), 'LDR'].plot(x=time_code, y='LDR', color='r', ax=ax)

os = offset * sampling_rate

plt.title("Experiment ID: " + exp_id)
plt.legend(['LDR data', 'outliers removed', 'labelled data'])
plt.xlabel('Time (s)')
plt.ylabel('Voltage (scaled, unitless)')
plt.show()
