import numpy as np
import math


# adapted from https://stackoverflow.com/a/26026189, but returns idx not array[idx]
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


# from https://stackoverflow.com/a/16562028
def reject_outliers(data, m=5.189):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return s<m


# scales matrix of windows so that each window is between min_value and 1
def scale_windows(windowed_data, norm_lower=-1):
    difference = 1 - norm_lower

    max_per_window = np.amax(windowed_data, axis=1)[:, np.newaxis]
    min_per_window = np.amin(windowed_data, axis=1)[:, np.newaxis]

    min_max_per_window = max_per_window - min_per_window

    return (difference * (windowed_data - min_per_window) / min_max_per_window) + norm_lower
