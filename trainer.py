import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from math import ceil
from keras import backend as K
from keras.layers import LeakyReLU
import datetime
import os
from os.path import expanduser
import glob
import json

from create_windows import get_all_data_and_labels
from data_utils import scale_windows

gpu = K.tensorflow_backend._get_available_gpus()

def load_windows(path_to_exps, path_to_validation, window_duration, window_every, norm_lower):
    # exps = sorted(glob.glob(path_to_exps + '*.npz'))

    # cv_idx = np.random.choice(range(len(exps)), cv_num, replace=False)
    # print('keeping', cv_idx)

    X_train, y_train = [], []
    X_test, y_test = [], []

    # for i, f in enumerate(exps):
    #     exp = np.load(f)
    #     windows = exp['windows']
    #     labels = exp['labels']

    #     # if i in cv_idx:
    #     #     cv_file.write(f + '\n')

    #     #     X_test.append(windows)
    #     #     y_test.append(labels)

    #     # else:
    #     X_train.append(windows)
    #     y_train.append(labels)

    # X_train = np.vstack(X_train)
    # y_train = np.concatenate(y_train)

    X_train, y_train = get_all_data_and_labels(path_to_exps, window_duration, window_every)
    X_test, y_test = get_all_data_and_labels(path_to_validation, window_duration, window_every)

    # val_exp = np.load(path_to_validation)
    # X_test = val_exp['windows']
    # y_test = val_exp['labels']

    X_train = scale_windows(X_train, norm_lower=norm_lower)
    X_test = scale_windows(X_test, norm_lower=norm_lower)

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    X_train = np.expand_dims(X_train, axis=-1)

    X_test, y_test = SMOTE().fit_resample(X_test, y_test)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, y_train, X_test, y_test


def create_model(window_duration, window_every, conv_layers, conv_channels, conv_kernels, conv_stride, conv_activation, conv_dropout, rnn_layers, rnn_units, rnn_dropout, optimizer):
    window_duration = int(window_duration)
    window_every = int(window_every)

    conv_layers = int(conv_layers)
    conv_channels = int(conv_channels)
    conv_kernels = int(conv_kernels)
    conv_stride = int(conv_stride)
    conv_activation = conv_activation
    conv_dropout = float(conv_dropout)

    rnn_layers = int(rnn_layers)
    rnn_units = int(rnn_units)
    rnn_dropout = float(rnn_dropout)

    optimizer = optimizer

    sampling_rate = 5

    # data, labels = get_all_data_and_labels(60, 10, './data_with_labels/')

    model = keras.Sequential()

    input_shape = (window_duration * sampling_rate, 1)

    for i in range(conv_layers):
        if conv_activation is 'leaky_relu':
            if i == 0:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid', input_shape=input_shape))
            else:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid'))
            model.add(LeakyReLU())

        else:
            if i == 0:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid', input_shape=input_shape, activation=conv_activation))
            else:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid', activation=conv_activation))

        model.add(layers.Dropout(conv_dropout))

    if gpu:
        LSTM_fun = layers.GRU

    else:
        LSTM_fun = layers.GRU

    for i in range(rnn_layers):
        if i < rnn_layers - 1:
            if conv_layers == 0:
                model.add(LSTM_fun(rnn_units, input_shape=input_shape, recurrent_dropout=rnn_dropout, return_sequences=True))
            else:
                model.add(LSTM_fun(rnn_units, recurrent_dropout=rnn_dropout, return_sequences=True))
        else:
            if conv_layers == 0:
                model.add(LSTM_fun(rnn_units, recurrent_dropout=rnn_dropout, input_shape=input_shape))
            else:
                model.add(LSTM_fun(rnn_units, recurrent_dropout=rnn_dropout))

    if conv_layers == 0 and rnn_layers == 0:
        model.add(layers.Dense(conv_channels, input_shape=input_shape, activation='tanh'))
        model.add(layers.Dropout(conv_dropout))
        model.add(layers.Flatten())

    elif rnn_layers == 0:
        model.add(layers.Flatten())

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_with_params(window_duration, window_every, conv_layers, conv_channels, conv_kernels, conv_stride, conv_activation, conv_dropout, rnn_layers, rnn_units, rnn_dropout, batch_size, norm_lower, optimizer, callbacks): 

    # variables
    window_duration = int(window_duration)
    window_every = int(window_every)

    conv_layers = int(conv_layers)
    conv_channels = int(conv_channels)
    conv_kernels = int(conv_kernels)
    conv_stride = int(conv_stride)
    conv_dropout = float(conv_dropout)

    rnn_layers = int(rnn_layers)
    rnn_units = int(rnn_units)
    rnn_dropout = float(rnn_dropout)

    batch_size = int(batch_size)

    norm_lower = int(norm_lower)

    sampling_rate = 5

    # data, labels = get_all_data_and_labels(60, 10, './data_with_labels/')

    X_train, y_train, X_test, y_test = load_windows('./data/labeled_data/training/', './data/labeled_data/validation/', window_duration, window_every, norm_lower)

    model = keras.Sequential()

    input_shape = (window_duration * sampling_rate, 1)

    for i in range(conv_layers):
        if conv_activation is 'leaky_relu':
            if i == 0:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid', input_shape=input_shape))
            else:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid'))
            model.add(LeakyReLU())

        else:
            if i == 0:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid', input_shape=input_shape, activation=conv_activation))
            else:
                model.add(layers.Conv1D(filters=conv_channels, kernel_size=conv_kernels, strides=conv_stride, padding='valid', activation=conv_activation))

        model.add(layers.Dropout(conv_dropout))

    if gpu:
        LSTM_fun = layers.GRU

    else:
        LSTM_fun = layers.GRU

    for i in range(rnn_layers):
        if i < rnn_layers - 1:
            if conv_layers == 0:
                model.add(LSTM_fun(rnn_units, input_shape=input_shape, recurrent_dropout=rnn_dropout, return_sequences=True))
            else:
                model.add(LSTM_fun(rnn_units, recurrent_dropout=rnn_dropout, return_sequences=True))
        else:
            if conv_layers == 0:
                model.add(LSTM_fun(rnn_units, recurrent_dropout=rnn_dropout, input_shape=input_shape))
            else:
                model.add(LSTM_fun(rnn_units, recurrent_dropout=rnn_dropout))

    if conv_layers == 0 and rnn_layers == 0:
        model.add(layers.Dense(conv_channels, input_shape=input_shape, activation='tanh'))
        model.add(layers.Dropout(conv_dropout))
        model.add(layers.Flatten())

    elif rnn_layers == 0:
        model.add(layers.Flatten())

    # model.add(
    #     layers.Conv1D(
    #         filters=64, kernel_size=3, padding='valid', activation='relu'))

    # model.add(
    #     layers.Conv1D(
    #         filters=64, kernel_size=3, padding='valid', activation='relu'))

    # model.add(layers.LSTM(32, input_shape=input_shape, return_sequences=True))

    # model.add(layers.CuDNNLSTM(128))
    # model.add(layers.LSTM(128))

    # model.add(layers.Flatten())

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

    # X_test, y_test = get_all_data_and_labels(60, 10, 'validation_data_with_labels')

    # print(X_train.shape)

    batch_size = batch_size
    epochs = 1000

    # train_gen = batch_generator(X_train, y_train, batch_size=batch_size)
    # valid_gen = batch_generator(X_test, y_test, batch_size=batch_size)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)

    # model.fit_generator( generator=train_gen, epochs=epochs, steps_per_epoch=ceil(X_train.shape[0] / batch_size), validation_data=valid_gen, validation_steps=ceil(X_test.shape[0] / batch_size), max_queue_size=1000)

    # score = model.evaluate(X_test, y_test)

    # print(score)

    # y_pred = model.predict_classes(X_test)

    # positives = y_pred == 1

    # print(positives.shape)

    # print(X_test.shape)

    # # for trace in X_test[positives[:, 0], :, 0]:
    # #     plt.plot(trace)
    # #     plt.show()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []


    def on_epoch_end(self, batch, logs={}):
        loss = logs.get('val_loss')
        """@nni.report_intermediate_result(loss)"""
        self.losses.append(loss)


    def on_train_end(self, logs={}):
        final_loss = min(self.losses)
        """@nni.report_final_result(final_loss)"""


def prepare_model_dir_and_train(model_params):
    home = expanduser("~")
    code_dir = 'Deep Learning'
    if not gpu:
        code_dir = os.path.join('Desktop/4th Year Project/Code', code_dir)
    model_name = str(datetime.datetime.now()).split('.')[0]
    model_dir = os.path.join(home, code_dir, 'models', model_name)

    json_fp = os.path.join(model_dir, 'params.json')

    with open(json_fp, 'w') as fp:
        json.dump(model_params, fp)

    os.makedirs(model_dir, exist_ok=True)

    checkpoint_format = '{epoch:02d}-{val_loss:.4f}-{val_acc:.2f}.hdf5'
    checkpoint_fn = os.path.join(model_dir, checkpoint_format)

    log_fn = os.path.join(home, code_dir, 'logs', model_name)

    callbacks = [LossHistory()]
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True))
    callbacks.append(keras.callbacks.EarlyStopping(patience=40))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=0.0001, verbose=1))
    callbacks.append(keras.callbacks.TensorBoard(log_dir=log_fn, batch_size=batch_size))

    model = train_with_params(model_params, callbacks)

if __name__ == "__main__":
    home = expanduser("~")
    code_dir = 'Deep Learning'
    if not gpu:
        code_dir = os.path.join('Desktop/4th Year Project/Code', code_dir)
    model_name = str(datetime.datetime.now()).split('.')[0]
    model_dir = os.path.join(home, code_dir, 'models', model_name)

    os.makedirs(model_dir, exist_ok=True)

    json_fp = os.path.join(model_dir, 'params.json')

    checkpoint_format = '{epoch:02d}-{val_loss:.4f}-{val_acc:.2f}.hdf5'
    checkpoint_fn = os.path.join(model_dir, checkpoint_format)

    log_fn = os.path.join(home, code_dir, 'logs', model_name)

    # variables
    """@nni.variable(nni.choice(10, 15, 20, 25, 30, 45, 60, 120), name=window_duration)"""
    window_duration = 60 # seconds
    """@nni.variable(nni.choice(5, 10, 20, 30), name=window_every)"""
    window_every = 10 # seconds

    """@nni.variable(nni.choice(0, 1, 2, 3, 4), name=conv_layers)"""
    conv_layers = 1
    """@nni.variable(nni.choice(4, 8, 16, 32, 64), name=conv_channels)"""
    conv_channels = 64
    """@nni.variable(nni.choice(3, 5, 7), name=conv_kernels)"""
    conv_kernels = 3
    """@nni.variable(nni.choice(1, 2, 3), name=conv_stride)"""
    conv_stride = 2
    """@nni.variable(nni.choice('relu', 'tanh', 'leaky_relu'), name=conv_activation)"""
    conv_activation = 'relu'
    """@nni.variable(nni.choice(0.1, 0.2, 0.3, 0.4, 0.5), name=conv_dropout)"""
    conv_dropout = 0.2

    """@nni.variable(nni.choice(0), name=rnn_layers)"""
    rnn_layers = 0
    """@nni.variable(nni.choice(2, 4, 8, 16, 32, 64, 128), name=rnn_units)"""
    rnn_units = 128
    """@nni.variable(nni.choice(0.1, 0.2, 0.3, 0.4, 0.5), name=rnn_dropout)"""
    rnn_dropout = 0.2

    """@nni.variable(nni.choice(16, 32, 64, 128), name=batch_size)"""
    batch_size = 128
    """@nni.variable(nni.choice(-1, 0), name=norm_lower)"""
    norm_lower = -1

    """@nni.variable(nni.choice('adam', 'sgd'), name=optimizer)"""
    optimizer = 'adam'

    model_params = dict(window_duration=window_duration, window_every=window_every, conv_layers=conv_layers, conv_channels=conv_channels, conv_kernels=conv_kernels, conv_stride=conv_stride, conv_activation=conv_activation, conv_dropout=conv_dropout, rnn_layers=rnn_layers, rnn_units=rnn_units, rnn_dropout=rnn_dropout, optimizer=optimizer, batch_size=batch_size, norm_lower=norm_lower)

    with open(json_fp, 'w') as fp:
        json.dump(model_params, fp)

    callbacks = [LossHistory()]
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True))
    callbacks.append(keras.callbacks.EarlyStopping(min_delta=0.0002, patience=40))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=0.0001, verbose=1))
    callbacks.append(keras.callbacks.TensorBoard(log_dir=log_fn, batch_size=batch_size))

    model = train_with_params(window_duration, window_every, conv_layers, conv_channels, conv_kernels, conv_stride, conv_activation, conv_dropout, rnn_layers, rnn_units, rnn_dropout, batch_size, norm_lower, optimizer, callbacks)
