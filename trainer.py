import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from math import ceil
from keras import backend as K
import datetime
import os
import glob

from create_windows import get_all_data_and_labels


def load_windows(path_to_exps, model_dir, cv_num=3, normalize_range=(-1, 1)):
    exps = sorted(glob.glob(path_to_exps + '*.npz'))

    cv_idx = np.random.choice(range(len(exps)), cv_num, replace=False)
    print('keeping', cv_idx)

    X_train, y_train = [], []
    X_test, y_test = [], []

    cv_fn = os.path.join(model_dir, 'cross_validation.txt')

    with open(cv_fn, 'a') as cv_file:
        for i, f in enumerate(exps):
            exp = np.load(f)
            windows = exp['windows']
            labels = exp['labels']

            if i in cv_idx:
                cv_file.write(f + '\n')

                X_test.append(windows)
                y_test.append(labels)

            else:
                X_train.append(windows)
                y_train.append(labels)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)

    return X_train, y_train, X_test, y_test


def train_with_params(random_seed=42,
                      normalize=True,
                      linearize=True,
                      window_duration=60,
                      response_duration=120,
                      offset=15):

    print(K.tensorflow_backend._get_available_gpus())

    model_name = str(datetime.datetime.now()).split('.')[0]
    model_dir = os.path.join('models', model_name)

    os.makedirs(model_dir, exist_ok=True)

    sampling_rate = 5

    # data, labels = get_all_data_and_labels(60, 10, './data_with_labels/')

    X_train, y_train, X_test, y_test = load_windows('./data/windowed_data/LDR/', model_dir)

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    X_train = np.expand_dims(X_train, axis=-1)

    X_test, y_test = SMOTE().fit_resample(X_test, y_test)
    X_test = np.expand_dims(X_test, axis=-1)

    model = keras.Sequential()

    input_shape = (window_duration * sampling_rate, 1)

    model.add(layers.Conv1D(filters=64, kernel_size=3, padding='valid', input_shape=input_shape, activation='relu'))

    # model.add(
    #     layers.Conv1D(
    #         filters=64, kernel_size=3, padding='valid', activation='relu'))

    # model.add(
    #     layers.Conv1D(
    #         filters=64, kernel_size=3, padding='valid', activation='relu'))

    # model.add(layers.LSTM(32, input_shape=input_shape, return_sequences=True))

    # model.add(layers.CuDNNLSTM(128))
    model.add(layers.LSTM(128))

    # model.add(layers.Flatten())

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

    # X_test, y_test = get_all_data_and_labels(60, 10, 'validation_data_with_labels')

    # print(X_train.shape)

    batch_size = 128
    epochs = 1000

    # train_gen = batch_generator(X_train, y_train, batch_size=batch_size)
    # valid_gen = batch_generator(X_test, y_test, batch_size=batch_size)

    checkpoint_format = '{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint_fn = os.path.join(model_dir, checkpoint_format)

    log_fn = os.path.join('logs', model_name)

    callbacks = []
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True))
    callbacks.append(keras.callbacks.EarlyStopping(patience=50))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=0.0001, verbose=1))
    callbacks.append(keras.callbacks.TensorBoard(log_dir=log_fn, batch_size=batch_size))

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

    return model

if __name__ == "__main__":
    train_with_params()
