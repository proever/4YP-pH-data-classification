import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import data_labeling

data, labels = data_labeling.get_data_and_labels()

data = np.expand_dims(data, axis=-1)

model = tf.keras.Sequential()

model.add(layers.Conv1D(
    filters=16,
    kernel_size=3,
    padding='valid',
    input_shape=(150, 1),
    activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.20, random_state=42)

model.fit(X_train, y_train, epochs=10)

score = model.evaluate(X_test, y_test)

print(score)
