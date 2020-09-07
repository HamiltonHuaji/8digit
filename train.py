#!/usr/bin/env python3

import tensorflow as tf
from PIL import Image
import numpy as np

learning_rate = 0.01

from answer import answer


def read(filename):
    i = Image.open(filename)
    a = (np.array(i) / 2 + np.random.rand(64, 32, 3) * 32 + 32).astype(np.uint8)
    return a

# read(list(answer.keys())[0])


train = [
    (read(_), answer[_])
    for _ in answer.keys()
]

x_train, y_ = zip(*train)

x_train = np.array(x_train, dtype=np.uint8)
# y_train = np.zeros([len(y_), 10], dtype=np.uint8)
# for i, _ in enumerate(y_):
#     y_train[i][_] = 1
y_train = np.array(y_, dtype=np.uint8)

print(x_train.shape)
print(y_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 32, 3)),
    tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(64, 32, 3), padding="same"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(64, 32, 3), padding="same"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(input_shape=(64, 32, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
print(model.output_shape)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
print(test_acc)

probability_model = tf.keras.Sequential(model)

x_test = np.expand_dims(x_train[0], 0)

prediction = probability_model.predict(x_test)

print(prediction)
