# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:58:00 2019

@author: Panos
"""

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

import matplotlib.pyplot as plt
from keras.preprocessing import image

print(tf.__version__)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

NUM_CLASSES = 10
WIDTH = 28
HEIGHT = 28

x_train , x_test = x_train / 255.0 , x_test / 255.0

x_train  = x_train.reshape(x_train.shape[0], WIDTH, HEIGHT, 1)
x_test  = x_test.reshape(x_test.shape[0], WIDTH, HEIGHT, 1)

print(x_train.shape)
print(x_train.dtype)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, kernel_size=(3,3),
                               activation=tf.nn.relu, input_shape=(WIDTH, HEIGHT, 1)),
        tf.keras.layers.Conv2D(20, kernel_size=(3,3),
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)      
])
        
        
model.compile(optimizer='Adam', #tf.keras.optimizers.Adam,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])#tf.keras.metrics.categorical_accuracy)

model.fit(x_train, y_train,
          batch_size=512,
          epochs=2)

plt.gray()
plt.imshow(x_test[3].reshape(WIDTH,HEIGHT))
plt.show()

print(y_test[3])
print(model.predict(x_test)[3])
print(model.predict_classes(x_test)[3])

#print(model.evaluate(x_test, y_test))