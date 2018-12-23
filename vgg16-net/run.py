from __future__ import absolute_import, division, print_function
import os 
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

import tensorflow as tf
import keras.backend as K
import numpy as np
tf.enable_eager_execution()
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from model import VGG


batch_size = 128
nb_epoch = 1
img_rows, img_cols = 32, 32
epochs = 1

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

testY = tf.one_hot(testY, depth=10).numpy()
trainY = tf.one_hot(trainY, depth=10).numpy()

testY = testY.astype(np.int64)
testX = testX.astype(np.int64)


model = VGG('VGG16', 10)

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

dummy_x = tf.zeros((1, 32, 32, 3))
model._set_inputs(dummy_x)
print(model(dummy_x).shape)

model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs,
          validation_data=(testX, testY), verbose=1)

evaluate on test set
scores = model.evaluate(testX, testY, batch_size, verbose=1)
print("Final test loss and accuracy :", scores)
