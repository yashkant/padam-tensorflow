from __future__ import absolute_import, division, print_function
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from wide_resnet import WRNModel
from keras import backend as K



batch_size = 100
nb_epoch = 100
img_rows, img_cols = 32, 32
epochs = 100

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


model = WRNModel()

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

dummy_x = tf.zeros((1, 32, 32, 3))
model._set_inputs(dummy_x)
print(model(dummy_x).shape)
# train
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs,
          validation_data=(testX, testY), verbose=1)

# evaluate on test set
scores = model.evaluate(testX, testY, batch_size, verbose=1)
print("Final test loss and accuracy :", scores)



