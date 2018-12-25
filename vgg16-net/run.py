from __future__ import absolute_import, division, print_function
import os 
import sys
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"

import tensorflow as tf
import keras.backend as K
import numpy as np
tf.enable_eager_execution()

import keras.callbacks as callbacks
import keras.utils.np_utils as kutils

from model import VGG
from padam import Padam
from amsgrad import AMSGrad

dataset = 'cifar10'
optimizer = 'padam'

hyperparameters = {
    'cifar10': {
        'epoch': 200,
        'batch_size': 128,
        'decay_after': 50
    },
    'cifar100': {
        'epoch': 200,
        'batch_size': 128,
        'decay_after': 50  
    },
    'imagenet': {
        'epoch': 100,
        'batch_size': 256,
        'decay_after': 30
    }
}

if dataset == 'cifar10':
    from keras.datasets import cifar10
    (trainX, trainY), (testX, testY) = cifar10.load_data()
elif dataset == 'cifar100':
    from keras.datasets import cifar100
    (trainX, trainY), (testX, testY) = cifar100.load_data()

batch_size = hyperparameters[dataset]['batch_size']
nb_epoch = 1
img_rows, img_cols = 32, 32
epochs = hyperparameters[dataset]['epoch']
train_size = trainX.shape[0]
print(train_size)

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

testY = tf.one_hot(testY, depth=10).numpy()
trainY = tf.one_hot(trainY, depth=10).numpy()

tf.train.create_global_step()

base_learning_rate = 0.1

learning_rate = tf.train.exponential_decay(0.1, tf.train.get_global_step() * batch_size,
                                       hyperparameters[dataset]['decay_after']*train_size, 0.1, staircase=True)

model = VGG('VGG16', 10)

if optimizer == 'padam':
    padamw = tf.contrib.opt.extend_with_decoupled_weight_decay(Padam)
    opw = padamw(weight_decay=0.0005, learning_rate=0.1, p=0.125, beta1=0.9, beta2=0.999)
elif optimizer == 'adam':
    adamw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
    opw = adamw(weight_decay=0.0001, learning_rate=0.001, beta1=0.9, beta2=0.99)
elif optimizer == 'amsgrad':
    amsgradw = tf.contrib.opt.extend_with_decoupled_weight_decay(AMSGrad)
    opw = amsgradw(weight_decay=0.0001, learning_rate=0.001, beta1=0.9, beta2=0.99)
elif optimizer == 'sgd':
    sgdw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.MomentumOptimizer)
    opw = sgdw(weight_decay=0.0005, learning_rate=0.1, momentum=0.9)


model.compile(optimizer=opw, loss='categorical_crossentropy',
                  metrics=['accuracy'], global_step=tf.train.get_global_step())

dummy_x = tf.zeros((1, 32, 32, 3))
model._set_inputs(dummy_x)
print(model(dummy_x).shape)

model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs,
          validation_data=(testX, testY), verbose=1)

scores = model.evaluate(testX, testY, batch_size, verbose=1)
print("Final test loss and accuracy :", scores)
