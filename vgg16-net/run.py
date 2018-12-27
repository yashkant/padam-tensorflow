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

if dataset == 'cifar10':
    MEAN = [0.4914, 0.4822, 0.4465]
    STD_DEV = [0.2023, 0.1994, 0.2010]
elif dataset == 'cifar100':
    MEAN = [0.507, 0.487, 0.441]
    STD_DEV = [0.267, 0.256, 0.276]

def normalize(t):
    t = tf.div(tf.subtract(t, MEAN), STD_DEV) 
    return t

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

optim_params = {
    'padam': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'b1': 0.9,
        'b2': 0.999
    },
    'adam': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99
    },
    'adamw': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99
    },
    'amsgrad': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99
    },
    'sgd': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'm': 0.9
    }
}

hp = hyperparameters[dataset]
op = optim_params[optimizer]
if optimizer == 'adamw' and dataset=='imagenet':
    op['weight_decay'] = 0.05 

if dataset == 'cifar10':
    from keras.datasets import cifar10
    (trainX, trainY), (testX, testY) = cifar10.load_data()
elif dataset == 'cifar100':
    from keras.datasets import cifar100
    (trainX, trainY), (testX, testY) = cifar100.load_data()

batch_size = hp['batch_size']
nb_epoch = 1
img_rows, img_cols = 32, 32
epochs = hp['epoch']
train_size = trainX.shape[0]
print(train_size)

trainX = trainX.astype('float32')
# trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
trainX = trainX/255
testX = testX.astype('float32')
# testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
testX = testX/255

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

# testY = tf.one_hot(testY, depth=10).numpy()
# trainY = tf.one_hot(trainY, depth=10).numpy()

tf.train.create_global_step()

base_learning_rate = 0.1

learning_rate = tf.train.exponential_decay(0.1, tf.train.get_global_step() * batch_size,
                                       hp['decay_after']*train_size, 0.1, staircase=True)

model = VGG('VGG16', 10)

if optimizer == 'padam':
    optim = Padam(learning_rate=op['lr'], p=op['p'], beta1=op['b1'], beta2=op['b2'])
elif optimizer == 'adamw':
    optimizer = tf.train.AdamOptimizer(learning_rate=op['lr'], beta1=op['b1'], beta2=op['b2'])
elif optimizer == 'adamw':
    adamw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
    optim = adamw(weight_decay=op['weight_decay'], learning_rate=op['lr'],  beta1=op['b1'], beta2=op['b2'])
elif optimizer == 'amsgrad':
    optim = AMSGrad(learning_rate=op['lr'], beta1=op['b1'], beta2=op['b2'])
elif optimizer == 'sgd':
    optim = tf.train.MomentumOptimizer(learning_rate=op['lr'], momentum=op['m'])

dummy_x = tf.zeros((1, 32, 32, 3))
model._set_inputs(dummy_x)
y = model(dummy_x)
print(model(dummy_x).shape)

# if optimizer is not 'adamw':
#     var   = model.trainable_weights

#     lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var
#                         if 'bias' not in v.name ]) * op['weight_decay']
#     loss = (tf.reduce_mean(tf.keras.backend.categorical_crossentropy(target=,output=)) + lossL2)
# else:
loss = 'categorical_crossentropy'

model.compile(optimizer=optim, loss=loss,
                  metrics=['accuracy'], global_step=tf.train.get_global_step())

from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(zoom_range=0.125,
                            preprocessing_function=normalize,
                            horizontal_flip=True,
                            )
datagen_test = ImageDataGenerator(
                            preprocessing_function=normalize,
                            )

model.fit_generator(datagen_train.flow(trainX, trainY, batch_size = batch_size), epochs = epochs, 
                                 validation_data = (testX, testY), verbose=1)

scores = model.evaluate(datagen_test.flow(testX, testY, batch_size = batch_size), verbose=1)
print("Final test loss and accuracy :", scores)
