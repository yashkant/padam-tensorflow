from __future__ import absolute_import, division, print_function
import os 
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

import tensorflow as tf
import keras.backend as K
import numpy as np
tf.enable_eager_execution()

import keras.callbacks as callbacks
import keras.utils.np_utils as kutils


from keras.callbacks import CSVLogger


import matplotlib.pyplot as plt

from amsgrad import AMSGrad
from eager_resnet import Resnet
from padam import Padam

dataset = 'cifar10'
optimizer = 'adamw'

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
        'b2': 0.999, 
        'color': 'darkred',
        'linestyle':'-',
    },
    'adam': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'orange',
        'linestyle':'--',
    },
    'adamw': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'magenta',
        'linestyle':'--',
    },
    'amsgrad': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color' : 'darkgreen',
        'linestyle':'-.',
    },
    'sgd': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'm': 0.9,
        'color': 'blue',
        'linestyle':'-',
    }
}



if dataset == 'cifar10':
    from keras.datasets import cifar10
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    """print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)"""
    #(trainX, trainY), (testX, testY) = (trainX[:100], trainY[:100]), (testX[:100], testY[:100] )
    #print(type(trainY[0][0]))
    """print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)"""
elif dataset == 'cifar100':
    from keras.datasets import cifar100
    (trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

#testY = testY.astype(np.int64)
#trainY = trainY.astype(np.int64)
#testY = tf.one_hot(testY, depth=10).numpy()
#trainY = tf.one_hot(trainY, depth=10).numpy()

dataset = 'cifar10'
hp = hyperparameters[dataset]
batch_size = hp['batch_size']
epochs = hp['epoch']

# resnet cifar10 training and plots


history_resnet = {}
for optimizer in ['padam', 'adam', 'adamw', 'amsgrad', 'sgd']:
    op = optim_params[optimizer]
    if optimizer is not 'adamw':
        model = Resnet(training= True, data_format= K.image_data_format(), classes = 10, wt_decay = op['weight_decay'])
    else:
        model = Resnet(training= True, data_format= K.image_data_format(), classes = 10, wt_decay = 0)
    
    if optimizer == 'padam':
        optim = Padam(learning_rate=op['lr'], p=op['p'], beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate=op['lr'], beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'adamw':
        adamw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
        optim = adamw(weight_decay=op['weight_decay'], learning_rate=op['lr'],  beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'amsgrad':
        optim = AMSGrad(learning_rate=op['lr'], beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'sgd':
        optim = tf.train.MomentumOptimizer(learning_rate=op['lr'], momentum=op['m'])
    
    model.compile(optimizer=optim, loss='categorical_crossentropy',
                      metrics=['accuracy'], global_step=tf.train.get_global_step())
    
    #dummy_x = tf.zeros((1, 32, 32, 3))
    #model._set_inputs(dummy_x)
    #print(model(dummy_x).shape)    

    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    history_resnet[optimizer] = model.fit(trainX, trainY, batch_size= batch_size , epochs= epochs, validation_data=(testX, testY), verbose=1, callbacks=[csv_logger])



#train plot
plt.figure(1)
for optimizer in ['padam', 'adam', 'adamw', 'amsgrad', 'sgd']:
    op = optim_params[optimizer]
    train_loss = history_resnet[optimizer].history('loss')

    epoch_count = range(1, len(train_loss) + 1)
    plt.plot(epoch_count, train_loss, color=op['color'], linestyle=op['linestyle'])
plt.legend(['padam', 'adam', 'adamw', 'amsgrad', 'sgd'])
plt.xlabel('Epochs')
plt.ylabel('Train Loss')

#test plot
plt.figure(2)
for optimizer in ['padam', 'adam', 'adamw', 'amsgrad', 'sgd']:
    op = optim_params[optimizer]
    test_loss = history_resnet[optimizer].history('val_loss')
    epoch_count = range(1, len(test_loss) + 1)
    plt.plot(epoch_count, test_loss, color=op['color'], linestyle=op['linestyle'])
plt.legend(['padam', 'adam', 'adamw', 'amsgrad', 'sgd'])
plt.xlabel('Epochs')
plt.ylabel('Test Error')

plt.show()
