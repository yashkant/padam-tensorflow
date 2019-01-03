from __future__ import absolute_import, division, print_function
import os 
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]= "1"
#os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
print(os.getcwd())
import tensorflow as tf
import keras.backend as K
import numpy as np
tf.enable_eager_execution()
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from eager_resnet import Resnet
import matplotlib.pyplot as plt
import h5py

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

print(sys.path)

from amsgrad import AMSGrad
from padam import Padam

pretrained = False
dataset = 'cifar10'

if dataset == 'cifar10':
    MEAN = [0.4914, 0.4822, 0.4465]
    STD_DEV = [0.2023, 0.1994, 0.2010]
    from keras.datasets import cifar10
    (trainX, trainY), (testX, testY) = cifar10.load_data()

elif dataset == 'cifar100':
    MEAN = [0.507, 0.487, 0.441]
    STD_DEV = [0.267, 0.256, 0.276]
    from keras.datasets import cifar100
    (trainX, trainY), (testX, testY) = cifar100.load_data()

def preprocess(t):
    paddings = tf.constant([[2, 2,], [2, 2],[0,0]])
    t = tf.pad(t, paddings, 'CONSTANT')
    t = tf.image.random_crop(t, [32, 32, 3])
    t = normalize(t) 
    return t

def normalize(t):
    t = tf.div(tf.subtract(t, MEAN), STD_DEV) 
    return t

def save_model(filepath, model):
    file = h5py.File(filepath,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight'+str(i),data=weight[i])
    file.close()

def load_model(filepath, model):
    file=h5py.File(filepath,'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    return model

hyperparameters = {
    'cifar10': {
        'epoch': 200,
        'batch_size': 128,
        'decay_after': 50,
        'classes':10
    },
    'cifar100': {
        'epoch': 200,
        'batch_size': 128,
        'decay_after': 50,
        'classes':100 
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
        'linestyle':'-'
    },
    'adam': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'orange',
        'linestyle':'--'
    },
    'adamw': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'magenta',
        'linestyle':'--'
    },
    'amsgrad': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color' : 'darkgreen',
        'linestyle':'-.'
    },
    'sgd': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'm': 0.9,
        'color': 'blue',
        'linestyle':'-'
    }
}


hp = hyperparameters[dataset]
batch_size = hp['batch_size']
epochs = hp['epoch']

#(trainX, trainY), (testX, testY) = (trainX[:2], trainY[:2]), (testX[:2], testY[:2] )

trainX = trainX.astype('float32')
# trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
trainX = trainX/255
testX = testX.astype('float32')
# testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
testX = testX/255

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)
print("Image format:",K.image_data_format())

#testY = testY.astype(np.int64)
#trainY = trainY.astype(np.int64)
#testY = tf.one_hot(testY, depth=10).numpy()
#trainY = tf.one_hot(trainY, depth=10).numpy()

tf.train.create_global_step()

train_size = trainX.shape[0]

datagen_train = ImageDataGenerator(
                            preprocessing_function=preprocess,
                            horizontal_flip=True,
                            )
datagen_test = ImageDataGenerator(
                            preprocessing_function=normalize,
                            )


# resnet cifar10 training and plots

optim_array = ['padam', 'adam', 'adamw', 'amsgrad', 'sgd']

history_resnet = {}

for optimizer in optim_array:

    logfile = 'log_'+optimizer+ '_' + dataset +'.csv'
    f = open(logfile, "w+")

    op = optim_params[optimizer]

    if optimizer == 'adamw' and dataset=='imagenet':
        op['weight_decay'] = 0.05 


    if optimizer is not 'adamw':
        model = Resnet(training= True, data_format= K.image_data_format(), classes = 10, wt_decay = op['weight_decay'])
    else:
        model = Resnet(training= True, data_format= K.image_data_format(), classes = 10, wt_decay = 0)

    learning_rate = tf.train.exponential_decay(op['lr'], tf.train.get_global_step() * batch_size,
                                       hp['decay_after']*train_size, 0.1, staircase=True)
    if optimizer == 'padam':
        optim = Padam(learning_rate=learning_rate, p=op['p'], beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'adamw':
        adamw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
        optim = adamw(weight_decay=op['weight_decay'], learning_rate=learning_rate,  beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'amsgrad':
        optim = AMSGrad(learning_rate=learning_rate, beta1=op['b1'], beta2=op['b2'])
    elif optimizer == 'sgd':
        optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=op['m'])

    dummy_x = tf.zeros((batch_size, 32, 32, 3))
    
    model._set_inputs(dummy_x)
    #model(dummy_x)
    #print(model(dummy_x).shape)
    
    filepath = 'model_'+optimizer+'.h5'

    if pretrained:
        model = load_model(filepath, model)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'], global_step=tf.train.get_global_step())
    
    else :
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'], global_step=tf.train.get_global_step())
        csv_logger = CSVLogger(logfile, append=True, separator=';')
        history_resnet[optimizer] = model.fit_generator(datagen_train.flow(trainX, trainY, batch_size = batch_size), epochs = epochs, 
                                                                 validation_data = datagen_test.flow(testX, testY, batch_size = batch_size), verbose=1, callbacks = [csv_logger])
    
 
    scores = model.evaluate_generator(datagen_test.flow(testX, testY, batch_size = batch_size), verbose=1)
    print("Final test loss and accuracy:", scores)
    
    save_model(filepath, model)
    f.close()



if (pretrained==False):
    #train plot      
    plt.figure(1)
    for optimizer in optim_array:
        op = optim_params[optimizer]
        train_loss = history_resnet[optimizer].history['loss']
        epoch_count = range(1, len(train_loss) + 1)
        plt.plot(epoch_count, train_loss, color=op['color'], linestyle=op['linestyle'])
    plt.legend(optim_array)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    
    #test plot
    plt.figure(2)
    for optimizer in optim_array:
        op = optim_params[optimizer]
        test_loss = history_resnet[optimizer].history['val_loss']
        epoch_count = range(1, len(test_loss) + 1)
        plt.plot(epoch_count, test_loss, color=op['color'], linestyle=op['linestyle'])
    plt.legend(optim_array)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    
    #plt.show()
    plt.savefig('figure_'+dataset+'.png')