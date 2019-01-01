from __future__ import absolute_import, division, print_function

import tensorflow as tf
import keras.backend as K
import numpy as np

tf.enable_eager_execution()
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras import regularizers

cfg = {
    'test': [64, 'M', 128, 'M', 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes, weight_decay):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.num_classes = num_classes
        self.wd = weight_decay
        self.convlayers = self._make_convlayers(cfg[vgg_name])
        self.fc_layers = self._make_fc_layers(num_classes)

    def _make_convlayers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
            else:
                layers.append(tf.keras.layers.Conv2D(x, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.wd)))
                channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
                layers.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
                layers.append(tf.keras.layers.Activation('relu')) 
        return layers

    def _make_fc_layers(self, num_classes):
        layers=[]
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.Dense(self.num_classes, kernel_regularizer=regularizers.l2(self.wd)))
        return layers
    
    def call(self, inputs):
        prev_out = inputs
        for layer in self.convlayers:
            prev_out = layer(prev_out)
        for layer in self.fc_layers:
            prev_out = layer(prev_out)
        return tf.nn.softmax(prev_out)

if __name__ == '__main__':
    batch_size = 1
    nb_epoch = 1
    img_rows, img_cols = 32, 32
    epochs = 1

    # (trainX, trainY), (testX, testY) = cifar10.load_data()

    # trainX = trainX.astype('float32')
    # trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
    # testX = testX.astype('float32')
    # testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

    # trainY = kutils.to_categorical(trainY)
    # testY = kutils.to_categorical(testY)

    # testY = tf.one_hot(testY, depth=10).numpy()
    # trainY = tf.one_hot(trainY, depth=10).numpy()

    # testY = testY.astype(np.int64)
    # testX = testX.astype(np.int64)


    model = VGG('test', 10)

    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])

    dummy_x = tf.zeros((1, 32, 32, 3))
    model._set_inputs(dummy_x)
    print(model(dummy_x).shape)
    # train
    # model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs,
              # validation_data=(testX, testY), verbose=1)

    # evaluate on test set
    # scores = model.evaluate(testX, testY, batch_size, verbose=1)
    # print("Final test loss and accuracy :", scores)
