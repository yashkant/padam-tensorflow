from __future__ import absolute_import, division, print_function

import warnings
import tensorflow as tf
import os
import sys
import keras.backend as K
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
tf.enable_eager_execution()
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras import regularizers

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
# DEFAULT_DTYPE = tf.float32

class Resnet(tf.keras.Model):
    
    def build_blocks(self, out_filters, num_blocks, stride):
        blocks = []
        strides = [stride] + [1]*(num_blocks-1)
        for _stride in strides:
            blocks.append(self.make_basic_block(self.in_filters, out_filters, _stride))
            self.in_filters = out_filters
        return blocks

    def make_basic_block(self, in_filters, out_filters, stride=1):

        block = []

        layers = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters = out_filters, kernel_size = 3, strides= stride, padding = "valid", use_bias = False, kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(self.wd)),
            tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                                                            scale=True, fused=True),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters = out_filters, kernel_size = 3, strides= 1, padding = "valid", use_bias = False, kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(self.wd)),
            tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                                                            scale=True, fused=True),

            # tf.keras.layers.Activation('relu')
            ])

        if stride != 1 or in_filters != out_filters:
            shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters = out_filters, kernel_size = 1, strides= stride, padding = "valid", use_bias = False, kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(self.wd)),
                 tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                                                            scale=True, fused=True)])
        else:
            shortcut = tf.keras.Sequential([])

        block.append(layers)
        block.append(shortcut)

        return block

    def _create_ResnetModel(self, filters):
        # filters : Number of fliters used in 1st blk (after initial layer)
        # kernel_size : kerel size for all the basic building blocks.
        # blocklist : List of values, each value shows the number of basic building blocks in the ith blk.
        # strides : stride for the 1st conv layer of all 4 blks (each basic building blk made of 2 conv layers)
        # For resnet18, blocklist = [2, 2, 2, 2], eack blk made of 2 basic building blocks.

        self.intial_conv = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding = "valid", data_format = self.data_format, use_bias = False, kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(self.wd)),
            tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                                                            scale=True, fused=True),
            tf.keras.layers.Activation('relu')
            ])

        self.blocks_1 = self.build_blocks(64, self.block_list[0], stride=1)
        self.blocks_2 = self.build_blocks(128, self.block_list[1], stride=2)
        self.blocks_3 = self.build_blocks(256, self.block_list[2], stride=2)
        self.blocks_4 = self.build_blocks(512, self.block_list[3], stride=2)

        self.final_dense = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=4), 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.classes, activation='softmax', kernel_regularizer=regularizers.l2(self.wd))
            ])
    
    
    # resnet with basic building blocks
    def __init__(self, data_format, initial_filters=64, block_list=[2, 2, 2, 2], classes=10, wt_decay = 0.001):

        """training: Either True or False, whether we are currently training the
           model. Needed for batch norm.
          data_format: The input format ('channels_last' or 'channels_first')."""
    
        super(Resnet, self).__init__()
    
        self.in_filters = initial_filters
        self.block_list = block_list
        self.num_blocks = len(block_list)
        self.data_format = data_format
        self.wd = wt_decay
        self.classes = classes
        # self.training = training
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
    
        self._create_ResnetModel(filters = initial_filters)

        # self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # self.flatten = tf.keras.layers.Flatten(data_format = self.data_format)
        # self.fc = tf.keras.layers.Dense(self.classes, kernel_regularizer=regularizers.l2(self.wd))
    
    def call(self, inputs):
        """if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])"""

        inputs = self.intial_conv(inputs)

        for block in self.blocks_1:
            block_x = block[0](inputs)
            block_y = block[1](inputs)
            inputs = tf.keras.layers.Activation('relu')(block_x+ block_y)

        for block in self.blocks_2:
            block_x = block[0](inputs)
            block_y = block[1](inputs)
            inputs = tf.keras.layers.Activation('relu')(block_x+ block_y)

        for block in self.blocks_3:
            block_x = block[0](inputs)
            block_y = block[1](inputs)
            inputs = tf.keras.layers.Activation('relu')(block_x+ block_y)

        for block in self.blocks_4:
            block_x = block[0](inputs)
            block_y = block[1](inputs)
            inputs = tf.keras.layers.Activation('relu')(block_x+ block_y)

        return self.final_dense(inputs)

if __name__ == '__main__':

    batch_size = 128
    img_rows, img_cols = 32, 32
    epochs = 200

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainX = trainX.astype('float32')
    trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
    testX = testX.astype('float32')
    testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
    
    trainY = kutils.to_categorical(trainY, num_classes = 10)
    testY = kutils.to_categorical(testY, num_classes = 10)
    
    #testY = testY.astype(np.int64)
    #trainY = trainY.astype(np.int64)

    print(K.image_data_format())
    model = Resnet(data_format='channels_last')

    model._set_inputs(tf.zeros((batch_size, 32, 32, 3)))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs,validation_data=(testX, testY), verbose=1)
    scores = model.evaluate(testX, testY, batch_size, verbose=1)
     
 
 
 
