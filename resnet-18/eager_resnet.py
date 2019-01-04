from __future__ import absolute_import, division, print_function

import warnings
import tensorflow as tf

import keras.backend as K
import numpy as np
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
tf.enable_eager_execution()
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras import regularizers


_BATCH_NORM_DECAY = 0.1
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32


class Resnet(tf.keras.Model):
    
    def conv2d_fixed_padding(self, filters, kernel_size, strides):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        # Returns output feature map of same spatial size as input if stride=1 else 
        # halves the input dimensions.
        model_x = []
        if strides > 1:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            model_x.append(tf.keras.layers.ZeroPadding2D([[pad_beg, pad_end], [pad_beg, pad_end]], data_format = self.data_format))
            model_x.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "valid", data_format = self.data_format, use_bias = False,
                           kernel_regularizer=regularizers.l2(self.wd)))
            #kernel_initializer=  tf.keras.initializers.VarianceScaling(scale=1.0/27, mode='fan_in',distribution='uniform',
            #model_x = tf.keras.Sequential([tf.keras.layers.ZeroPadding2D([[pad_beg, pad_end], [pad_beg, pad_end]], data_format = self.data_format),
            #	tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "valid", data_format = self.data_format, use_bias = False,
            #	           kernel_regularizer=regularizers.l2(self.wd))])
           # model_x.append(tf.keras.layers.ZeroPadding2D([[pad_beg, pad_end], [pad_beg, pad_end]], data_format = self.data_format))
            #return model_x
        else : 
            model_x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding = "same", data_format = self.data_format, use_bias = False,
                kernel_regularizer=regularizers.l2(self.wd))
            #return tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding = "same", data_format = self.data_format, use_bias = False,
             #   kernel_regularizer=regularizers.l2(self.wd) )
        return model_x
        #model_x.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "same", data_format = self.data_format, use_bias = False, kernel_initializer='VarianceScaling' ))
              
     
    
    def _building_block_v1(self, filters, strides):
        """A basic building block, without a bottleneck.
        Args:
          strides: The stride to use for the first layer out of the two (2nd layer always has 
          stride 1)
        """
        layer_a = [] # has 5 elements
  
        
        layer_a.append(self.conv2d_fixed_padding(filters = filters, kernel_size = 3, strides = strides))
        layer_a.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, fused=True, gamma_initializer = tf.keras.initializers.RandomUniform(minval = 0, maxval = 1.0)))
        layer_a.append(tf.keras.layers.Activation('relu'))
        layer_a.append(self.conv2d_fixed_padding(filters = filters, kernel_size = 3, strides=1))
        layer_a.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, fused=True, gamma_initializer = tf.keras.initializers.RandomUniform(minval = 0, maxval = 1.0)))
    
        return layer_a
    
    def block_layer(self, filters, strides, blocks):
        """Creates one layer of blocks for the ResNet model.
        Args:
          filters: The number of filters for this blk.
          blocks: The number of basic building blocks contained in the layer.
          strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.

        """
        model_y = []
        model_y.append(self._building_block_v1(filters, strides))
    
        for i in range(1, blocks):
            model_y.append(self._building_block_v1(filters, 1))
        
        return model_y
    
    
    def _create_ResnetModel(self, filters = 64, kernel_size = 3, strides = 2):
        # filters : Number of fliters used in 1st blk (after initial layer)
        # kernel_size : kerel size for all the basic building blocks.
        # blocklist : List of values, each value shows the number of basic building blocks in the ith blk.
        # strides : stride for the 1st conv layer of all 4 blks (each basic building blk made of 2 conv layers)
        # For resnet18, blocklist = [2, 2, 2, 2], eack blk made of 2 basic building blocks.
        model = []    
        # conv1
        initial_layer = self.conv2d_fixed_padding(filters = 64, kernel_size = 3, strides = 1)      
        model.append([[initial_layer, tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, fused=True, gamma_initializer = tf.keras.initializers.RandomUniform(minval = 0, maxval = 1.0))]])
        
        # conv2, conv3, conv4, conv5
        for i in range(len(self.block_list)):
            num_filters = filters * (2**i)
            blk = self.block_layer(num_filters, strides=(1 if i==0 else strides), blocks=self.block_list[i])
            model.append(blk)
    
        return model
    
    
    # resnet with basic building blocks
    def __init__(self, training, data_format, initial_filters=64, block_list=[2, 2, 2, 2], classes=10, wt_decay = 0.001):

        """training: Either True or False, whether we are currently training the
           model. Needed for batch norm.
          data_format: The input format ('channels_last' or 'channels_first')."""
    
        super(Resnet, self).__init__()
    
        self.initial_filters = initial_filters
        self.block_list = block_list
        self.num_blocks = len(block_list)
        self.data_format = data_format
        self.wd = wt_decay
        self.classes = classes
        self.training = training
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        
        
        # self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, fused=True, gamma_initializer = tf.keras.initializers.RandomUniform(minval = 0, maxval = 1.0))
        #self.relu = tf.keras.layers.Activation('relu')

        self.model = self._create_ResnetModel(filters = self.initial_filters)

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten(data_format = self.data_format)
    
    def call(self, inputs, training=None, mask=None):
        """if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])"""
        #print(inputs.shape)
        #print(inputs)
        inputs = self.model[0][0][0](inputs)
        inputs = self.model[0][0][1](inputs)

        inputs = tf.keras.layers.Activation('relu')(inputs)

        #print(inputs.shape)
        for blk in range(self.num_blocks):
            blk_index = blk+1
            for basic_bblk in range(self.block_list[blk]):
                # 1st basic building block in each blk uses proj shortcut and stride of 2 else normal shortcut
                #print(inputs.shape)
                if (basic_bblk == 0 and blk!=0):
                    short = self.conv2d_fixed_padding(filters=self.initial_filters*(2**blk), kernel_size=1, strides=2)[0](inputs)
                    short = self.conv2d_fixed_padding(filters=self.initial_filters*(2**blk), kernel_size=1, strides=2)[1](short)
                    #sprint(blk)
                    #print(short.shape)

                    short = tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                     scale=True, trainable=training, fused=True, gamma_initializer = tf.keras.initializers.RandomUniform(minval = 0, maxval = 1.0))(short)

                    #print(short.shape)
                else :
                    short = inputs
                    #print(inputs.shape)
                for lyr in range(len(self.model[blk_index][basic_bblk])):
                    if blk!=0 and basic_bblk==0:
                        if lyr==0:
                            for q in range(len(self.model[blk_index][basic_bblk][lyr])):
                                inputs = self.model[blk_index][basic_bblk][lyr][q](inputs)
                        else:
                        	inputs = self.model[blk_index][basic_bblk][lyr](inputs)

                    else:
                        inputs = self.model[blk_index][basic_bblk][lyr](inputs)
                    #print(inputs.shape)
                #print(inputs.shape)  
                #print(short.shape)
                inputs = inputs + short
                inputs = tf.keras.layers.Activation('relu')(inputs)

        #print(inputs.shape)
        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
        # ResNet does an Average Pooling layer over pool_size,
        # but that is the same as doing a reduce_mean. We do a reduce_mean
        # here because it performs better than AveragePooling2D.
        """axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        inputs = tf.reduce_mean(inputs, axes, keepdims=True)
        print(inputs.shape)
        inputs = tf.squeeze(inputs, axes)
        print(inputs.shape)
        inputs = tf.layers.dense(inputs=inputs, units=self.classes, kernel_regularizer=regularizers.l2(self.wd))
        print(inputs.shape)
        return inputs"""
    
        #inputs = tf.squeeze(inputs, axes)
        inputs = self.avg_pool(inputs)
        #print(inputs.shape)
        inputs = self.flatten(inputs) 
        #print(inputs.shape)
        
        
        inputs = tf.keras.layers.Dense(self.classes, kernel_regularizer=regularizers.l2(self.wd))#, kernel_initializer =tf.keras.initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform', seed=None)
                                           #, bias_initializer = tf.keras.initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform', seed=None))(inputs)

        #print(inputs)
        return inputs  #tf.nn.softmax(inputs)


if __name__ == '__main__':
    batch_size = 128
    nb_epoch = 2
    img_rows, img_cols = 32, 32
    epochs = 200

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    #(trainX, trainY), (testX, testY) = (trainX[:2], trainY[:2]), (testX[:2], testY[:2])

    trainX = trainX.astype('float32')
    trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
    testX = testX.astype('float32')
    testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
    
    trainY = kutils.to_categorical(trainY, num_classes = 10)
    testY = kutils.to_categorical(testY, num_classes = 10)
    
    #testY = testY.astype(np.int64)
    #trainY = trainY.astype(np.int64)
    #testY = tf.one_hot(testY, depth=10).numpy()
    #trainY = tf.one_hot(trainY, depth=10).numpy()


    print(K.image_data_format())
    #testY = testY.astype(np.int64)
    model = Resnet(training= False, data_format='channels_last')

    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])

    dummy_x = tf.zeros((10, 300, 300, 3))
    model._set_inputs(dummy_x)
    #print(model(dummy_x).shape)

    model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs,validation_data=(testX, testY), verbose=1)
    scores = model.evaluate(testX, testY, batch_size, verbose=1)
    #print("Final test loss and accuracy :", scores)
    
    
 
 
 
 
 
