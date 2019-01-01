from __future__ import absolute_import, division, print_function

import warnings
import tensorflow as tf

import keras.backend as K
import numpy as np

tf.enable_eager_execution()
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32

class Resnet(tf.keras.Model):
    
    def conv2d_fixed_padding(self, filters, kernel_size, strides):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        # Returns output feature map of same spatial size as input if stride=1 else 
        # halves the input dimensions.
        if strides > 1:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            model_x = tf.keras.Sequential([tf.keras.layers.ZeroPadding2D([[pad_beg, pad_end], [pad_beg, pad_end]], data_format = self.data_format),tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "valid", data_format = self.data_format, use_bias = False, kernel_initializer='VarianceScaling' )])
           # model_x.append(tf.keras.layers.ZeroPadding2D([[pad_beg, pad_end], [pad_beg, pad_end]], data_format = self.data_format))
            return model_x
        else : 
            return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "same", data_format = self.data_format, use_bias = False, kernel_initializer='VarianceScaling' )
        #model_x.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "same", data_format = self.data_format, use_bias = False, kernel_initializer='VarianceScaling' ))
              
     
    
    def _building_block_v1(self, filters, strides):
        """A basic building block, without a bottleneck.
        Args:
          strides: The stride to use for the first layer out of the two (2nd layer always has 
          stride 1)
        """
        layer_a = [] # has 5 elements
  
        
        layer_a.append(self.conv2d_fixed_padding(filters = filters, kernel_size = 3, strides = strides))
        layer_a.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, trainable=self.training, fused=True))
        layer_a.append(tf.keras.layers.Activation('relu'))
    
        layer_a.append(self.conv2d_fixed_padding(filters = filters, kernel_size = 3, strides=1))
        layer_a.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, trainable=self.training, fused=True))
    
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
        conv1=[]
        initial_layer = self.conv2d_fixed_padding(filters = 64, kernel_size = 7, strides = 2)
        conv1.append(initial_layer)
        conv1.append(tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = "same", data_format = self.data_format))
        
        model.append([conv1])
        
        # conv2, conv3, conv4, conv5
        for i in range(len(self.block_list)):
            num_filters = filters * (2**i)
            blk = self.block_layer(num_filters, strides=(1 if i==0 else strides), blocks=self.block_list[i])
            model.append(blk)
    
        return model
    
    
    # resnet with basic building blocks
    def __init__(self, training, data_format, initial_filters=64, block_list=[2, 2, 2, 2], classes=10):

        """training: Either True or False, whether we are currently training the
           model. Needed for batch norm.
          data_format: The input format ('channels_last' or 'channels_first')."""
    
        super(Resnet, self).__init__()
    
        self.initial_filters = initial_filters
        self.block_list = block_list
        self.num_blocks = len(block_list)
        self.data_format = data_format
        self.classes = classes
        self.training = training
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
    
        self.model = self._create_ResnetModel(filters = initial_filters)
    
    def call(self, inputs, training=None, mask=None):
        """if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])"""
        
        for i in range(len(self.model[0][0])):
            inputs = self.model[0][0][i](inputs)
    
        for blk in range(self.num_blocks):
            blk_index = blk+1
            for basic_bblk in range(self.block_list[blk]):
                # 1st basic building block in each blk uses proj shortcut and stride of 2 else normal shortcut
                #print(inputs.shape)
                if (basic_bblk == 0 and blk!=0):
                    short = self.conv2d_fixed_padding(filters=self.initial_filters*(2**blk), kernel_size=1, strides=2)(inputs)
                    #sprint(blk)
                    #print(short.shape)
                    short = tf.keras.layers.BatchNormalization(axis=self.channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                     scale=True, trainable=training, fused=True)(short)
                    #print(short.shape)
                else :
                    short = inputs
                    #print(inputs.shape)
                for lyr in range(len(self.model[blk_index][basic_bblk])):
                    inputs = self.model[blk_index][basic_bblk][lyr](inputs)
                    #print(inputs.shape)
                #print(inputs.shape)  
                #print(short.shape)
                inputs = inputs + short
                inputs = tf.nn.relu(inputs)
    
        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
        # ResNet does an Average Pooling layer over pool_size,
        # but that is the same as doing a reduce_mean. We do a reduce_mean
        # here because it performs better than AveragePooling2D.
        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    
        inputs = tf.squeeze(inputs, axes)
        inputs = tf.layers.dense(inputs=inputs, units=self.classes)
    
        return inputs


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


    model = Resnet(training= False, data_format='channels_last')

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
    
    
 
 
 
 
 
