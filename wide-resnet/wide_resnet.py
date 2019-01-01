# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.

# Reference

- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

"""

import warnings
import tensorflow as tf
import keras.backend as K
import numpy as np 


# keras.layers.ZeroPadding2D(padding="VALID")


class WRNModel(tf.keras.Model):

    def __conv1_block(self, multiplier):

        model = []
        out_channels = 16*multiplier
        model.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model.append(tf.keras.layers.Conv2D(out_channels, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        # TODO: Fix batchnorm weight initialization
        model.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model.append(tf.keras.layers.Activation('relu'))
        return out_channels, [model]


    def __conv2_block(self, input_channels, k=1, dropout=0.0):

        model_x = []
        model_y = []
        out_channels = 16*k
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        # # Check if input number of filters is same as 16 * k, else create convolution2d for this input
        # if input_channels != 16 * k:
        #     model_x.append(tf.keras.layers.Conv2D(16 * k, (1, 1), activation='linear', padding="VALID", kernel_initializer = self.conv_w_init))
        model_y.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model_y.append(tf.keras.layers.Conv2D(16 * k, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg)) 
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis)) 
        model_y.append( tf.keras.layers.Activation('relu'))

        # if dropout > 0.0:
        #     model_y.append(tf.keras.layers.Dropout(dropout)(x)) 
        model.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model_y.append(tf.keras.layers.Conv2D(16 * k, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        return out_channels, [model_x, model_y]


    def __conv3_block(self, input_channels, k=1, dropout=0.0, stride = 1):

        model_x = []
        model_y = []
        out_channels = 32*k

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if stride != 1 or input_channels != 32 * k:
            model_x.append(tf.keras.layers.Conv2D(32 * k, (1, 1), strides = stride, padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))

        model_y.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model_y.append(tf.keras.layers.Conv2D(32 * k, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        # if dropout > 0.0:
        #     model_y.append(tf.keras.layers.Dropout(dropout))
            
        model_y.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model_y.append(tf.keras.layers.Conv2D(32 * k, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        return out_channels, [model_x, model_y]


    def __conv4_block(self, input_channels, k=1, dropout=0.0, stride = 1):

        model_x = []
        model_y = []
        out_channels = 64*k

        channel_axis = 1 if K.image_dim_ordering() == 'th' else -1

        if stride != 1 or input_channels != 64 * k:
            model_x.append(tf.keras.layers.Conv2D(64 * k, (1, 1), strides = stride,  padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        
        model_y.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model_y.append(tf.keras.layers.Conv2D(64 * k, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        # if dropout > 0.0:
        #     model_y.append(tf.keras.layers.Dropout(dropout))

        model_y.append(tf.keras.layers.ZeroPadding2D(padding=1))
        model_y.append(tf.keras.layers.Conv2D(64 * k, (3, 3), padding="VALID", kernel_initializer = self.conv_w_init, kernel_regularizer=self.l2_reg))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))


        return out_channels, [model_x, model_y]



    # Enclose the function below in the call function of the model! 


    def __create_wide_residual_network(self, nb_classes, include_top, depth=28,
                                       multiplier=4, dropout=0.0, activation='softmax'):
        ''' 

        Creates a Wide Residual Network model and stores it to an array and returns it.
        Each element in the array is a block that constitutes a single block of the 
        network. Inside each block we there are non-zero paths that we sum at last 
        to build this block. 

        Args:
            nb_classes: Number of output classes
            img_input: Input tensor or layer
            include_top: Flag to include the last dense layer
            depth: Depth of the network. Compute N = (n - 4) / 6.
                   For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
                   For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
                   For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
            multiplier: multiplier of the network.
            dropout: Adds dropout if value is greater than 0.0

        Returns:a Keras Model
        '''

        N = (depth - 4) // 6

        model = []
        channels, blk = self.__conv1_block(multiplier)
        model.append(blk)
        nb_conv = 4


        for i in range(N):
            channels, blk = self.__conv2_block(channels, multiplier, dropout)
            model.append(blk)
            nb_conv += 2

        # model.append([[tf.keras.layers.MaxPooling2D((2, 2))]])


        for i in range(N):
            if(i == 0): 
                channels, blk = self.__conv3_block(channels, multiplier, dropout, stride = 2)
            channels, blk = self.__conv3_block(channels, multiplier, dropout)

            model.append(blk)
            nb_conv += 2

        # model.append([[tf.keras.layers.MaxPooling2D((2, 2))]])


        for i in range(N):
            if(i == 0): 
                channels, blk = self.__conv3_block(channels, multiplier, dropout, stride = 2)
            channels, blk = self.__conv3_block(channels, multiplier, dropout)

            model.append(blk)
            nb_conv += 2

        # Adding average pool instead of GAP! 
        if include_top:
            model.append([[tf.keras.layers.AveragePooling2D(pool_size=8), tf.keras.Flatten(), tf.keras.layers.Dense(nb_classes, activation=activation, kernel_regularizer=self.l2_reg)]])

        return model

    def __init__(self, depth=28, multiplier=4, dropout_rate=0.0,
                        include_top=True,
                        input_tensor=None, input_shape=None,
                        classes=10, activation='softmax', wd = 1e-4):

        super(WRNModel, self).__init__()
        
        self.n = int((depth-4)/6)
        self.conv_w_init = tf.initializers.random_normal(mean=0.0, stddev= np.sqrt(2.0/self.n))
        self.bn_w_init = tf.constant_initializer(1.0)
        self.bn_b_init = tf.constant_initializer(0.0)
        self.wd = wd
        self.l2_reg = tf.keras.regularizers.l2(wd)
        self.depth = depth
        self.multiplier = multiplier
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.input_tensor = input_tensor
        

        # if (depth - 4) % 6 != 0:
        #     raise ValueError('Depth of the network must be such that (depth - 4)'
        #                      'should be divisible by 6.')

        self.model = self.__create_wide_residual_network(classes, include_top, depth, multiplier,
                                           dropout_rate, activation)


    def call(self, input):
        """Execute the model"""

        prev_blk_output = input 

        # print("Printing Model: ",self.model)

        for i_blk in range(0, len(self.model)):
            temp_paths = []
            for i_path in range(0, len(self.model[i_blk])):
                path_output = prev_blk_output
                for lyr in range(0, len(self.model[i_blk][i_path])):
                    path_output = self.model[i_blk][i_path][lyr](path_output)
                temp_paths.append(path_output)

            if(len(temp_paths) > 1):
                prev_blk_output = tf.keras.layers.Add()(temp_paths) 
            else:
                prev_blk_output = temp_paths[0]

        return prev_blk_output
