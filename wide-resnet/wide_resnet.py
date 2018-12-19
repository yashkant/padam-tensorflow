# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.

# Reference

- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

"""

import warnings
import tensorflow as tf
import keras.backend as K



class WRNModel(tf.keras.Model):

    def __conv1_block(self):

        model = []
        out_channels = 16
        model.append(tf.keras.layers.Conv2D(out_channels, (3, 3), padding='same'))
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        model.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model.append(tf.keras.layers.Activation('relu'))
        return out_channels, [model]


    def __conv2_block(self, input_channels, k=1, dropout=0.0):

        model_x = []
        model_y = []
        out_channels = 16*k
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        # Check if input number of filters is same as 16 * k, else create convolution2d for this input
        if input_channels != 16 * k:
            model_x.append(tf.keras.layers.Conv2D(16 * k, (1, 1), activation='linear', padding='same'))

        model_y.append(tf.keras.layers.Conv2D(16 * k, (3, 3), padding='same')) 
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis)) 
        model_y.append( tf.keras.layers.Activation('relu'))

        if dropout > 0.0:
            model_y.append(tf.keras.layers.Dropout(dropout)(x)) 

        model_y.append(tf.keras.layers.Conv2D(16 * k, (3, 3), padding='same'))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        return out_channels, [model_x, model_y]


    def __conv3_block(self, input_channels, k=1, dropout=0.0):

        model_x = []
        model_y = []
        out_channels = 32*k

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if input_channels != 32 * k:
            model_x.append(tf.keras.layers.Conv2D(32 * k, (1, 1), activation='linear', padding='same'))

        model_y.append(tf.keras.layers.Conv2D(32 * k, (3, 3), padding='same'))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        if dropout > 0.0:
            model_y.append(tf.keras.layers.Dropout(dropout))

        model_y.append(tf.keras.layers.Conv2D(32 * k, (3, 3), padding='same'))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        return out_channels, [model_x, model_y]


    def __conv4_block(self, input_channels, k=1, dropout=0.0):

        model_x = []
        model_y = []
        out_channels = 64*k

        channel_axis = 1 if K.image_dim_ordering() == 'th' else -1

        if input_channels != 64 * k:
            model_x.append(tf.keras.layers.Conv2D(64 * k, (1, 1), activation='linear', padding='same'))

        model_y.append(tf.keras.layers.Conv2D(64 * k, (3, 3), padding='same'))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))

        if dropout > 0.0:
            model_y.append(tf.keras.layers.Dropout(dropout))

        model_y.append(tf.keras.layers.Conv2D(64 * k, (3, 3), padding='same'))
        model_y.append(tf.keras.layers.BatchNormalization(axis=channel_axis))
        model_y.append(tf.keras.layers.Activation('relu'))


        return out_channels, [model_x, model_y]



    # Enclose the function below in the call function of the model! 


    def __create_wide_residual_network(self, nb_classes, include_top, depth=28,
                                       width=8, dropout=0.0, activation='softmax'):
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
            width: Width of the network.
            dropout: Adds dropout if value is greater than 0.0

        Returns:a Keras Model
        '''

        N = (depth - 4) // 6

        model = []
        channels, blk = self.__conv1_block()
        model.append(blk)
        nb_conv = 4


        for i in range(N):
            channels, blk = self.__conv2_block(channels, width, dropout)
            model.append(blk)
            nb_conv += 2

        model.append([[tf.keras.layers.MaxPooling2D((2, 2))]])


        for i in range(N):
            channels, blk = self.__conv3_block(channels, width, dropout)
            model.append(blk)
            nb_conv += 2

        model.append([[tf.keras.layers.MaxPooling2D((2, 2))]])


        for i in range(N):
            channels, blk = self.__conv4_block(channels, width, dropout)
            model.append(blk)
            nb_conv += 2

        if include_top:
            model.append([[tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(nb_classes, activation=activation)]])

        return model

    def __init__(self, depth=28, width=8, dropout_rate=0.0,
                        include_top=True,
                        input_tensor=None, input_shape=None,
                        classes=10, activation='softmax'):

        super(WRNModel, self).__init__()

        self.depth = depth
        self.width = width
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.input_tensor = input_tensor
        

        if (depth - 4) % 6 != 0:
            raise ValueError('Depth of the network must be such that (depth - 4)'
                             'should be divisible by 6.')

        self.model = self.__create_wide_residual_network(classes, include_top, depth, width,
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
            print("Print block output shapes: ", prev_blk_output.shape)

        return prev_blk_output


    
