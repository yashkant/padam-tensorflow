import warnings
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32

class Resnet(tf.keras.Model):
    
    def conv2d_fixed_padding(self, filters, kernel_size, strides):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        # returns output feature map of same spatial size as input.
        model_x = []
        if strides > 1:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
    
            model_x.append(tf.keras.layers.ZeroPadding2D([[pad_beg, pad_end], [pad_beg, pad_end]], data_format = self.data_format))
    
        model_x.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding = "same", data_format = self.data_format, use_bias = False, kernel_initializer='VarianceScaling' ))
        
        return model_x
     
    
    def _building_block_v1(self, filters, strides):
        """A basic building block, without a bottleneck.
        Args:
          training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
          strides: The stride to use for the first layer out of the two (2nd layer always has 
          stride 1)
        """
        layer_a = [] # has 5 elements
    
        channel_axis = 1 if self.data_format == 'channels_first' else -1
    
        layer_a.append(self.conv2d_fixed_padding(filters = filters, kernel_size = 3, strides = strides))
        layer_a.append(tf.keras.layers.BatchNormalization(axis=channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.training, fused=True))
        layer_a.append(tf.keras.layers.Activation('relu'))
    
        layer_a.append(self.conv2d_fixed_padding(filters = filters, kernel_size = 3, strides=1))
        layer_a.append(tf.keras.layers.BatchNormalization(axis=channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.training, fused=True))
    
        return layer_a
    
    def block_layer(self, filters, strides, blocks):
        """Creates one layer of blocks for the ResNet model.
        Args:
          filters: The number of filters for this blk.
          bottleneck: Is the block created a bottleneck block.
          blocks: The number of basic building blocks contained in the layer.
          strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
          training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
          data_format: The input format ('channels_last' or 'channels_first').
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
        initial_layer = self.conv2d_fixed_padding(filters = 64, kernel_size = 7, strides = 2)
        model.append([[initial_layer, tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = "same", data_format = self.data_format)]])
        
        # conv2, conv3, conv4, conv5
        for i in range(len(self.block_list)):
            num_filters = filters * (2**i)
            blk = self.block_layer(num_filters, strides, self.block_list[i])
            model.append(blk)
    
        return model
    
    
    # resnet with basic building blocks
    def __init__(self, training, data_format, initial_filters=64, block_list=[2, 2, 2, 2], classes=10):
    
        super(ResNet, self).__init__()
    
        self.initial_filters = initial_filters
        self.block_list = block_list
        self.num_blocks = len(block_list)
        self.data_format = data_format
        self.classes = classes
        self.training = training
    
        self.model = self._create_ResnetModel(block_list, data_format, filters = initial_filters)
    
    def call(self, inputs, training=None, mask=None):
        """if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])"""
    
        for i in range(len(self.model[0][0])):
            inputs = model[0][0][i](inputs)
    
        for blk in range(self.num_blocks):
            blk_index = blk+1
            for basic_bblk in range(self.block_list[blk]):
                # 1st basic building block in each blk uses proj shortcut and stride of 2 else normal shortcut
                if (basic_bblk == 0 ): 
                    short = self.conv2d_fixed_padding(filters=self.initial_filters*(2**blk), kernel_size=1, strides=2)(inputs)
                    short = tf.keras.layers.BatchNormalization(axis=channel_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                     scale=True, training=training, fused=True)(short)
                else :
                    short = inputs
    
                for lyr in range(len(model[blk_index][basic_bblk])):
                    inputs = model[blk_index][basic_bblk][lyr](inputs)
                
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
    
    
 
 
 
 
 