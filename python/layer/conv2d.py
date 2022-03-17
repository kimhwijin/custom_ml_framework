from python.layer.base_layer import Layer
import numpy as np
from python.initializer import initializers
from python.regularizer import regularizers
from python.utils import conv_utils

class Conv2D(Layer):
    def __init__(self, 
                filters, 
                kernel_size=(3,3), 
                strides=(1,1), 
                dilation_rate=(1,1),
                padding='valid', 
                use_bias=True, 
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                dtype=np.float32,
                groups=1
                ):
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2)
        self.strides = conv_utils.normalize_tuple(strides, 2)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2)
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding == 'causal':
            raise ValueError('Causal 패딩은 Conv1D 에서만 사용됩니다.')
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_regularizer = regularizers.get(bias_regularizer)
        self.dtype = dtype
        self.groups = groups

    def compute_output_shape(self, input_shape):
        #input_shape : n x filters x width x height
        kernel_width, kernel_height = self.kernel_size
        input_width, input_height = input_shape[-2:]
    
    def build(self, input_shape):
        #nx32x32x64 -> 128x3x3
        #w : 64x128x3x3
        #b : 1x128x1x1
        #nx32x32x128

        input_channel = input_shape[-1]
        
        kernel_shape = (input_channel, self.filters, *self.kernel_size)
        self.weight = self.kernel_initializer(kernel_shape, dtype=self.dtype)

        if self.use_bias:
            bias_shape = (1, self.filters, ) + (1,) * len(self.kernel_size)
            self.bias = self.bias_initializer(bias_shape, dtype=self.dtype)
        