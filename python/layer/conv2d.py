from python.layer.base_layer import Layer
import numpy as np
from python.initializer import initializers
from python.regularizer import regularizers
from python.utils import conv_utils
import warnings


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
                groups=1,
                training=True
                ):

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2)
        self.strides = conv_utils.normalize_tuple(strides, 2)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2)
        self.padding = conv_utils.valid_check_padding(padding)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_regularizer = regularizers.get(bias_regularizer)
        self.dtype = dtype
        self.groups = groups
        self.training = training

    def compute_output_shape(self, input_shape):
        # O = (I - K + 2P) / S + 1
        ih, iw = input_shape[1:3]
        oh, ph = conv_utils.compute_output_padding_size(self.padding, ih, self.kernel_size[0], self.strides[0])
        ow, pw = conv_utils.compute_output_padding_size(self.padding, iw, self.kernel_size[1], self.strides[1])
        
        self.padding_size = (ph, pw)

        return (input_shape[0], oh, ow, self.filters)
        
    def build(self, input_shape):
        # Input shape :  N x XH x XW x XC
        # Output shape : N x YH x YW x YC
        # W :            XC x YC x KH x KW
        # b :            1 x YC x 1 x 1

        input_channel = input_shape[-1]
        
        kernel_shape = (input_channel, self.filters, *self.kernel_size)
        self.weight = self.kernel_initializer(kernel_shape, dtype=self.dtype)

        if self.use_bias:
            bias_shape = (self.filters, )
            self.bias = self.bias_initializer(bias_shape, dtype=self.dtype)
        
        self.output_size = self.compute_output_shape(input_shape)

    def forward(self, inputs):
        if inputs.dtype != self.dtype:
            warnings.warn("입력과 커널의 데이터 타입이 일치하지 않습니다. input dtype : {}, kernel dtype : {}".format(inputs.dtype, self.weight.dtype))
            inputs = inputs.astype(self.dtype)

        
        x = inputs

        # ( N x YH x YW , XC x KH x KW )
        flat_x = conv_utils.img2col(x, self.output_size, self.kernel_size, self.strides, self.padding_size)
        # ( XC x KW x KW , YC )
        flat_w = self.weight.reshape(self.filters, -1).T
        # ( N x YH x YW , YC )
        flat_y = np.dot(flat_x, flat_w) + self.bias
        # Reshape : ( N x YH x YW x YC )
        # Transpose : ( N x YC x YH x YW )
        y = flat_y.reshape(*self.output_size).transpose(0, 3, 1, 2)
        
        if self.training:
            self.x = x
            self.flat_x = flat_x
            self.flat_w = flat_w
    
        return y


    
