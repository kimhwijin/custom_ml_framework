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
        # W :            KH x KW x XC x YC
        # b :            1 x 1 x 1 x YC
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        _, _, _, XC = input_shape
        _, _, _, YC = self.output_shape
        KH, KW = self.kernel_size

        kernel_shape = (KH, KW, XC, YC)
        self.weight = self.kernel_initializer(kernel_shape, dtype=self.dtype)

        if self.use_bias:
            bias_shape = (1, 1, 1, YC)
            self.bias = self.bias_initializer(bias_shape, dtype=self.dtype)
        

    def forward(self, inputs):
        if inputs.dtype != self.dtype:
            warnings.warn("입력과 커널의 데이터 타입이 일치하지 않습니다. input dtype : {}, kernel dtype : {}".format(inputs.dtype, self.weight.dtype))
            inputs = inputs.astype(self.dtype)

        
        x = inputs

        # ( N x YH x YW , XC x KH x KW )
        flat_x = conv_utils.img2col(x, self.output_shape, self.kernel_size, self.strides, self.padding_size)
        

        # weight : KH x KW x XC x YC
        #Transpose : XC x KH x KW x YC
        #Reshape : ( XC x KH x KW , YC )
        flat_w = self.weight.transpose((2,0,1,3)).reshape(-1, self.filters)
        
        flat_b = self.bias.reshape(1, -1)
        # Dot : ( N x YH x YW , YC ) + ( 1 x 1 x 1, YC)
        flat_y = np.dot(flat_x, flat_w) + flat_b
        # Reshape : N x YH x YW x YC
        y = flat_y.reshape(*self.output_shape)
        
        if self.training:
            self.flat_x = conv_utils.average_flat_x(flat_x, y.shape)
            self.flat_w = flat_w
        return y


    def backprop(self, dLdy, optimizer):
        # dLdy : YH x YW x YC
        # dydw : ( YH x YW x YC ) x ( KH x KW x XC x YC )
        # dLdw : KH x KW x XC x YC
        YH, YW, YC = dLdy.shape
        KH, KW, XC, YC =  self.weight.shape
        
        # Reshape : ( YH x YW , YC )
        flat_dLdy = dLdy.reshape(-1, YC)
        
        # flat_x.T : ( XC x KH x KW , YH x YW )
        # flat_dLdy : ( YH x YW , YC )
        # Dot : ( XC x KH x KW , YC )
        # Reshape : XC x KH x KW x YC
        # Transpose : KH x KW x XC x YC

        dLdw = np.dot(self.flat_x.T, flat_dLdy).reshape(XC, KH, KW, YC).transpose((1, 2, 0, 3))

        kernel_regularize_term = self.kernel_regularizer(self.weight) if self.kernel_regularizer is not None else 0
        kernel_delta = dLdw + kernel_regularize_term

        self.weight = self.weight - optimizer.learning_rate * kernel_delta

        
        if self.use_bias:
            bias_regularize_term = self.bias_regularizer(self.bias) if self.bias_regularizer is not None else 0
            bias_delta = dLdy + bias_regularize_term
            self.bias = self.bias - optimizer.learning_rate * bias_delta

        # flat_dLdy : ( YH x YW , YC )
        # flat_w.T : ( YC , XC x KH x KW )
        # Dot : ( YH x YW , XC x KH x KW )
        flat_dLdx = np.dot(flat_dLdy ,self.flat_w.T)
        dLdx = conv_utils.col2img(flat_dLdx, self.input_shape, self.output_shape, self.kernel_size, self.strides, self.padding_size)

        return dLdx