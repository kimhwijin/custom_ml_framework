from python.layer.base_layer import Layer
import numpy as np
import warnings
from python.initializer import initializers
from python.regularizer import regularizers
'''
tensorflow keras 참고
'''

class Dense(Layer):
    def __init__(self,
                units,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                dtype=np.float32,
                training=True,
                ):
    
        self.units = units if isinstance(units, int) else int(units)

        if self.units <= 0:
            raise ValueError("units 이 0 이하 입니다. units : {}".format(self.units))
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # self.kernel_constraint = kernel_constraint
        # self.bias_constraint = bias_constraint
        self.dtype = dtype
        self.training = training

    def build(self, input_shape):
        
        output_dim = input_shape[-1]
        kernel_shape = (self.units, output_dim)
        self.weight = self.kernel_initializer(kernel_shape, dtype=self.dtype)
        
        if self.use_bias:
            bias_shape = (1, output_dim)
            self.bias = self.bias_initializer(bias_shape, dtype=self.dtype)
        
        self.built = True
    
    def forward(self, inputs):
        if inputs.dtype != self.dtype:
            warnings.warn("입력과, 커널의 데이터 타입이 다름. input dtype : {}, kernel dtype : {}".format(inputs.dtype, self.weight.dtype))
            inputs = inputs.astype(self.dtype)
        if self.training:
            self.inputs = inputs

        outputs = np.matmul(inputs, self.weight) + self.bias if self.use_bias else np.matmul(inputs, self.weight)
        return outputs
    

    def backprop(self, dLdy, learning_rate):
        #y = xw + b
        #dLdw = dydw * dLdy
        #dLdb = dydb * dLdy = dLdy

        kernel_regularize_term = self.kernel_regularizer(self.weight) if self.kernel_regularizer is not None else 0

        dLdw = np.matmul(self.inputs.T, dLdy)
        kernel_delta = dLdw + kernel_regularize_term
        self.weight = self.weight - learning_rate * kernel_delta

        if self.use_bias:
            bias_regularize_term = self.bias_regularizer(self.bias) if self.bias_regularizer is not None else 0
            bias_delta = dLdy + bias_regularize_term
            self.bias = self.bias - learning_rate * bias_delta