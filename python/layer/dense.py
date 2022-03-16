from python.layer.base_layer import Layer
import numpy as np
import warnings

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
                kernel_constraint=None,
                bias_constraint=None,
                dtype=np.float32
                ):
    
        self.units = units if isinstance(units, int) else int(units)

        if self.units < 0:
            raise ValueError("")
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.dtype = dtype

    def build(self, input_shape):
        
        output_dim = input_shape[-1]
        kernel_shape = (self.units, output_dim)

        self.weight = self.kernel_initializer(kernel_shape, dtype=self.dtype)
        
        if self.use_bias:
            self.bias = self.bias_initializer(output_dim, dtype=self.dtype)
        
        self.built = True
    
    def call(self, inputs):
        if inputs.dtype is not self.dtype:
            warnings.warn("입력과, 커널의 데이터 타입이 다름.")
            inputs = inputs.astype(self.dtype)
        
        outputs = np.matmul(inputs, self.weight) + self.bias if self.use_bias else np.matmul(inputs, self.weight)

        return outputs
    
        
        