from python.layer.base_layer import Layer
from python.activation import activations

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
                bias_constraint=None
                ):
    
        self.units = units if isinstance(units, int) else int(units)

        if self.units < 0:
            raise ValueError("")
        

        
        