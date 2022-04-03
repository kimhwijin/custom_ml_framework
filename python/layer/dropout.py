import numpy as np
from base_layer import Layer


class Dropout(Layer):
    def __init__(self,
                rate,
                seed=None,
                training=True,
                dtype=np.float32
                ):

        self.rate = rate
        self.seed = seed
        self.training = training
        self.dtype = dtype

    def forward(self, inputs):
        x = inputs
        if not self.training:
            return x
        
        mask = np.random.binomial(1, self.rate, inputs.shape, dtype=self.dtype)
        y = x * mask / (1. - self.rate)
        self.mask = mask
        return y

    def backprop(self, dLdy):
        dLdx = dLdy * self.mask / (1.-self.rate)
        return dLdx

        
class MCDropout(Layer):
    def __init__(self,
                rate,
                n_iters,
                seed=None,
                training=True,
                dtype=np.float32):

        self.dropout = Dropout(rate, seed, training, dtype)
        self.n_iters = n_iters
    
    def forward(self, inputs):
        x = inputs

        if not self.training:
            x = np.expand_dims(x, axis=0)
            y = self.dropout(x)
            for _ in range(self.n_iters-1):
                y = np.r_[y, self.dropout(x)]
            return y

        y = self.dropout(x)

        return y
            
    def backprop(self, dLdy):
        dLdx = self.dropout.backprop(dLdy)
        return dLdx
        
    
