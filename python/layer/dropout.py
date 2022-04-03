import numpy as np
from base_layer import Layer


class Dropout(Layer):
    def __init__(self,
                rate,
                training=True,
                seed=None,
                dtype=np.float32
                ):

        self.rate = rate
        self.seed = seed
        self.training = training

    def forward(self, inputs):
        x = inputs

        if not self.training:
            return x
        
        mask = np.random.binomial(1, self.rate, inputs.shape, dtype=self.dtype)
        y = x * mask / (1. - self.rate)
        self.mask = mask
        self.y = y

        return y

    def backprop(self, dLdy):
        dLdx = dLdy * self.mask / (1.-self.rate)
        return dLdx

        


