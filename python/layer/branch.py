import numpy as np
from base_layer import *


class ParallelLayer(Layer):
    def __init__(self, *args):
        self.layers = args
    
    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)

    def forward(self, inputs):
        x = inputs
        branches = []
        for layer in self.layers[1:]:
            branches.append(layer.forward(inputs))

        y = branches[0]
        for branch in branches[1:]:
            y = np.c_[y, branch]

        return y
    
    def backprop(self, dLdy, optimizer):
        dLdx = self.layers[0].backprop(dLdy)
        for layer in self.layers[1:]:
            dLdx += layer.backprop(dLdy, optimizer)
        
        return dLdx

    
class ResNetLayer(ParallelLayer):
    def __init__(self, residual_block):
        skip_layer = IdentityLayer()
        super(ResNetLayer, self).__init__(skip_layer, residual_block)
