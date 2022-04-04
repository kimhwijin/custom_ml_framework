
class Layer():
    pass

class IdentityLayer(Layer):
    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        pass

    def forward(self, inputs):
        return inputs

    def backprop(self, dLdy, optimizer):
        return dLdy
