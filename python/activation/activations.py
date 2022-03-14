from git import Object
from matplotlib.pyplot import sca
import numpy as np


class Activation(Object):
    def __init__(self, name, training):
        self.name = name
        self.fn = get(name)
        self.diff_fn = get(name + '_diff')
        self.training = training

    def forward(self, x, **kwargs):
        
        if self.training:
            output, self.diff_args = self.fn(x, training=self.training, **kwargs)
        else:
            output, _ = self.fn(x, training=self.training, **kwargs)

        self.fn_configs = kwargs

        return output
    
    def backprop(self):
        return self.diff_fn(**self.diff_args, self.fn_configs)



def softmax(x, axis=-1, training=True):
    if len(x.shape) > 1:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        s = np.sum(e, axis=axis, keepdims=True)
        output = e / s
    else:
        raise ValueError()
    return (output, {'x': None, 'y': output}) if training else (output, None)


def softmax_diff(x, y, axis=-1):
    return (1-y) * y


def elu(x, alpha=1.0, training=True):
    output = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return (output, {'x': x, 'y': None}) if training else (output, None)

def elu_diff(x, y, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def selu(x, training=True):
    alpha = 1.67326324
    scale = 1.05070098
    output = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return (output, {'x': x, 'y': None}) if training else (output, None)

def selu_diff(x, y):
    alpha = 1.67326324
    scale = 1.05070098
    return np.where(x > 0, scale, scale * alpha * np.exp(x))


def relu(x, training=True):
    output = np.maximum(x, 0)
    return (output, {'x': None, 'y': output}) if training else (output, None)

def relu_diff(x, y):
    return np.sign(y)

def leaky_relu(x, alpha=0.3, training=True):
    if alpha < 0:
        alpha = 0
    output = np.where(x > 0, x, alpha * x)
    return (output, {'x': x, 'y': None}) if training else (output, None)

def leaky_relu_diff(x, y, alpha=0.3):
    return np.where(x > 0, 1, alpha)

def sigmoid(x, training=True):
    output = 1 / (1 + np.exp(-x))
    return (output, {'x': None, 'y': output}) if training else (output, None)

def sigmoid_diff(x, y):
    return (1 - y) * y

def hard_sigmoid(x, training=True):
    # x < -2.5 : 0
    # -2.5 <= x <= 2.5 : 0.2 * x + 0.5
    # 2.5 < x : 1
    output = np.where(x > 2.5, 1, np.where(x < -2.5, 0, 0.2 * x + 0.5))
    return (output, {'x': x, 'y': output}) if training else (output, None)

def hard_sigmoid_diff(x, y):
    return np.where(x > 2.5 and x < -2.5, 0, 0.2)

def tanh(x, training=True):
    output = 2 / (1 + np.exp(-2 * x)) - 1
    return (output, {'x': None, 'y': output}) if training else (output, None)

def tanh_diff(x, y):
    return 1 - y ** 2

def exponential(x, training=True):
    output = np.exp(x)
    return (output, {'x': None, 'y': output}) if training else (output, None)

def exponential_diff(x, y):
    return y

def linear(x, training=True):
    return (x, {'x': None, 'y': None}) if training else (x, None)

def linear_diff(x, y):
    return 1

def parse_activation(identifier):
    return globals().get(identifier)

def get(identifier):
    if identifier is None:
        return linear
    elif isinstance(identifier, str):
        return parse_activation(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError("")

def relu_derv(output):
    y = output['output']
    return np.sign(y)

def elu_derv(output):
    x = output['x']
    y = output['output']
    np.where(x > 0, 1, np.exp())

def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)

def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)

def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)

def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels

    
    


