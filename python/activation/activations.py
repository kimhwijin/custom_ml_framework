import numpy as np

def softmax(x, axis=-1):
    if len(x.shape) > 1:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        s = np.sum(e, axis=axis, keepdims=True)
        output = e / s
    else:
        raise ValueError()
    return {'output': output, 'x': x}

def elu(x, alpha=1.0):
    output = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return {'output': output, 'x': None}

def selu(x):
    alpha = 1.67326324
    scale = 1.05070098
    output = scale * elu(x, alpha)['output']
    return {'output': output, 'x': None}

def relu(x):
    output = np.maximum(x, 0)
    return {'output': output, 'x': None}

def softplus(x):
    output = np.log(np.exp(x) + 1)
    return {'output': output, 'x': x}

def softsign(x):
    output = x / (np.abs(x) + 1)
    return {'output': output, 'x': x}

def leaky_relu(x, alpha=0.3):
    if alpha < 0:
        alpha = 0
    output = np.where(x > 0, x, alpha * x)
    return {'output': output, 'x': None}

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return {'output': output, 'x': None}

def hard_sigmoid(x):
    # x < -2.5 : 0
    # -2.5 <= x <= 2.5 : 0.2 * x + 0.5
    # 2.5 < x : 1
    output = np.where(x > 2.5, 1, np.where(x < -2.5, 0, 0.2 * x + 0.5))
    return {'output': output, 'x': None}

def tanh(x):
    output = 2 * sigmoid(2*x)['output'] - 1
    return {'output': output, 'x': None}

def hard_tanh(x):
    output = 2 * hard_sigmoid(2 * x)['output'] - 1
    return {'output': output, 'x': None}

def exponential(x):
    output = np.exp(x)
    return {'output': output, 'x': None}

def linear(x):
    return {'output': x, 'x': None}

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


def relu_derv(y):
    return np.sign(y)

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

    
    


