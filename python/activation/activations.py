import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derv(y):
    return np.sign(y)

def sigmoid(x):
    return np.exp(-relu(x)) / (1.0 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)

def tanh(x):
    return 2 * sigmoid(2*x) - 1

def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)

def softmax(x):
    max_element = np.max(x, axis=1)
    diff =(x.T - max_element).T
    exp = np.ex(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.T / sum_exp).T
    return probs

def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)

def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels

def linear(x):
    return x

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
    
    


