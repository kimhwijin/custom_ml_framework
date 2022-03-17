import numpy as np

def l2(weight, alpha=0.01):
    return alpha * np.sqrt(np.sum(np.square(weight)))

def l2_diff(weight, alpha=0.01):
    return alpha * np.abs(weight)

def l1(weight, alpha=0.01):
    return alpha * np.sum(np.abs(weight))

def l1_diff(weight, alpha=0.01):
    return alpha


def parse_activation(identifier):
    return globals().get(identifier)

def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, str):
        return parse_activation(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError("")
        