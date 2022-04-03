import math
import numpy as np

'''
pytorch 참고
'''

def glorot_uniform(shape, dtype, gain=1.):
    fan_in, fan_out = _calculate_fan_in_fan_out(shape)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return np.random.uniform(-a, a, shape).astype(dtype)

def zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype):
    return np.zeros(shape, dtype=dtype)

def _calculate_fan_in_fan_out(shape):
    #dense : [input_dim, output_dim]
    #conv 2d : [input_dim, output_dim, 3, 3]
    
    if len(shape) < 2:
        raise ValueError("데이터가 2차원 보다 작습니다. shape : {}".format(shape))
    
    input_dim = shape[0]
    output_dim = shape[1]
    receptive_field_size = 1
    if len(shape) > 2:
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = input_dim * receptive_field_size
    fan_out = output_dim * receptive_field_size

    return fan_in, fan_out

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