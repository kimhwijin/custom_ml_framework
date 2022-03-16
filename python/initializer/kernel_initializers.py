import math
import numpy as np

'''
pytorch 참고
'''

def glorot_uniform(kernel_shape, dtype, gain=1.):
    fan_in, fan_out = _calculate_fan_in_fan_out(kernel_shape)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return np.random.uniform(-a, a, kernel_shape, dtype)


def _calculate_fan_in_fan_out(kernel_shape):
    #dense : [input_dim, output_dim]
    #conv 2d : [input_dim, output_dim, 3, 3]
    
    if len(kernel_shape) < 2:
        raise ValueError("")
    
    input_dim = kernel_shape[0]
    output_dim = kernel_shape[1]
    receptive_field_size = 1
    if len(kernel_shape) > 2:
        for s in kernel_shape[2:]:
            receptive_field_size *= s
    fan_in = input_dim * receptive_field_size
    fan_out = output_dim * receptive_field_size

    return fan_in, fan_out