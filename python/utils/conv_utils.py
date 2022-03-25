import numpy as np
import math
#tensorflow 참고

def normalize_tuple(value, rank):
    if isinstance(value, int):
        value_tuple = (value, ) * rank
    elif isinstance(value, (tuple, list)):
        value_tuple = tuple(value)
        if len(value) != rank:
            raise ValueError("입력 인자의 형태가 올바르지 않습니다. value : {}, rank : {}".format(value, rank))
        else:
            for single_value in value_tuple:
                if not isinstance(single_value, int):
                    raise ValueError("입력 인자의 타입이 올바르지 않습니다. value : {}, single_value : {}".format(value_tuple, single_value))

    return value_tuple

def valid_check_padding(value):
    if not isinstance(value, str):
        raise ValueError("입력 인자 패딩 타입 올바르지 않습니다. value : {}, type : {}".format(value, type(value)))
    padding = value.lower()
    if padding not in ('valid', 'same'):
        raise ValueError("입력 패딩 종류가 valid, same, causal 중 하나가 아닙니다. value : {}".format(value))

    return padding


def compute_padding_size(padding, i, f, s):
    if padding == 'same':
        o = i // s
        p = math.ceil(s(o-1)+f - i) / 2
    else:
        p = 0
    return p

def compute_output_padding_size(padding, i, k, s):
    # O = (I - K + 2P) / S + 1
    
    if padding == 'same':
        o = math.ceil(i / s)
        p = math.ceil(((o - 1) * s + k - i) / 2)
    elif padding == 'valid':
        o = math.floor((i - k) / s + 1)
        p = 0

    return o, p


def img2col(x, y_shape, kernel_size, stride, padding_size):
    # x : N, XH, XW, XC
    # y : N, YH, YW, YC

    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding_size

    n, xh, xw, xc = x.shape
    n, yh, yw, yc = y_shape
    
    # N, XH + 2PH, XW + 2PW, XC
    padded_x = np.pad(x, [(0, 0), (ph, ph), (pw, pw), (0, 0)], 'constant')

    # N, KH, KW, YH, YW, XC
    col = np.zeros((n, kh, kw, yh, yw, xc))

    for h in range(kh):
        h_max = h + sh * yh
        for w in range(kw):
            w_max = w + sw * yw
            col[:, h, w, :, :, :] = padded_x[:, h:h_max:sh, w:w_max:sw, :]
    
    # Transpose : N , YH , YW , XC , KH, KW
    # Reshape : N x YH x YH , XC x KH x KW
    col = np.transpose(col, (0,3,4,5,1,2)).reshape(n*yh*yw, -1)
    return col

    

