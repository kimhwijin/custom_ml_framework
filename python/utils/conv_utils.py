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

    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding_size

    N, XH, XW, XC = x.shape
    N, YH, YW, YC = y_shape
    
    # N, XH + 2PH, XW + 2PW, XC
    padded_x = np.pad(x, [(0, 0), (PH, PH), (PW, PW), (0, 0)], 'constant')

    # N, KH, KW, YH, YW, XC
    col = np.zeros((N, KH, KW, YH, YW, XC))

    for h in range(KH):
        h_max = h + SH * YH
        for w in range(KW):
            w_max = w + SW * YW
            col[:, h, w, :, :, :] = padded_x[:, h:h_max:SH, w:w_max:SW, :]
    
    # Transpose : N , YH , YW , XC , KH, KW
    # Reshape : N x YH x YW , XC x KH x KW
    col = col.transpose((0,3,4,5,1,2)).reshape(N*YH*YW, -1)
    return col

def average_flat_x(flat_x, y_shape):
    n, yh, yw, _ = y_shape
    return np.mean(flat_x.reshape(n, yh*yw, -1), axis=0)
    

def col2img(flat_dLdx, x_shape, y_shape, kernel_size, stride, padding_size):
    # ( YH x YW , XC x KH x KW )
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding_size

    _, XH, XW, XC = x_shape
    _, YH, YW, YC = y_shape
    # XC x KH x KW x YH x YW
    flat_dLdx = flat_dLdx.reshape(YH, YW, XC, KH, KW).transpose((2, 3, 4, 0, 1))

    img = np.zeros((XC, XH + 2 * PH + SH - 1, XW + 2 * PW + SW - 1))
    for y in range(KH):
        y_max = y + SH * YH
        for x in range(KW):
            x_max = x + SW * YW
            img[:, y:y_max:SH, x:x_max:SW] += flat_dLdx[:, y, x, :, :]
    
    return img[:, PH:XH + PH, PH:XW + PH]

