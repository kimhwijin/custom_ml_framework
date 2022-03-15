import numpy as np


def binary_crossentropy(y_true, y_pred, axis=-1):
    epsilon = 1e-7
    return -(y_true * np.log(y_pred + epsilon) + (1-y_true)*np.log(1-y_pred+epsilon))

def binary_crossentropy_diff(y_true, y_pred, upstream_gradient=1, axis=-1):
    epsilon = 1e-7
    return -(y_true / (y_pred+epsilon) - (1-y_true) / (1-y_pred + epsilon)) * upstream_gradient

def categorical_crossentropy(y_true, y_pred, axis=-1, batch_axis=0):
    epsilon = 1e-7
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=axis), axis=batch_axis)

def softmax_categorical_crossentropy_diff(y_true, y_pred):
    return y_pred - y_true