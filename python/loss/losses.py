import numpy as np


def binary_crossentropy(y_true, y_pred, axis=-1):
    epsilon = 1e-7
    return -(y_true * np.log(y_pred + epsilon) + (1-y_true)*np.log(1-y_pred+epsilon))

def categorical_crossentropy(y_true, y_pred, axis=-1):
    epsilon = 1e-7
    return -np.mean(y_true * np.log(y_pred + epsilon), axis=axis, keepdims=False)

