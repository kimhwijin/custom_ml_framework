import numpy as np
from base_layer import Layer
from python.initializer import initializers
from python.regularizer import regularizers

class BatchNormalization(Layer):

    def __init__(self,
                # 일단 axis 는 무조건 -1 이라고 가정함
                axis=-1,
                momentum=0.99,
                epsilon=1e-3,
                beta_initializer='zeros',
                gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones',
                beta_regularizer=None,
                gamma_regularizer=None,
                training=True,
                dtype = np.float32
                ):
        self.axis= axis
        self.momentum = momentum
        self.epsilone = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.training = training
        self.dtype = dtype
        self.built = False

    def build(self, input_shape):
        if not isinstance(self.axis, int):
            raise TypeError("axis는 아직 int 만 사용할 수 있음")

        norm_size = input_shape[self.axis]
        self.norm_axis = input_shape[:self.axis]

        self.beta = self.beta_initializer(norm_size, self.dtype)
        self.gamma = self.gamma_initializer(norm_size, self.dtype)
        self.moving_mean = self.moving_mean_initializer(norm_size, self.dtype)
        self.moving_variance = self.moving_variance_initializer(norm_size, self.dtype)
        
        self.built = True

    def forward(self, inputs):
        x = inputs
        
        if self.training:
            _mean = np.mean(x, axis=self.norm_axis)
            _variance = np.var(x, axis=self.norm_axis)

            self.moving_mean = self.moving_mean * self.momentum + _mean * (1. - self.momentum)
            self.moving_variance = self.moving_variance * self.momentum + _variance * (1. - self.momentum)

        else:
            _mean = self.moving_mean
            _variance = self.moving_variance
        
        _std = np.sqrt(_variance + self.epsilone)
        x_hat = (x - _mean) / _std
        y = self.gamma * x_hat + self.beta

        if self.training:
            self.std = _std
            self.x_hat = x_hat
        return y
    
    def backprop(self, dLdy, optimizer):

        #--
        dLdGamma = np.sum(dLdy * self.x_hat, axis=self.norm_axis)

        gamma_regularize_term = self.gamma_regularizer(self.gamma) if self.gamma_regularizer is not None else 0
        dLdGamma += gamma_regularize_term
        self.gamma -= optimizer.learning_rate * dLdGamma

        #--
        dLdBeta = np.sum(dLdy, axis=self.norm_axis)

        beta_regularize_term = self.beta_regularizer(self.beta) if self.beta_regularizer is not None else 0
        dLdBeta += beta_regularize_term
        self.beta -= optimizer.learning_rate * dLdBeta
        #--
        dLdx = dLdy * self.gamma / self.std

        return dLdx
        
        

        


