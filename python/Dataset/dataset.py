import numpy as np
from Activation.activation import *

class Dataset(object):
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
    
    def __str__(self):
        return '{}({}, {}+{}+{}'.format(self.name, self.mode, len(self.X_train), len(self.X_test), len(self.X_valid))
    
    @property
    def train_count(self):
        return len(self.X_train)
    
    def get_train_data(self, batch_size, nth):
        from_idx = nth * batch_size
        to_idx = (nth + 1) * batch_size

        return self.X_train[self.indices[from_idx:to_idx]], self.y_train[self.indices[from_idx:to_idx]]
    
    def shuffle_train_data(self, size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)
    
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_validate_data(self, count):
        self.valid_indices = np.arange(len(self.X_valid))
        np.random.shuffle(self.valid_indices)

        return self.X_valid[self.valid_indices[0:count]], self.y_valid[self.valid_indices[0:count]]
    
    def get_visualize_data(self, count):
        return self.get_validate_data(count)
    

    def shuffle_data(self, X, y, train_ratio=0.8, valid_ratio=0.05):
        data_count = len(X)

        train_count = int(data_count * train_ratio)
        valid_count = int(data_count * valid_ratio)
        test_count = data_count - (train_count + valid_count)

        train_from, train_to = 0, train_count
        valid_from, valid_to = train_count, train_count + valid_count
        test_from, test_to = train_count + valid_count, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)

        self.X_train = X[indices[train_from: train_to]]
        self.X_valid = X[indices[valid_from: valid_to]]
        self.X_test = X[indices[test_from: test_to]]

        self.input_shape = X.shape[1:]
        self.output_shape = y.shape[1:]

        return indices[train_from: train_to], indices[valid_from: valid_to], indices[test_from: test_to]

    
    def backprop_postproc(self, G_loss, aux, mode=None):
        if mode is None: mode = self.mode

        #mse loss
        if mode == 'regression':
            diff = aux
            shape = diff.shape

            g_loss_square = np.ones(shape) / np.prod(shape)
            g_square_diff = 2 * diff
            g_diff_output = 1

            G_square = g_loss_square * G_loss
            G_diff = g_square_diff * G_square
            G_output = g_diff_output * G_diff
            
        elif mode == 'binary':
            y, output = aux
            shape = output.shape

            g_loss_entropy = np.ones(shape) / np.prod(shape)
            g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

