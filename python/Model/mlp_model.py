import numpy as np
import time
from Activation.activation import *

class MlpModel(object):
    def __init__(self, name, dataset, hidden_configs):
        super(MlpModel, self).__init__(name, dataset)
        self.init_parameters(hidden_configs)

    def init_parameters(self, hidden_configs):
        self.hidden_configs = hidden_configs
        self.hidden_params = []

        prev_layer_input_shape = self.dataset.input_shape

        for hidden_config in hidden_configs:
            hidden_param, prev_layer_input_shape = self.alloc_layer_param(prev_layer_input_shape, hidden_config)
            self.hidden_params.append(hidden_param)
        
        output_dim = int(np.prod(self.dataset.output_shape))
        self.output_params, _ = self.alloc_layer_param(prev_layer_input_shape, output_dim)
    
    def alloc_layer_param(self, input_shape, hidden_config):
        input_dim = np.prod(input_shape)
        output_dim = hidden_config

        weight, bias = self.alloc_param_pair([input_dim, output_dim])

        return {'w': weight, 'b': bias}, output_dim
    
    def alloc_param_pair(self, shape):
        weight = np.random.normal(0, self.rand_std, shape)
        bias = np.zeros([shape[-1]])
        return weight, bias


    def model_train(self, epochs=10, batch_size=10, learning_rate=0.001, report=0):
        self.learning_rate = learning_rate

        steps_per_epoch = int(self.dataset.train_count / batch_size)
        train_start_time = start_epoch_time = int(time.time())
        if report != 0:
            print('Model {} train started: '.format(self.name))
        
        for epoch in range(epochs):
            costs = []
            accs = []
            self.dataset.shuffle_train_data(batch_size*steps_per_epoch)

            for n in range(steps_per_epoch):
                X_train, y_train = self.dataset.get_train_data(batch_size, n)
                cost, acc = self.train_step(X_train, y_train)
                costs.append(cost)
                accs.append(accs)

            if report > 0 and (epoch + 1) % report == 0:
                X_valid, y_valid = self.dataset.get_validate_data(100)
                acc = self.eval_accuracy(X_valid, y_valid)
                epoch_end_time = int(time.time())
                epoch_time = epoch_end_time - start_epoch_time
                total_time = epoch_end_time - train_start_time
                self.dataset.train_print_result(epoch+1, costs, accs, acc, epoch_time, total_time)
                start_epoch_time = epoch_end_time
            
        print('Model {} train ended in {} secs: '.format(self.name, total_time))


    def model_test(self):
        X_test, y_test = self.dataset.get_test_data()
        test_start_time = int(time.time())
        acc = self.eval_accuracy(X_test, y_test)
        test_end_time = int(time.time())
        self.dataset.test_print_result(self.name, acc, test_end_time-test_start_time)

    
    def model_visualize(self, num):
        print('Model {} Visualization'.format(self.name))
        X_de, y_de = self.dataset.get_visualize_data(num)
        est = self.get_estimate(X_de)
        self.dataset.visualize(X_de, est, y_de)

    def train_step(self, x, y):
        self.is_training = True

        output, aux_nn = self.forward_neuralnet(x)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(x, y, output)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        self.is_training = False

        return loss, accuracy

    def forward_neuralnet(self, x):
        hidden = x
        aux_layers = []

        for n, hidden_config in enumerate(self.hidden_configs):
            hidden, aux = self.forward_layer(hidden, hidden_config, self.hidden_params[n])
            aux_layers.append(aux)
        
        output, aux_out = self.forward_layer(hidden, None, self.output_params)

        return output, [aux_out, aux_layers]

    def backprop_neuralnet(self, G_output, aux):
        aux_out, aux_layers = aux

        G_hidden = self.backprop_layer(G_output, None, self.output_params, aux_out)

        for n in reversed(range(len(self.hidden_configs))):
            hidden_config, param, aux = self.hidden_configs[n], self.hidden_params[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hidden_config, param, aux)
        
        return G_hidden
    
    def forward_layer(self, x, hidden_config, param):
        y = np.matmul(x, param['w']) + param['b']
        if hidden_config is not None: y = relu(y)
        return y, [x, y]
    
    def backprop_layer(self, G_y, hidden_config, param, aux):
        x, y = aux
        if hidden_config is not None: G_y = relu_derv(y) * G_y

        g_y_weight = x.T
        g_y_x = param['w'].T

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_x = np.matmul(G_y, g_y_x)

        param['w'] -= self.learning_rate * G_weight
        param['b'] -= self.learning_rate * G_bias

        return G_x
    
    def forward_postproc(self, output, y):
        loss, aux_loss = self.dataset.forward_postproc(output, y)
        #for regularization
        extra, aux_extra = self.forward_extra_cost(y)
        return loss + extra, [aux_loss, aux_extra]
        

    def backprop_postproc(self, G_loss, aux):
        aux_loss, aux_extra = aux
        self.backprop_extra_cost(G_loss, aux_extra)
        G_output = self.dataset.backprop_postproc(G_loss, aux_loss)
        return G_output
    
    def eval_accuracy(self, x, y, output=None):
        if output is None:
            output, _ = self.forward_neuralnet(x)
        accuracy = self.dataset.eval_accuracy(x, y, output)
        return accuracy
    
    def get_estimate(self, x):
        output, _ = self.forward_neuralnet(x)
        estimate = self.dataset.get_estimate(output)
        return estimate
    



        