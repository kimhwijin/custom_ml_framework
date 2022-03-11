from tracemalloc import start
import numpy as np
import time

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

                


        