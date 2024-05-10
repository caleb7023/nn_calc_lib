#!/user/bin/env

# author : caleb7023

import numpy as np

def __init__(self, neuron_bias:float=None, neuron_weights:np.ndarray=None, input_size:int=None, *, bias_random_range:tuple=None, weights_random_range:tuple=None, activation_function)->None:

    if neuron_weights is None:
        if input_size is None:
            raise ValueError("input_size must be provided if neuron_weights is not provided")
        elif input_size < 1:
            raise ValueError("input_size must be greater than 0")
        else:
            if weights_random_range is None:
                weights_random_range = (-1, 1)
            self.neuron_weights = np.random.uniform(weights_random_range[0], weights_random_range[1], input_size)
    else:
        self.neuron_weights = neuron_weights

    if neuron_bias is None:
        self.neuron_bias = np.random.uniform(bias_random_range[0], bias_random_range[1])
    else:
        self.neuron_bias = neuron_bias

    self.activation_function = activation_function
    self.output_value        = None



def forward(self, input_data:np.ndarray)->None:

    if input_data.ndim != 1:
        raise ValueError("input_data must be a 1D array")
    if len(input_data) != len(self.neuron_weights):
        raise ValueError("input_data must have the same length as neuron_weights")

    self.output_value = self.activation_function(np.sum(input_data*self.neuron_weights) + self.neuron_bias)