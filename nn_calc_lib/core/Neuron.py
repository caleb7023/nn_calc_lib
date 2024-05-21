#!/user/bin/env

# author : caleb7023

import numpy as np

import inspect

from warnings import warn

class Neuron:
    def __init__(self, neuron_bias:float=None, neuron_weights:np.ndarray=None, input_size:int=None, *, bias_random_range:tuple=(-1, 1), weights_random_range:tuple=(-1, 1), activation_function, derivative_function)->None:

        if derivative_function is not None:
            if not callable(derivative_function):
                raise ValueError("derivative_function must be a callable object")
            else:
                arg_len = len(inspect.signature(derivative_function).parameters)
                if arg_len < 2:
                    raise ValueError("derivative_function must have 2 parameters.\nThe first one is going to be the input and the second one is going to be the output of the activation function.")
                if 2 < arg_len:
                    self.__index_size_warning(arg_len)

        if not callable(activation_function):
            raise ValueError("activation_function must be a callable object")

        if neuron_weights is None:
            if input_size is None:
                raise ValueError("input_size must be provided if neuron_weights is not provided")
            if input_size < 1:
                raise ValueError("input_size must be greater than 0")
            if 2 < len(weights_random_range):
                self.__index_size_warning(len(weights_random_range))
            self._neuron_weights = np.random.uniform(weights_random_range[0], weights_random_range[1], input_size)
        else:
            self._neuron_weights = neuron_weights

        if neuron_bias is None:
            if 2 < len(bias_random_range):
                self.__index_size_warning(len(bias_random_range))
            self._neuron_bias = np.random.uniform(bias_random_range[0], bias_random_range[1])
        else:
            self._neuron_bias = neuron_bias

        self._derivative_function = derivative_function
        self._activation_function = activation_function
        self.value = None



    def __index_size_warning(self, index:int)->None:
        warn(f"index size {index} is above of 2. above it was ignored.")



    def forward_propagation(self, input_neural_network_value:np.ndarray)->None:

        if input_neural_network_value.ndim != 1:
            raise ValueError("input_data must be a 1D array")
        
        if len(input_neural_network_value) != len(self._neuron_weights):
            raise ValueError(f"input_data must have the same length as neuron_weights. The input_data length is {len(input_neural_network_value)} and the neuron_weights length (the input data size that was set) is {len(self._neuron_weights)}.")
        
        self.last_input = input_neural_network_value
        self.sum_ = np.sum(input_neural_network_value*self._neuron_weights)
        self.value = self._activation_function(self.sum_ + self._neuron_bias)



    def backward_propagation(self, loss:float, learning_rate:float)->np.ndarray:
        next_layer_losses = np.zeros(len(self._neuron_weights), dtype=float)
        if self._derivative_function is None:
            derivative = __derivative(self.sum_ + self._neuron_bias, self.value, self._activation_function)
        else:
            derivative = self._derivative_function(self.sum_ + self._neuron_bias, self.value)
        next_layer_losses = loss*derivative*self._neuron_weights
        self._neuron_weights -= learning_rate*loss*derivative*self.last_input
        self._neuron_bias -= learning_rate*loss*derivative
        return next_layer_losses



def __derivative(x:float, y:float, activation_function)->float:
    MAX_ERROR = 0.0001
    error = 1
    activation_function_x = activation_function(x)
    derivative_result = ((activation_function(x+0.01)-activation_function_x)) * 100
    dt = 0.02
    while MAX_ERROR < error:
        last_derivative_result = derivative_result
        derivative_result = (activation_function(x+dt)-activation_function_x)/(dt)
        error = abs(derivative_result-last_derivative_result)
        dt /= 2