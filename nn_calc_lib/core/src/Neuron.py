#!/user/bin/env

# author : caleb7023

import numpy as np

from typing import Callable

from warnings import warn

import core.src.NeuralNetworkLayer as NeuralNetworkLayer

def __init__(self, neuron_bias:float=None, neuron_weights:np.ndarray=None, input_size:int=None, *, bias_random_range:tuple=None, weights_random_range:tuple=None, activation_function: Callable[..., float])->None:

    if neuron_weights is None:
        if input_size is None:
            raise ValueError("input_size must be provided if neuron_weights is not provided")
        elif input_size < 1:
            raise ValueError("input_size must be greater than 0")
        else:
            if weights_random_range is None:
                weights_random_range = (-1, 1)
            elif 2 < len(weights_random_range):
                self.__index_size_warning(len(weights_random_range))
            self._neuron_weights = np.random.uniform(weights_random_range[0], weights_random_range[1], input_size)
    else:
        self._neuron_weights = neuron_weights
    
    if not callable(activation_function):
        raise ValueError("activation_function must be a callable object")

    if neuron_bias is None:
        if bias_random_range is None:
            bias_random_range = (-1, 1)
        elif 2 < len(bias_random_range):
            self.__index_size_warning(len(bias_random_range))
        self._neuron_bias = np.random.uniform(bias_random_range[0], bias_random_range[1])
    else:
        self._neuron_bias = neuron_bias

    self._activation_function = activation_function
    self.value = None

def __index_size_warning(self, index:int)->None:
    warn(f"index size {index} is above of 2. above it was ignored.")

def forward(self, input_neural_network:NeuralNetworkLayer)->None:

    if input_neural_network.value.ndim != 1:
        raise ValueError("input_data must be a 1D array")
    if len(input_neural_network.value) != len(self._neuron_weights):
        raise ValueError("input_data must have the same length as neuron_weights")

    self = self._activation_function(np.sum(input_neural_network*self._neuron_weights) + self._neuron_bias)