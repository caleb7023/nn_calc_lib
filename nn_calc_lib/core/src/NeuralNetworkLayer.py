#!/user/bin/env

# author : caleb7023

import core.src.Neuron as Neuron

import numpy as np

def __init__(self, neurons:list=None, *, input_size:int=None, bias_random_range:tuple=None, weights_random_range:tuple=None, neuron_size:int=None, activation_function=None, is_input_layer:bool=False)->None:

    if neurons is None:
        if input_size is None:
            raise ValueError("input_size must be provided if neurons is not provided")
        elif input_size < 1:
            raise ValueError("input_size must be greater than 0")
        else:
            self._neurons = [Neuron(neuron_bias=None,
                                    neuron_weights=None,
                                    input_size=input_size,
                                    bias_random_range=bias_random_range,
                                    weights_random_range=weights_random_range,
                                    activation_function=activation_function
            ) for i in range(neuron_size)]
            self._value = np.empty(neuron_size, dtype=float)
    else:
        self._neurons = neurons
        self._value = np.empty(len(neurons), dtype=float)

def __mul__(self, other:np.ndarray)->np.ndarray:
    if other.ndim != 1:
        raise ValueError("input_data must be a 1D array")
    if len(other) != len(self.value):
        raise ValueError("input_data must have the same length as the size of the neural network layer")
    return self._value * other