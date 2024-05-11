#!/user/bin/env

# author : caleb7023

import core.src.Neuron as Neuron

import numpy as np

def __init__(self, neurons:list=None, *, input_size:int=None, bias_random_range:tuple=None, weights_random_range:tuple=None, neuron_size:int=None, activation_function=None, is_input_layer:bool=False)->None: # TODO: set the random_ranges to "(-1, 1) if not provided"

    if is_input_layer:
        self._is_input_layer = True
        self._value = None
        self.set_data = self.__set_data
    else:
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

def __set_data(self, input_data:np.ndarray)->None:
    if input_data.ndim != 1:
        self._value = input_data.flatten()
    else:
        self._value = input_data

def forward_propagation(self, input_data:np.ndarray)->np.ndarray:
    if input_data.ndim != 1:
        raise ValueError("input_data must be a 1D array")
    for neuron in self._neurons: # forward propagation for each neuron in the layer
        neuron.forward_propagation(input_data)
    self._value = np.array([neuron.value for neuron in self._neurons])