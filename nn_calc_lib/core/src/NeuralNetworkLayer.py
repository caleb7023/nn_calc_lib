#!/user/bin/env

# author : caleb7023

import core.src.Neuron as Neuron

import numpy as np

def __init__(self, neurons:list=None, *, input_size:int=None, bias_random_range:tuple=None, weights_random_range:tuple=None, neuron_size:int=None,activation_function)->None:

    if neurons is None:
        if input_size is None:
            raise ValueError("input_size must be provided if neurons is not provided")
        elif input_size < 1:
            raise ValueError("input_size must be greater than 0")
        else:
            self._neurons = [Neuron(neuron_bias=None, neuron_weights=None, input_size=input_size, activation_function=activation_function) for i in range(neuron_size)]
        
    self._value = None

def __mul__(self, other:np.ndarray)->np.ndarray:
    if other.ndim != 1:
        raise ValueError("input_data must be a 1D array")
    if len(other) != len(self.value):
        raise ValueError("input_data must have the same length as the size of the neural network layer")
    return self._value * other