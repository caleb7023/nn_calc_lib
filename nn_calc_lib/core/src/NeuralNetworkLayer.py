#!/user/bin/env

# author : caleb7023

import core.src.Neuron as Neuron

import numpy as np

def __init__(self, neurons:list=None, *, input_size:int=None, bias_random_range:tuple=(-1, 1), weights_random_range:tuple=(-1, 1), neuron_size:int=None, activation_function=None, derivative_function=None, is_input_layer:bool=False)->None:
    """
    
    ### Parameters

    neurons             : List of Neuron objects. For custom neural network.
    input_size          : The size of the input layer's neurons. Any dimension can be provided if it is a input layer.
    bias_random_range   : The random range of the hidden layer's neurons bias. Will be ignored if neurons is provided. If not provided, (-1, 1) will be used.
    weights_random_range: The random range of the hidden layer's neurons weights. Will be ignored if neurons is provided. If not provided, (-1, 1) will be used.
    neuron_size         : The size of the hidden layer's neurons. Only single dimension can be provided. If is_input_layer is True, Any dimension can be allowed but it is gonna be converted into a single dimension. Will be ignored if neurons is provided.
    activation_function : The activation function of the hidden layer's neurons. Will be ignored if neurons is provided.
    derivative_function : The derivative of the activation function. Will be ignored if neurons or activation_function is provided. Recommended to provide.
    is_input_layer      : Is the layer for input or not. If this was set to True, parameters other than neuron_size will be ignored.
    
    """

    if is_input_layer:
        if neuron_size is None:
            raise ValueError("neuron_size must be provided if is_input_layer is True")
        self._is_input_layer = True
        self._value = np.empty(neuron_size.flatten(), dtype=float)
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
                                        activation_function=activation_function,
                                        derivative_function=derivative_function,
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

def backword_propagation(self, losses:np.ndarray, learning_rate:float=0.01)->np.ndarray:
    if self._is_input_layer:
        return None
    if self._value is None:
        raise ValueError("forward_propagation must be called before calling backword_propagation")
    if losses.ndim != 1:
        raise ValueError("losses must be a 1D array")
    if len(losses) != len(self.value):
        raise ValueError("losses must have the same length as the size of the neural network layer")
    next_layer_losses = np.zeros(len(self._neurons), dtype=float)
    for neuron, loss in zip(self._neurons, losses):
        next_layer_losses += neuron.backword_propagation(loss, learning_rate)
    return next_layer_losses