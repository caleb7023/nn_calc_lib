#!/user/bin/env

# author : caleb7023

import numpy as np

import json as js

import core.src.NeuralNetworkLayer as NeuralNetworkLayer

def __init__(self, layers:list=None, propatiy:list|str=None)->None:
    """

    ### Parameters

    network : List of NeuralNetworkLayer objects. For custom neural network.
    propatiy: The propatiy of the new neural network when the network is not provided. List or path as a string.
    
    Either network or propatiy must be provided.

    ### Propatiy

    ```python
    propatiy = [
        {
            "neuron_size"   : int|tuple|list, # The size of the input layer's neurons. Any dimension can be provided.
            "is_input_layer": True,           # This layer is a input layer so it must be True.
        },
        {
            "neurons"             : list,     # The list of the hidden layer's neurons.
            "neuron_size"         : int,      # The size of the hidden layer's neurons. Only single dimension can be provided. Will be ignored if neurons is provided.
            "activation_function" : callable, # The activation function of the hidden layer's neurons. Will be ignored if neurons is provided.
            "bias_random_range"   : tuple,    # The random range of the hidden layer's neurons bias. Will be ignored if neurons is provided. If not provided, (-1, 1) will be used.
            "weights_random_range": tuple,    # The random range of the hidden layer's neurons weights. Will be ignored if neurons is provided. If not provided, (-1, 1) will be used.
            "derivative_function" : callable, # The derivative of the activation function. Will be ignored if neurons or activation_function is provided. Recommended to provide.
        },
        {
            # The same as the second dictionary.
        },
        # ... so on
    ]
    ```

    Or you can input a path as a string.

    """
    if layers is None:
        if propatiy is None:
            raise ValueError("network must be provided if propatiy is not provided")
        if propatiy.__class__ == str:
            with open(propatiy) as file:
                propatiy = js.load(file)
        self.layers = [NeuralNetworkLayer(**layer) for layer in propatiy]
    else:
        if not layers[0].is_input_layer:
            raise ValueError("The first layer must be an input layer. Which can be created by creating a layer by setting is_input_layer to True")
        self.layers = layers
    self.value = self.layers[-1].value

def forward_propagation(self, input_data:np.ndarray)->np.ndarray:
    self._layers[0].set_data(input_data)
    for layer in self._layers[1:]:
        layer.forward_propagation()
    self.value = self.layers[-1].value

def backword_propagation(self, target_value:np.ndarray, learning_rate:float=0.01)->None:
    if self._value is None:
        raise ValueError("forward_propagation must be called before calling backword_propagation")
    if target_value.ndim != 1:
        raise ValueError("target_value must be a 1D array")
    if len(target_value) != len(self.value):
        raise ValueError("target_value must have the same length as the size of the neural network layer")
    output_layer_losses = 2 * (self.value-target_value)
    self.layers[-1].backword_propagation(output_layer_losses, learning_rate)