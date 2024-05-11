#!/user/bin/env

# author : caleb7023

import numpy as np

import core.src.NeuralNetworkLayer

def __init__(self, layers:list=None, propatiy:list=None)->None:
    """
    network : list of NeuralNetworkLayer objects. Default is None but must be provided if propatiy is not provided
    propatiy: list of NeuralNetworkLayer objects. Default is None but must be provided if network is not provided
    """
    if layers is None:
        if propatiy is None:
            raise ValueError("network must be provided if propatiy is not provided")
        self.layers = []
    else:
        if not layers[0].is_input_layer:
            raise ValueError("The first layer must be an input layer. Which can be created by creating a layer by setting is_input_layer to True")
        self.layers = layers

def forward_propagation(self, input_data:np.ndarray)->np.ndarray:
    self._layers[0].set_data(input_data)
    for layer in self._layers[1:]:
        layer.forward_propagation()
