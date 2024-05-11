#!/user/bin/env

# author : caleb7023

import numpy as np

import core.src.NeuralNetworkLayer

def __init__(self, network:list=None, propatiy:list=None)->None:
    """
    network : list of NeuralNetworkLayer objects. Default is None but must be provided if propatiy is not provided
    propatiy: list of NeuralNetworkLayer objects. Default is None but must be provided if network is not provided
    """
    if network is None:
        if propatiy is None:
            raise ValueError("network must be provided if propatiy is not provided")
        self.network = []
    else:
        self.network = network

def forward_propagation(self, input_data:np.ndarray)->np.ndarray:
    self.layers = [input_data]
    for i in range(len(self.network)):
        self.layers.append(self.network[i].forward(self.layers[i]))
    return self.layers[-1]