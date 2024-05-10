#!/user/bin/env

# author : caleb7023

import numpy as np

def __init__(self, network:list)->None:
    self.network = network

def forward_propagation(self, input_data:np.ndarray)->np.ndarray:
    self.layers = [input_data]
    for i in range(len(self.network)):
        self.layers.append(self.network[i].forward(self.layers[i]))
    return self.layers[-1]