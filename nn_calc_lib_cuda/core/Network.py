#!/user/bin/env

# author : caleb7023

import cupy as cp

import pickle as pk

from . import NetworkLayer

class Network:

    def __init__(self, layers:list[NetworkLayer.Input|NetworkLayer.Activation]|str=None)->None:
        """

        ### Parameters

        network : List of layer or string(path) objects. For custom neural network.
        propatiy: The propatiy of the new neural network when the network is not provided. List or path as a string.
        
        Either network or propatiy must be provided.

        ### Layers

        List of layer types.
        The first layer must be an input layer.
        The classes below can be set.

        ```python
        nclc.NetworkLayer.Input # Input layer
        nclc.NetworkLayer.Activation # Activation layer
        ```

        Or you can input a path as a string. (In development)

        """

        # Check for errors in the parameters
        if layers[0].__class__ != NetworkLayer.Input:
            raise ValueError("The first layer must be an input layer")

        self.layers = layers
        self.value = None
        self.losses = None

    def forward(self, input_data:cp.ndarray)->cp.ndarray:
        if input_data.ndim != 1:
            raise ValueError("input_data must be a 1D array")
        self.layers[0].value = input_data.astype(float)
        for i, layer in enumerate(self.layers[1:]):
            layer.forward(self.layers[i].value)
        self.value = self.layers[-1].value

    def backward(self, target_value:cp.ndarray, learning_rate:float=0.01)->None:
        # Check for errors
        if self.value is None:
            raise ValueError("forward_propagation must be called before calling backword_propagation")
        if target_value.ndim != 1:
            raise ValueError("target_value must be a 1D array")
        if len(target_value) != len(self.value):
            raise ValueError("target_value must have the same length as the size of the neural network layer")

        # Calculate the loss of the output layer
        losses = 2 * (self.value-target_value)

        if cp.all(losses == 0):
            self.losses = cp.zeros_like(self.layers[0].value)
            return

        # Backward propagation
        for layer in self.layers[1:][::-1]:
            losses = layer.backward(losses, learning_rate)

    
    def update(self)->None:
        for layer in self.layers[1:]:
            layer.update()
    
    def save(self, path:str)->None:
        with open(path, "wb") as file:
            pk.dump(self, file)



def __Load_file(path:str)->Network:
    pass

def Read(propatiy:list)->Network:
    pass