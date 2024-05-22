#!/user/bin/env

# author : caleb7023

import cupy as cp

import json as js

from .NeuralNetworkLayer import NeuralNetworkLayer

class NeuralNetwork:

    def __init__(self, layers:list[dict]=None, propatiy:list|str=None)->None:
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
                "derivative_function" : callable, # The derivative of the activation function. Will be ignored if neurons or activation_function is provided. Recommended to provide if it is a function not lerated from the activation function.
            },
            {
                # The same as the second dictionary.
            },
            # ... so on
        ]
        ```

        Or you can input a path as a string. (In development)

        """
        if layers is None:
            # If the propatiy is not provided and the network is not provided at the same time
            if propatiy is None:
                raise ValueError("propatiy must be provided if neural network is not provided")
            # If the propatiy is a path
            if propatiy.__class__ == str:
                # Load the propatiy from the file
                with open(propatiy) as file:
                    propatiy = js.load(file)

                # Check the propatiy for errors
                if propatiy.__class__ != list:
                    raise ValueError("propatiy must be a list")

                elif len(propatiy)<2:
                    raise ValueError("propatiy must have at least two layer")

                elif not propatiy[0]["is_input_layer"]:
                    raise ValueError("The first layer must be an input layer. Which can be created by creating a layer by setting is_input_layer to True")

                elif not all([layer.get("is_input_layer") in [None, False] for layer in propatiy[1:]]):
                    raise ValueError("The other layers must have is_input_layer set to None or False")

                elif not all([layer.__class__ == dict for layer in propatiy]):
                    raise ValueError("All layers must be a dictionary")

            # Create the neural network layers
            self.layers = [NeuralNetworkLayer(**{**layer, "input_size":propatiy[i-1]["neuron_size"]}) for i, layer in enumerate(propatiy)]

        else:
            if not layers[0].is_input_layer:
                raise ValueError("The first layer must be an input layer. Which can be created by creating a layer by setting is_input_layer to True")
            self.layers = layers
        self.value = self.layers[-1].value

    def forward_propagation(self, input_data:cp.ndarray)->cp.ndarray:
        if input_data.ndim != 1:
            raise ValueError("input_data must be a 1D array")
        data = input_data.astype(float)
        for layer in self.layers[1:]:
            layer.forward_propagation(data)
            data = layer.value
        self.value = self.layers[-1].value

    def backward_propagation(self, target_value:cp.ndarray, learning_rate:float=0.01)->None:
        # Check for errors
        if self.value is None:
            raise ValueError("forward_propagation must be called before calling backword_propagation")
        if target_value.ndim != 1:
            raise ValueError("target_value must be a 1D array")
        if len(target_value) != len(self.value):
            raise ValueError("target_value must have the same length as the size of the neural network layer")
        
        # Calculate the loss of the output layer
        losses = 2 * (self.value-target_value)

        # Backward propagation
        for layer in self.layers[1:][::-1]:
            losses = layer.backword_propagation(losses, learning_rate)