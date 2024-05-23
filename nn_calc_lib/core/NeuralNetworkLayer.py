#!/user/bin/env

# author : caleb7023

import numpy as np

class NeuralNetworkLayer:

    def __init__(self, neuron_weights:np.ndarray=None, *, neuron_bias:np.ndarray=None, input_size:int=None, bias_random_range:tuple=(-1, 1), weights_random_range:tuple=(-1, 1), neuron_size:int=None, activation_function=None, derivative_function=None, is_input_layer:bool=False)->None:
        """
        
        ### Parameters

        ```plaintext
        neuron_weights      : The weights of the hidden layer's neurons.
        neuron_bias         : The bias of the hidden layer's neurons. Must be provided if neuron_weights is provided.
        input_size          : The size of the input layer's neurons. Any dimension can be provided if it is a input layer.
        bias_random_range   : The random range of the hidden layer's neuron's bias. Will be ignored if neuron_weights and neuron_bias is provided. If not provided, (-1, 1) will be used.
        weights_random_range: The random range of the hidden layer's neuron's weights. Will be ignored if neurons is provided. If not provided, (-1, 1) will be used.
        neuron_size         : The size of the hidden layer's neurons. Only single dimension can be provided. Will be ignored if neuron_weights and neuron_bias is provided.
        activation_function : The activation function of the hidden layer's neurons. Will be ignored if neuron_weights and neuron_bias is provided.
        derivative_function : The derivative of the activation function. Recommended to provide.
        is_input_layer      : Is the layer for input or not. If this was set to True, parameters other than neuron_size will be ignored.
        ```

        """

        self.neurons_weights:np.ndarray # Shape: (input_size, neuron_size)
        self.neurons_biases :np.ndarray # Shape: (neuron_size)

        if is_input_layer:
            if neuron_size is None:
                raise ValueError("neuron_size must be provided if is_input_layer is True")
        else:
            if neuron_weights is None and neuron_bias is None:
                if input_size is None:
                    raise ValueError("input_size must be provided if neurons is not provided")
                elif input_size < 1:
                    raise ValueError("input_size must be greater than 0")
                # Create neurons for the layer
                self.neurons_weights = np.random.uniform(weights_random_range[0], weights_random_range[1], (input_size, neuron_size))
                self.neurons_biases  = np.random.uniform(bias_random_range   [0], bias_random_range   [1],              neuron_size )
            else:
                self.neurons_weights = neuron_weights
                self.neurons_biases  = neuron_bias
            self.activation_function = activation_function
            self.derivative_function = derivative_function
            self.neurons_weights_update = np.zeros_like(self.neurons_weights)
            self.neurons_biases_update  = np.zeros_like(self.neurons_biases)
        self.value = None
        self.is_input_layer = is_input_layer

    def forward_propagation(self, input_data:np.ndarray)->np.ndarray:
        if input_data.ndim != 1:
            raise ValueError("input_data must be a 1D array")
        # Forward propagation for the layer
        self.last_input = input_data
        self.actfunc_input = (self.neurons_weights * input_data[:, None]).sum(axis=0) + self.neurons_biases
        self.value = self.activation_function(self.actfunc_input)

    def backward_propagation(self, losses:np.ndarray, learning_rate:float=0.01)->np.ndarray:
        # Checking parameters for error
        if self.value is None:
            raise ValueError("forward_propagation must be called before calling backword_propagation")
        if losses.ndim != 1:
            raise ValueError("losses must be a 1D array")
        if len(losses) != len(self.value):
            raise ValueError("losses must have the same length as the size of the neural network layer")

        # Backward propagation
        if self.derivative_function is not None:
            derivative = self.derivative_function(self.actfunc_input, self.value)
        else:
            derivative = __derivative(self.actfunc_input, self.value, self.activation_function)

        gradient = derivative * losses
        gradient_update = gradient * learning_rate
        # Calculate the losses for the previous layer
        layer_losses = (gradient[None, :] * self.neurons_weights).sum(axis=1)
        # Update the weights and biases
        self.neurons_weights_update -= self.last_input[:, None] * gradient_update
        self.neurons_biases_update  -= gradient_update
        # Return the losses for the previous layer
        return layer_losses
    
    def update(self)->None:
        self.neurons_weights += self.neurons_weights_update
        self.neurons_biases  += self.neurons_biases_update
        self.neurons_weights_update = np.zeros_like(self.neurons_weights)
        self.neurons_biases_update  = np.zeros_like(self.neurons_biases)


def __derivative(x:np.ndarray, y:np.ndarray, activation_function)->np.ndarray:
    MAX_ERROR = 0.0001
    error = 1
    activation_function_x = activation_function(x)
    derivative_result = ((activation_function(x+0.01)-activation_function_x))*100
    dt = 0.005
    dt_recip = 200
    while MAX_ERROR < error:
        last_derivative_result = derivative_result
        derivative_result = (activation_function(x+dt)-activation_function_x)*dt_recip
        error = abs(derivative_result-last_derivative_result)
        dt_recip *= 2
        dt *= 0.5