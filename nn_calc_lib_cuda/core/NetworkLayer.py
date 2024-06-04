#!/user/bin/env

# author : caleb7023

import cupy as cp
from warnings import warn
from . import ActivationFunction as ac

__all__ = [
    "Input",
    "Activation",
]

class Input:
    def __init__(self, size)->None:
        self.size = size
        self.value = None



class Activation:

    def __init__(self, inputs:int, *, neurons:int, activation_function, derivative_function=None, bias_random_range:tuple[float]=(-1, 1), weights_random_range:tuple[float]=(-1, 1))->None:
        """
        
        ### Parameters

        ```plaintext
        >Required
         [int]         inputs              : The size of the input layer's neurons.
         [int]         neurons             : The size of the hidden layer's neurons.
         [callable]    activation_function : The activation function of the hidden layer's neurons.
        >Not Required
         [callable]    derivative_function : The derivative of the activation function. Recommended to provide if the function is original.
         [tuple[float]]bias_random_range   : The random range of the hidden layer's neuron's bias. If not provided, (-1, 1) will be used.
         [tuple[float]]weights_random_range: The random range of the hidden layer's neuron's weights. If not provided, (-1, 1) will be used.
        ```

        """

        self.neurons_weights:cp.ndarray # Shape: (inputs, neurons)
        self.neurons_biases :cp.ndarray # Shape: (neurons)

        # Check for errors in the parameters
        
        # Check parameter is the correct type
        if type(inputs)!=int:
            raise ValueError("inputs must be an integer")
        if type(neurons)!=int:
            raise ValueError("neurons must be an integer")
        if not callable(activation_function):
            raise ValueError("activation_function must be a callable")
        if derivative_function is not None and callable(derivative_function):
            raise ValueError("derivative_function must be a callable")
        if not isinstance(bias_random_range, (tuple, list)):
            raise ValueError("bias_random_range must be a tuple/list")
        if not isinstance(weights_random_range, (tuple, list)):
            raise ValueError("weights_random_range must be a tuple/list")

        # Check tuple/list parameter values are correct type
        if any([type(i)!=float for i in bias_random_range]):
            raise ValueError("bias_random_range must be a tuple/list of floats")
        if any([type(i)!=float for i in weights_random_range]):
            raise ValueError("weights_random_range must be a tuple/list of floats")
        
        # Check the parameter values are correct
        if inputs <= 0:
            raise ValueError("inputs must be greater than 0")
        if neurons <= 0:
            raise ValueError("neurons must be greater than 0")
        if len(bias_random_range) != 2:
            raise ValueError("bias_random_range must have 2 values")
        if len(weights_random_range) != 2:
            raise ValueError("weights_random_range must have 2 values")
        
        # Create neurons for the layer
        self.neurons_weights = cp.random.uniform(weights_random_range[0], weights_random_range[1], (inputs, neurons))
        self.neurons_biases  = cp.random.uniform(bias_random_range   [0], bias_random_range   [1],          neurons )

        self.activation_function = activation_function

        if derivative_function is None and activation_function.__name__ in ac.__all__:
            self.derivative_function = getattr(ac.derivative, activation_function.__name__)
        else:
            derivative_function = derivative_function

        self.neurons_weights_update = cp.zeros((inputs, neurons))
        self.neurons_biases_update  = cp.zeros(         neurons )

        self.value = None

    def forward(self, input_data:cp.ndarray)->cp.ndarray:
        # Forward propagation for the layer
        self.last_input = input_data
        self.actfunc_input = (self.neurons_weights * input_data[:, None]).sum(axis=0) + self.neurons_biases
        self.value = self.activation_function(self.actfunc_input)

    def backward(self, losses:cp.ndarray, learning_rate:float=0.01)->cp.ndarray:
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
        self.neurons_weights_update = cp.zeros_like(self.neurons_weights)
        self.neurons_biases_update  = cp.zeros_like(self.neurons_biases)



class __Convolution:

    def __init__(self, input_size:tuple[int], input_channels:int, channels:int, kernels_size:tuple[int], padding:int, stride:int, dilation:int, activation_function, derivative_function=None, kernel_random_range:tuple[float]=(-1, 1))->None:
        """
        
        ### Parameters

        ```plaintext
        >Required
         [tuple[int]]input_size          : The size of the input layer's neurons. Any dimension can be provided.
         [int]       input_channels      : The amount of channels in the input(previous) layer.
         [int]       channels            : The amount of channels in the layer.
         [tuple[int]]kernel_size         : The size of the kernel. Must be same lenght as input_size.
         [callable]  activation_function : The activation function of the hidden layer's neurons.
        >Not Required
         [callable]  derivative_function : The derivative of the activation function. Recommended to provide for better performance.
         [tuple[int]]kernel_random_range : The random range of the hidden layer's neuron's bias. If not provided, (-1, 1) will be used.
        ```

        """
        # Check for errors in the parameters

        # Check parameter is the correct type
        if type(input_channels)!=int:
            raise ValueError("input_channels must be an integer")
        if type(channels)!=int:
            raise ValueError("channels must be an integer")
        if type(padding)!=int:
            raise ValueError("padding must be an integer")
        if type(stride)!=int:
            raise ValueError("stride must be an integer")
        if type(dilation)!=int:
            raise ValueError("dilation must be an integer")
        if not isinstance(input_size, (tuple, list)):
            raise ValueError("input_size must be a tuple/list")
        if not isinstance(kernels_size, (tuple, list)):
            raise ValueError("kernel_size must be a tuple/list")
        if not isinstance(kernel_random_range, (tuple, list)):
            raise ValueError("kernel_random_range must be a tuple/list")
        if type(activation_function)!=callable:
            raise ValueError("activation_function must be a callable")
        if derivative_function is not None and type(derivative_function)!=callable:
            raise ValueError("derivative_function must be a callable")

        # Check tuple/list parameter values are correct type
        if any([type(i)!=int for i in input_size]):
            raise ValueError("input_size must be a tuple/list of integers")
        if any([type(i)!=int for i in kernels_size]):
            raise ValueError("kernel_size must be a tuple/list of integers")
        if any([type(i)!=float for i in kernel_random_range]):
            raise ValueError("kernel_random_range must be a tuple/list of floats")
        
        # Check the parameter values are correct
        if len(input_size) != len(kernels_size):
            raise ValueError("input_size must have the same length as the kernel_size")
        if len(kernel_random_range) != 2:
            raise ValueError("kernel_random_range must have 2 values")
        if input_channels <= 0:
            raise ValueError("input_channels must be greater than 0")
        if channels <= 0:
            raise ValueError("channels must be greater than 0")
        if not all([0 < i for i in input_size]):
            raise ValueError("input_size must have all positive values")
        if not all([0 < i for i in kernels_size]):
            raise ValueError("kernel_size must have all positive values")
        
        # Create kernels and biases for the layer
        self.kernels = cp.random.uniform(kernel_random_range[0], kernel_random_range[1], (channels,)+kernels_size+(input_channels,))
        self.biases  = cp.random.uniform(kernel_random_range[0], kernel_random_range[1],  channels                                 )

        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.kernel_update = cp.zeros_like(self.kernels)
        self.value = None

    def forward(self, input_data:cp.ndarray)->cp.ndarray:
        # Forward propagation for the layer
        self.last_input = input_data
        self.convolutions = cp.tensordot(self.kernels, input_data, axes=([2, 3, 4], [0, 1, 2])) + self.biases[:, None, None]
        self.value = self.activation_function(self.convolutions)

    def backward(self, losses:cp.ndarray, learning_rate:float=0.01)->cp.ndarray:
        pass

    def update(self)->None:
        self.kernels -= self.kernel_update
        self.kernel_update = cp.zeros_like(self.kernels)



class __Pooling:

    # Pooling types
    MaxPooling = 0
    AveragePooling = 1

    def __init__(self, pool_size:int, stride:int, type:int=0)->None:
        self.pool_size = pool_size
        self.stride = stride
        self.value = None
        self.type = type



def __derivative(x:cp.ndarray, y:cp.ndarray, activation_function)->cp.ndarray:
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
        