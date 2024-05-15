#!/user/bin/env

# author : caleb7023

import numpy as np

__all__ = [
    "step",
    "ident",
    "sigmoid",
    "hard_sigmoid",
    "log_sigmoid",
    "swish",
    "mish",
    "hard_swish",
    "relu",
    "relu6",
    "leaky_relu",
    "elu",
    "tanh",
    "tanh_shrink",
    "tanh_exp",
    "hard_tanh",
    "bent_ident",
    "hard_shrink",
    "soft_shrink",
    "derivative"
]

def ident       (x:float)->float:return x
def sigmoid     (x:float)->float:return 1/(1+np.exp(-x))
def hard_sigmoid(x:float)->float:return max(0, min(1, 0.2*x+0.5))
def log_sigmoid (x:float)->float:return 1/(1+np.exp(-np.log(x)))
def swish       (x:float)->float:return x*sigmoid(x)
def mish        (x:float)->float:return x*np.tanh(np.log(1+np.exp(x)))
def hard_swish  (x:float)->float:return x*hard_sigmoid(x) # link: https://arxiv.org/pdf/1905.02244v5
def relu        (x:float)->float:return max(0, x)
def relu6       (x:float)->float:return min(6, max(0, x))
def leaky_relu  (x:float)->float:return max(0.01*x, x)
def elu         (x:float)->float:return x if 0<=x else np.exp(x)-1 # link: https://arxiv.org/pdf/1511.07289v5
def tanh        (x:float)->float:return np.tanh(x)
def tanh_shrink (x:float)->float:return x-np.tanh(x)
def tanh_exp    (x:float)->float:return np.tanh(np.exp(x)) # link: https://arxiv.org/pdf/2003.09855v1
def hard_tanh   (x:float)->float:return max(-1, min(1, x))
def bent_ident  (x:float)->float:return ((np.sqrt(x**2+1)-1)/2)+x
def hard_shrink (x:float)->float:return 0 if -0.5<x<0.5 else x
def soft_shrink (x:float)->float:return x-0.5 if 0.5<x else x+0.5

class derivative:

    # x is the input
    # y is the output of the normal activation function

    def ident(x:float, y:float)->float:
        return 1

    def sigmoid(x:float, y:float)->float:
        return y*(1-y)
    
    def hard_sigmoid(x:float, y:float)->float:
        return 0.2 if -2.5<x<2.5 else 0
    
    def log_sigmoid(x:float, y:float)->float:
        return 1/(1+np.exp(-x))
    
    def swish(x:float, y:float)->float:
        return y+(1-y)/(1+np.exp(-x))
    
    def mish(x:float, y:float)->float:
        return 2*np.exp(x)+np.exp(2*x)+2
    
    def hard_swish(x:float, y:float)->float:
        return 0 if x<=-3 else 0.33333333333*x+1 if x<=3 else 1
    
    def relu(x:float, y:float)->float:
        return 0 if x < 0 else 1
    
    def relu6(x:float, y:float)->float:
        return 0 if x<0 or 6<x else 1
    
    def leaky_relu(x:float, y:float)->float:
        return 0.01 if x <= 0 else 1
    
    def elu(x:float, y:float)->float:
        return np.exp(x) if x < 0 else 1
    
    def tanh(x:float, y:float)->float:
        return 1-y**2
    
    def tanh_shrink(x:float, y:float)->float:
        return y**2
    
    def tanh_exp(x:float, y:float)->float:
        exp_x = np.exp(x)
        tanh_exp_x = np.tanh(exp_x)
        return tanh_exp_x - (tanh_exp_x**2-1)*x*exp_x
    
    def hard_tanh(x:float, y:float)->float:
        return 0 if -1<=x<=1 else 1
    
    def bent_ident(x:float, y:float)->float:
        return x/(2*np.sqrt(x**2+1))+1
    
    def hard_shrink(x:float, y:float)->float:
        return 0 if -0.5<x<0.5 else 1
    
    def soft_shrink(x:float, y:float)->float:
        return 0 if -0.5<x<0.5 else 1
    
    def _auto(x:float, y:float, activation_function)->float:
        max_error = 0.0001
        error = 1
        activation_function_x = activation_function(x)
        derivative_result = ((activation_function(x+0.01)-activation_function_x)) * 100
        dt = 0.02
        while max_error < error:
            last_derivative_result = derivative_result
            derivative_result = (activation_function(x+dt)-activation_function_x)/(dt)
            error = abs(derivative_result-last_derivative_result)
            dt /= 2