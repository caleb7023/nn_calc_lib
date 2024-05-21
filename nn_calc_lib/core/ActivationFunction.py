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

def ident       (x: np.ndarray) -> np.ndarray: return x
def sigmoid     (x: np.ndarray) -> np.ndarray: return 1 / (1 + np.exp(-x))
def hard_sigmoid(x: np.ndarray) -> np.ndarray: return np.maximum(0, np.minimum(1, 0.2 * x + 0.5))
def log_sigmoid (x: np.ndarray) -> np.ndarray: return 1 / (1 + np.exp(-np.log(x)))
def swish       (x: np.ndarray) -> np.ndarray: return x * sigmoid(x)
def mish        (x: np.ndarray) -> np.ndarray: return x * np.tanh(np.log(1 + np.exp(x)))
def hard_swish  (x: np.ndarray) -> np.ndarray: return x * hard_sigmoid(x) # link: https://arxiv.org/pdf/1905.02244v5
def relu        (x: np.ndarray) -> np.ndarray: return np.maximum(0, x)
def relu6       (x: np.ndarray) -> np.ndarray: return np.minimum(6, np.maximum(0, x))
def leaky_relu  (x: np.ndarray) -> np.ndarray: return np.maximum(0.01 * x, x)
def elu         (x: np.ndarray) -> np.ndarray: return x if np.all(x >= 0) else np.exp(x) - 1 # link: https://arxiv.org/pdf/1511.07289v5
def tanh        (x: np.ndarray) -> np.ndarray: return np.tanh(x)
def tanh_shrink (x: np.ndarray) -> np.ndarray: return x - np.tanh(x)
def tanh_exp    (x: np.ndarray) -> np.ndarray: return np.tanh(np.exp(x)) # link: https://arxiv.org/pdf/2003.09855v1
def hard_tanh   (x: np.ndarray) -> np.ndarray: return np.maximum(-1, np.minimum(1, x))
def bent_ident  (x: np.ndarray) -> np.ndarray: return ((np.sqrt(x ** 2 + 1) - 1) / 2) + x
def hard_shrink (x: np.ndarray) -> np.ndarray: return np.where((-0.5 < x) & (x < 0.5), 0, x)
def soft_shrink (x: np.ndarray) -> np.ndarray: return np.where((0.5 < x), x - 0.5, x + 0.5)

class derivative:

    # x is the input
    # y is the output of the normal activation function

    def ident(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y * (1 - y)
    
    def hard_sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where((-2.5 < x) & (x < 2.5), 0.2, 0)
    
    def log_sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def swish(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y + (1 - y) / (1 + np.exp(-x))
    
    def mish(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * np.exp(x) + np.exp(2 * x) + 2
    
    def hard_swish(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(x <= -3, 0, np.where(x <= 3, 0.33333333333 * x + 1, 1))
    
    def relu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0, 1)
    
    def relu6(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where((x < 0) | (6 < x), 0, 1)
    
    def leaky_relu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, 0.01, 1)
    
    def elu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(x) if np.all(x < 0) else np.ones_like(x)

    def tanh(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1 - y ** 2

    def tanh_shrink(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y ** 2

    def tanh_exp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x)
        tanh_exp_x = np.tanh(exp_x)
        return tanh_exp_x - (tanh_exp_x ** 2 - 1) * x * exp_x

    def hard_tanh(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where((-1 <= x) & (x <= 1), 0, 1)

    def bent_ident(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x / (2 * np.sqrt(x ** 2 + 1)) + 1

    def hard_shrink(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where((-0.5 < x) & (x < 0.5), 0, 1)

    def soft_shrink(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where((-0.5 < x) & (x < 0.5), 0, 1)