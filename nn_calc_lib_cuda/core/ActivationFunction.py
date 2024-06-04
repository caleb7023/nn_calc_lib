#!/user/bin/env

# author : caleb7023

import cupy as cp

__all__ = [
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

def ident       (x: cp.ndarray) -> cp.ndarray: return x
def sigmoid     (x: cp.ndarray) -> cp.ndarray: return 1 / (1 + cp.exp(-x))
def hard_sigmoid(x: cp.ndarray) -> cp.ndarray: return cp.maximum(0, cp.minimum(1, 0.2 * x + 0.5))
def log_sigmoid (x: cp.ndarray) -> cp.ndarray: return 1 / (1 + cp.exp(-cp.log(x)))
def swish       (x: cp.ndarray) -> cp.ndarray: return x * sigmoid(x)
def mish        (x: cp.ndarray) -> cp.ndarray: return x * cp.tanh(cp.log(1 + cp.exp(x)))
def hard_swish  (x: cp.ndarray) -> cp.ndarray: return x * hard_sigmoid(x) # link: https://arxiv.org/pdf/1905.02244v5
def relu        (x: cp.ndarray) -> cp.ndarray: return cp.maximum(0, x)
def relu6       (x: cp.ndarray) -> cp.ndarray: return cp.minimum(6, cp.maximum(0, x))
def leaky_relu  (x: cp.ndarray) -> cp.ndarray: return cp.maximum(0.01 * x, x)
def elu         (x: cp.ndarray) -> cp.ndarray: return x if cp.all(x >= 0) else cp.exp(x) - 1 # link: https://arxiv.org/pdf/1511.07289v5
def tanh        (x: cp.ndarray) -> cp.ndarray: return cp.tanh(x)
def tanh_shrink (x: cp.ndarray) -> cp.ndarray: return x - cp.tanh(x)
def tanh_exp    (x: cp.ndarray) -> cp.ndarray: return cp.tanh(cp.exp(x)) # link: https://arxiv.org/pdf/2003.09855v1
def hard_tanh   (x: cp.ndarray) -> cp.ndarray: return cp.maximum(-1, cp.minimum(1, x))
def bent_ident  (x: cp.ndarray) -> cp.ndarray: return ((cp.sqrt(x ** 2 + 1) - 1) / 2) + x
def hard_shrink (x: cp.ndarray) -> cp.ndarray: return cp.where((-0.5 < x) & (x < 0.5), 0, x)
def soft_shrink (x: cp.ndarray) -> cp.ndarray: return cp.where((0.5 < x), x - 0.5, x + 0.5)

class derivative:

    # x is the input
    # y is the output of the normal activation function

    def ident(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.ones_like(x)

    def sigmoid(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return y * (1 - y)
    
    def hard_sigmoid(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where((-2.5 < x) & (x < 2.5), 0.2, 0)
    
    def log_sigmoid(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return 1 / (1 + cp.exp(-x))
    
    def swish(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return y + (1 - y) / (1 + cp.exp(-x))
    
    def mish(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return 2 * cp.exp(x) + cp.exp(2 * x) + 2
    
    def hard_swish(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where(x <= -3, 0, cp.where(x <= 3, 0.33333333333 * x + 1, 1))
    
    def relu(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where(x < 0, 0, 1)
    
    def relu6(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where((x < 0) | (6 < x), 0, 1)
    
    def leaky_relu(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where(x <= 0, 0.01, 1)
    
    def elu(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.exp(x) if cp.all(x < 0) else cp.ones_like(x)

    def tanh(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return 1 - y ** 2

    def tanh_shrink(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return y ** 2

    def tanh_exp(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        exp_x = cp.exp(x)
        tanh_exp_x = cp.tanh(exp_x)
        return tanh_exp_x - (tanh_exp_x ** 2 - 1) * x * exp_x

    def hard_tanh(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where((-1 <= x) & (x <= 1), 0, 1)

    def bent_ident(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return x / (2 * cp.sqrt(x ** 2 + 1)) + 1

    def hard_shrink(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where((-0.5 < x) & (x < 0.5), 0, 1)

    def soft_shrink(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        return cp.where((-0.5 < x) & (x < 0.5), 0, 1)