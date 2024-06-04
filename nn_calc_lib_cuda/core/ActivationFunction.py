#!/user/bin/env

# author : caleb7023

import cupy as cp

from typing import Callable
from dataclasses import dataclass

def tanh_exp_derivative(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    exp_x = cp.exp(x)
    tanh_exp_x = cp.tanh(exp_x)
    return tanh_exp_x - (tanh_exp_x ** 2 - 1) * x * exp_x

@dataclass
class ActivationFunction:
    """
    ### ActivationFunction
    - fxn: Callable[[cp.ndarray], cp.ndarray]
    - derivative: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]
    """
    fxn: Callable[[cp.ndarray], cp.ndarray]
    derivative: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]

ident = ActivationFunction(
    lambda x  : x,
    lambda x,y: cp.ones_like(x)
)

sigmoid = ActivationFunction(
    lambda x  : 1 / (1 + cp.exp(-x)),
    lambda x,y: y * (1 - y)
)

hard_sigmoid = ActivationFunction(
    lambda x  : cp.maximum(0, cp.minimum(1, 0.2 * x + 0.5)),
    lambda x,y: cp.where((-2.5 < x) & (x < 2.5), 0.2, 0)
)

log_sigmoid = ActivationFunction(
    lambda x  : 1 / (1 + cp.exp(-cp.log(x))),
    lambda x,y: 1 / (1 + cp.exp(-x))
)

swish = ActivationFunction(
    lambda x  : x * sigmoid.fxn(x),
    lambda x,y: y + (1 - y) / (1 + cp.exp(-x))
)

mish = ActivationFunction(
    lambda x  : x * cp.tanh(cp.log(1 + cp.exp(x))),
    lambda x,y: 2 * cp.exp(x) + cp.exp(2 * x) + 2
)

hard_swish = ActivationFunction(
    lambda x  : x * hard_sigmoid.fxn(x),
    lambda x,y: cp.where(x <= -3, 0, cp.where(x <= 3, 0.33333333333 * x + 1, 1))
)

relu = ActivationFunction(
    lambda x  : cp.maximum(0, x),
    lambda x,y: cp.where(x < 0, 0, 1)
)

relu6 = ActivationFunction(
    lambda x  : cp.minimum(6, cp.maximum(0, x)),
    lambda x,y: cp.where((x < 0) | (6 < x), 0, 1)
)

leaky_relu = ActivationFunction(
    lambda x  : cp.maximum(0.01 * x, x),
    lambda x,y: cp.where(x <= 0, 0.01, 1)
)

elu = ActivationFunction(
    lambda x  : x if cp.all(x >= 0) else cp.exp(x) - 1,
    lambda x,y: cp.exp(x) if cp.all(x < 0) else cp.ones_like(x)
)

tanh = ActivationFunction(
    lambda x  : cp.tanh(x),
    lambda x,y: 1 - y ** 2
)

tanh_shrink = ActivationFunction(
    lambda x  : x - cp.tanh(x),
    lambda x,y: y ** 2
)

tanh_exp = ActivationFunction(
    lambda x  : cp.tanh(cp.exp(x)),
    tanh_exp_derivative
)

hard_tanh = ActivationFunction(
    lambda x  : cp.maximum(-1, cp.minimum(1, x)),
    lambda x,y: cp.where((-1 <= x) & (x <= 1), 0, 1)
)

bent_ident = ActivationFunction(
    lambda x  : ((cp.sqrt(x ** 2 + 1) - 1) / 2) + x,
    lambda x,y: x / (2 * cp.sqrt(x ** 2 + 1)) + 1
)

hard_shrink = ActivationFunction(
    lambda x  : cp.where((-0.5 < x) & (x < 0.5), 0, x),
    lambda x,y: cp.where((-0.5 < x) & (x < 0.5), 0, 1)
)

soft_shrink = ActivationFunction(
    lambda x  : cp.where((0.5 < x), x - 0.5, x + 0.5),
    lambda x,y: cp.where((-0.5 < x) & (x < 0.5), 0, 1)
)
