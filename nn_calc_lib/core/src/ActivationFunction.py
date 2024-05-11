#!/user/bin/env

# author : caleb7023

import numpy as np

__all__ = [
    'step',
    'ident',
    'sigmoid',
    'hard_sigmoid',
    'log_sigmoid',
    'swish',
    'mish',
    'hard_swish',
    'relu',
    'relu6',
    'leaky_relu',
    'elu',
    'celu',
    'tanh',
    'tanh_shrink',
    'tanh_exp',
    'hard_tanh',
    'bent_ident',
    'hard_shrink',
    'soft_shrink',
    'threshold'
]

def step        (x:float)->float:return 1 if x >= 0 else 0
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
def elu         (x:float)->float:return x if x > 0 else 0.01*(np.exp(x)-1) # link: https://arxiv.org/pdf/1511.07289v5
def tanh        (x:float)->float:return np.tanh(x)
def tanh_shrink (x:float)->float:return x-np.tanh(x)
def tanh_exp    (x:float)->float:return np.tanh(np.exp(x)) # link: https://arxiv.org/pdf/2003.09855v1
def hard_tanh   (x:float)->float:return max(-1, min(1, x))
def bent_ident  (x:float)->float:return ((np.sqrt(x**2+1)-1)/2)+x
def hard_shrink (x:float)->float:return x if abs(x) > 0.5 else 0
def soft_shrink (x:float)->float:return x-0.5 if x > 0.5 else x+0.5
def threshold   (x:float)->float:return 1 if x > 0 else 0