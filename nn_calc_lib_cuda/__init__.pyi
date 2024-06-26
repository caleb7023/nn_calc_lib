# author : caleb7023

import typing
import numpy as np

class NeuralNetwork:
    def __init__(self, layers:list=None, propatiy:list|str=None)->None:...
    def forward_propagation(self, input_data:np.ndarray)->np.ndarray:...
    def back_propagation(self, target:np.ndarray, learning_rate:float)->None:...
class ActivationFunction:
    def ident       (x:np.ndarray)->np.ndarray:...
    def sigmoid     (x:np.ndarray)->np.ndarray:...
    def hard_sigmoid(x:np.ndarray)->np.ndarray:...
    def log_sigmoid (x:np.ndarray)->np.ndarray:...
    def swish       (x:np.ndarray)->np.ndarray:...
    def mish        (x:np.ndarray)->np.ndarray:...
    def hard_swish  (x:np.ndarray)->np.ndarray:...
    def relu        (x:np.ndarray)->np.ndarray:...
    def relu6       (x:np.ndarray)->np.ndarray:...
    def leaky_relu  (x:np.ndarray)->np.ndarray:...
    def elu         (x:np.ndarray)->np.ndarray:...
    def tanh        (x:np.ndarray)->np.ndarray:...
    def tanh_shrink (x:np.ndarray)->np.ndarray:...
    def tanh_exp    (x:np.ndarray)->np.ndarray:...
    def hard_tanh   (x:np.ndarray)->np.ndarray:...
    def bent_ident  (x:np.ndarray)->np.ndarray:...
    def hard_shrink (x:np.ndarray)->np.ndarray:...
    def soft_shrink (x:np.ndarray)->np.ndarray:...
    class derivative:
        def ident       (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def sigmoid     (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def hard_sigmoid(x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def log_sigmoid (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def swish       (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def mish        (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def hard_swish  (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def relu        (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def relu6       (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def leaky_relu  (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def elu         (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def tanh        (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def tanh_shrink (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def tanh_exp    (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def hard_tanh   (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def bent_ident  (x:np.ndarray, y:np.ndarray)->np.ndarray:...
        def hard_shrink (x:np.ndarray, y:np.ndarray)->np.ndarray:...
