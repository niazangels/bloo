"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward and 
propagate gradients backwards.

For example a neural net may look like:
inputs -> Linear -> Tanh -> Linear -> Softmax
"""
import numpy as np
from typing import Dict
from bloo.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        pass

    def forward(self, input: Tensor) -> Tensor:
        """
        Produce the outputs for this set of inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layers
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Computes output = inputs @ w + b
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return input @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/db = b * f'(x)
        and  dy/da = a * f'(x)
        and  dy/dc = f'(x)
        
        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/dc = f'(x)
        """
        # grads are (batch_size x output_size)
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.input.T @ grad
        return grad @ self.params["w"].T
