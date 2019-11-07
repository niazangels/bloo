"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward and 
propagate gradients backwards.

For example a neural net may look like:
inputs -> Linear -> Tanh -> Linear -> Softmax
"""
import numpy as np
from typing import Dict, Callable
from bloo.tensor import Tensor

F = Callable[[Tensor], Tensor]


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


class Activation(Layer):
    """
    An activation function layer just applies a function
    elememt wise to its inputs
    """

    def __input__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return self.f(input)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(g(z))
        then dy/dz = f'(g(z)) * g'(z)
        """
        self.input = input
        return self.f_prime(self.input) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__()
