"""
A neural net is just a collection of layers
It behaves a lot like Layers.
"""
from typing import Sequence

from bloo.tensor import Tensor
from bloo.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
