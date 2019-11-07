"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward and 
propagate gradients backwards.

For example a neural net may look like:
inputs -> Linear -> Tanh -> Linear -> Softmax
"""
import numpy as numpy
from bloo.tensor import Tensor


class Layer:
    def __init__(self) -> None:
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
