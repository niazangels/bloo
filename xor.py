"""
A simple problem that isn't linearly separable
"""
import numpy as np

from bloo.nn import NeuralNet
from bloo.train import train
from bloo.layers import Linear, Tanh

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

net = NeuralNet(
    [Linear(input_size=2, output_size=2), Tanh(), Linear(input_size=2, output_size=2)]
)
train(net, inputs, targets, epochs=5000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
