"""
You better know how to solve the FizzBuzz problem
"""
from typing import List

import numpy as np

from bloo.train import train
from bloo.nn import NeuralNet
from bloo.layers import Linear, Tanh
from bloo.optimizer import SGD


def fizz_buzz_encode(n: int) -> List[int]:
    if n % 15 == 0:
        return [0, 0, 0, 1]
    elif n % 3 == 0:
        return [0, 0, 1, 0]
    elif n % 5 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]


inputs = np.array([binary_encode(x) for x in range(101, 1023)])
targets = np.array([fizz_buzz_encode(x) for x in range(101, 1023)])

# Cheating by not using batches
net = NeuralNet([Linear(10, 50), Tanh(), Linear(50, 4)])
train(net, inputs, targets, epochs=5000, optimizer=SGD(0.001))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])
