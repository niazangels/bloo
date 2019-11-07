"""
Helper to train a NeuralNet
"""
from numpy import np
from bloo.tensor import Tensor
from bloo.nn import NeuralNet
from bloo.loss import Loss, SSE
from bloo.optimizer import Optimizer, SGD
from bloo.data import DataIterator, BatchIterator


def train(
    net: NeuralNet,
    inputs: Tensor,
    targets: Tensor,
    epochs: int = 1000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = SSE(),
    optimizer: Optimizer = SGD(),
) -> None:
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
