from abc import ABC
from typing import Any
from .base import np, Nums


class FunctionalNode(ABC):
    def __init__(self):
        self.cache: Any = None

    def forward(self, x: Nums):
        raise NotImplementedError("Method not implemented")

    def backward(self, error: Nums):
        raise NotImplementedError("Method not implemented")


class Relu(FunctionalNode):

    def forward(self, x: Nums):
        self.cache = x
        return x * (x > 0)

    def backward(self, error: Nums):
        return error * (self.cache > 0)


class Sigmoid(FunctionalNode):

    def forward(self, x: Nums):
        y = 1 / (1 + np.exp(-x))
        self.cache = y
        return y

    def backward(self, error: Nums):
        # todo: test
        return error * (self.cache * (1 - self.cache))


class Softmax(FunctionalNode):

    def forward(self, x: Nums):
        e_x = np.exp(x)
        sm = np.sum(e_x)
        return e_x / sm

    def backward(self, error: Nums):
        return error


class Loss(ABC):
    def forward(self, pred: Nums, expected: Nums):
        raise NotImplementedError("Method not implemented")

    def backward(self, pred: Nums, expected: Nums):
        raise NotImplementedError("Method not implemented")


def cross_entropy(pred: Nums, expected: Nums):
    return -np.sum(expected * np.log(pred))
