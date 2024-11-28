from .base import Nums
from .base import np


def relu(x: Nums):
    return x * (x > 0)


def drelu(x: Nums):
    # производная функции relu
    return x > 0


def sigmoid(x: Nums):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: Nums):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x: Nums):
    e_x = np.exp(x)
    sm = np.sum(e_x)
    return e_x / sm


def cross_entropy(pred: Nums, expected: Nums):
    # z - предсказанное, y - ожидаемое
    return -np.sum(expected * np.log(pred))
