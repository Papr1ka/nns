from .base import Nums
from .base import np


class OptimizerBase:
    def init_params(self, weights: Nums, bias: Nums):
        raise NotImplemented("init_params not implemented")

    def step(
        self, weights: Nums, bias: Nums, learning_rate: float, de_dW: Nums, de_dB: Nums
    ) -> None:
        raise NotImplemented("step not implemented")


class NoOptimizer(OptimizerBase):
    # Просто градиентный спуск
    def __init__(self, *args) -> None:
        pass

    def init_params(self, *args):
        pass

    def step(self, weights, bias, learning_rate: float, de_dW, de_dB) -> None:
        weights -= learning_rate * de_dW
        bias -= learning_rate * de_dB


class ImpulseOptimizer(OptimizerBase):
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def init_params(self, weights: Nums, bias: Nums):
        self.v_dW = np.zeros_like(weights)
        self.v_dB = np.zeros_like(bias)

    def step(
        self, weights: Nums, bias: Nums, learning_rate: float, de_dW: Nums, de_dB: Nums
    ) -> None:
        self.v_dW = self.beta * self.v_dW + (1 - self.beta) * de_dW
        self.v_dB = self.beta * self.v_dB + (1 - self.beta) * de_dB
        weights -= learning_rate * self.v_dW
        bias -= learning_rate * self.v_dB


class RMSPropOptimizer(OptimizerBase):
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def init_params(self, weights: Nums, bias: Nums):
        self.v_dW = np.zeros_like(weights)
        self.v_dB = np.zeros_like(bias)

    def step(
        self, weights: Nums, bias: Nums, learning_rate: float, de_dW: Nums, de_dB: Nums
    ) -> None:
        self.v_dW = self.beta * self.v_dW + (1 - self.beta) * np.square(de_dW)
        self.v_dB = self.beta * self.v_dB + (1 - self.beta) * np.square(de_dB)
        weights -= learning_rate * de_dW / (np.sqrt(self.v_dW) + 1e-4)
        bias -= learning_rate * de_dB / (np.sqrt(self.v_dB) + 1e-4)


class AdamOptimizer(OptimizerBase):
    def __init__(self, beta1: float, beta2: float) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1

    def init_params(self, weights: Nums, bias: Nums):
        self.v_dW = np.zeros_like(weights)
        self.v_dB = np.zeros_like(bias)

        self.s_dW = np.zeros_like(weights)
        self.s_dB = np.zeros_like(bias)

    def step(
        self, weights: Nums, bias: Nums, learning_rate: float, de_dW: Nums, de_dB: Nums
    ) -> None:
        self.v_dW = self.beta1 * self.v_dW + (1 - self.beta1) * de_dW
        self.v_dB = self.beta1 * self.v_dB + (1 - self.beta1) * de_dB

        v_dW_corrected = self.v_dW / (1 - np.power(self.beta1, self.t))
        v_dB_corrected = self.v_dB / (1 - np.power(self.beta1, self.t))

        self.s_dW = self.beta2 * self.s_dW + (1 - self.beta2) * np.square(de_dW)
        self.s_dB = self.beta2 * self.s_dB + (1 - self.beta2) * np.square(de_dB)

        s_dW_corrected = self.s_dW / (1 - np.power(self.beta2, self.t))
        s_dB_corrected = self.s_dB / (1 - np.power(self.beta2, self.t))

        weights -= learning_rate * v_dW_corrected / np.sqrt(s_dW_corrected + 1e-8)
        bias -= learning_rate * v_dB_corrected / np.sqrt(s_dB_corrected + 1e-8)
        self.t += 1
