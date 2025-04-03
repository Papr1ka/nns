from abc import ABC
from typing import Dict, List, Tuple
from .base import Nums
from .base import np


"""
Оптимизаторы

Один оптимизатор обслуживает всю модели
"""


class OptimizerBase(ABC):
    def __init__(
        self,
        params: List[Tuple[Tuple[Nums, Nums], Tuple[Nums, Nums]]],
    ) -> None:
        """
        Args:
            params (List[Tuple[Tuple[Nums, Nums], Tuple[Nums, Nums]]]):
                Список кортежей вида (weights, bias) (de_dW, de_dB)
                для каждого ОБУЧАЕМОГО слоя
        """
        self.params = params

    def zero_grad(self):
        raise NotImplemented("optimizer zero_grad not implemented")

    def step_one(self, idx, weights, bias, de_dW, de_dB):
        """
        Выполняет шаг оптимизатора для слоя под индексом idx
        """
        raise NotImplemented("optimizer step not implemented")

    def step(self):
        """
        Выполняет шаг оптимизатора для всей модели
        """
        for idx, ((weights, bias), (de_dW, de_dB)) in enumerate(self.params):
            self.step_one(idx, weights, bias, de_dW, de_dB)
            de_dW.fill(0)
            de_dB.fill(0)


class SGD(OptimizerBase):
    """
    Стохастический градиентный спуск
    """

    def __init__(
        self,
        params: List[Tuple[Tuple[Nums, Nums], Tuple[Nums, Nums]]],
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.learning_rate = learning_rate

    def zero_grad(self):
        pass

    def step_one(self, idx, weights, bias, de_dW, de_dB) -> None:
        weights -= self.learning_rate * de_dW
        bias -= self.learning_rate * de_dB


class ImpulseOptimizer(OptimizerBase):
    def __init__(
        self,
        params: List[Tuple[Tuple[Nums, Nums], Tuple[Nums, Nums]]],
        beta: float,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta = beta

    def zero_grad(self):
        self.v_dWs = []
        self.v_dBs = []
        for (weights, bias), _ in self.params:
            v_dW = np.zeros_like(weights)
            v_dB = np.zeros_like(bias)
            self.v_dWs.append(v_dW)
            self.v_dBs.append(v_dB)

    def step_one(
        self,
        idx,
        weights: Nums,
        bias: Nums,
        de_dW: Nums,
        de_dB: Nums,
    ) -> None:
        self.v_dWs[idx] = self.beta * self.v_dWs[idx] + (1 - self.beta) * de_dW
        self.v_dBs[idx] = self.beta * self.v_dBs[idx] + (1 - self.beta) * de_dB
        weights -= self.learning_rate * self.v_dWs[idx]
        bias -= self.learning_rate * self.v_dBs[idx]


class RMSPropOptimizer(OptimizerBase):
    def __init__(
        self,
        params: List[Tuple[Tuple[Nums, Nums], Tuple[Nums, Nums]]],
        beta: float,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta = beta

    def zero_grad(self):
        self.v_dWs = []
        self.v_dBs = []
        for (weights, bias), _ in self.params:
            v_dW = np.zeros_like(weights)
            v_dB = np.zeros_like(bias)
            self.v_dWs.append(v_dW)
            self.v_dBs.append(v_dB)

    def step_one(
        self,
        idx,
        weights: Nums,
        bias: Nums,
        de_dW: Nums,
        de_dB: Nums,
    ) -> None:
        self.v_dWs[idx] = self.beta * self.v_dWs[idx] + (1 - self.beta) * np.square(
            de_dW
        )
        self.v_dBs[idx] = self.beta * self.v_dBs[idx] + (1 - self.beta) * np.square(
            de_dB
        )
        weights -= self.learning_rate * de_dW / (np.sqrt(self.v_dWs[idx]) + 1e-4)
        bias -= self.learning_rate * de_dB / (np.sqrt(self.v_dBs[idx]) + 1e-4)


class AdamOptimizer(OptimizerBase):
    def __init__(
        self,
        params: List[Tuple[Tuple[Nums, Nums], Tuple[Nums, Nums]]],
        beta1: float,
        beta2: float,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1

    def zero_grad(self):
        self.v_dWs = []
        self.v_dBs = []
        self.s_dWs = []
        self.s_dBs = []

        for (weights, bias), _ in self.params:
            v_dW = np.zeros_like(weights)
            v_dB = np.zeros_like(bias)

            s_dW = np.zeros_like(weights)
            s_dB = np.zeros_like(bias)

            self.v_dWs.append(v_dW)
            self.v_dBs.append(v_dB)

            self.s_dWs.append(s_dW)
            self.s_dBs.append(s_dB)

    def step_one(
        self,
        idx,
        weights: Nums,
        bias: Nums,
        de_dW: Nums,
        de_dB: Nums,
    ) -> None:
        self.v_dWs[idx] = self.beta1 * self.v_dWs[idx] + (1 - self.beta1) * de_dW
        self.v_dBs[idx] = self.beta1 * self.v_dBs[idx] + (1 - self.beta1) * de_dB

        v_dW_corrected = self.v_dWs[idx] / (1 - np.power(self.beta1, self.t))
        v_dB_corrected = self.v_dBs[idx] / (1 - np.power(self.beta1, self.t))

        self.s_dWs[idx] = self.beta2 * self.s_dWs[idx] + (1 - self.beta2) * np.square(
            de_dW
        )
        self.s_dBs[idx] = self.beta2 * self.s_dBs[idx] + (1 - self.beta2) * np.square(
            de_dB
        )

        s_dW_corrected = self.s_dWs[idx] / (1 - np.power(self.beta2, self.t))
        s_dB_corrected = self.s_dBs[idx] / (1 - np.power(self.beta2, self.t))

        weights -= self.learning_rate * v_dW_corrected / np.sqrt(s_dW_corrected + 1e-8)
        bias -= self.learning_rate * v_dB_corrected / np.sqrt(s_dB_corrected + 1e-8)
        self.t += 1
