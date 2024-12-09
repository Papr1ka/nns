from typing import Tuple, Callable

from .base import Nums, NumsToNums
from .base import np
from .optimizers import OptimizerBase


class Layer:
    """
    Реализует линейный слой в многослойном перцептроне

    input_size: int - количество входных нейронов
    output_size: int - количество выходных нейронов
    weights: np.ndarray[np.number] - веса слоя
    bias: np.ndarray[np.number] - смещения

    self.activation_function: NumsToNums - функция активации слоя
    self.activation_function_derivation: NumsToNums - производная функции активации слоя
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: Tuple[
            NumsToNums,
            NumsToNums,
        ],
        weights_initialize_function: Callable[[Tuple[int, int]], Nums] = None,
    ) -> None:
        """
        Args:
            input_size (int): размерность входных данных слоя
            output_size (int): размерность выходных данных слоя
            activation_function (NumsToNums, NumsToNums): кортеж из функции активации и её производной
            weights_initialize_function (Callable[[Tuple[int, int]], Nums]): функция инициализации весов,
            принимает кортеж shape
        """
        self.input_size = input_size
        self.output_size = output_size
        if weights_initialize_function is None:
            weights_initialize_function = np.random.random

        self.weights = weights_initialize_function((input_size, output_size))
        self.bias = weights_initialize_function((1, output_size))
        self.activation_function = activation_function[0]
        self.activation_function_derivation = activation_function[1]

        # Значения x и t, фиксируемые при forward, нужны для вычисления ошибки
        self._xt = ()
        """
        Контейнеры для хранения вычисленных ошибок для весов и смещений
        Необходимы для того, чтобы можно было обучать батчами
        
        В батче, для каждого примера необходимо вызвать forward и backward.
        В конце батча, для модификации весов, необходимо вызвать update
        _errors_log будет очищен
        """
        self._errors_log = ([], [])

    def forward(self, x: Nums) -> Nums:
        # линейное преобразование
        t = x @ self.weights + self.bias
        # нелинейное преобразование
        h = self.activation_function(t)

        # Записываем значения, необходимые для вычисления ошибки
        self._xt = (x, t)
        return h

    def backward(self, error: Nums) -> Nums:
        """
        Обратное распространение ошибки
        Запоминает производные ошибки, но не изменяет параметры модели
        Для изменения нужно вызвать update
        """
        x, t = self._xt
        self._xt = ()

        de_dT = error * self.activation_function_derivation(t)
        de_dW = x.T @ de_dT
        de_dB = de_dT
        de_dX = de_dT @ self.weights.T
        self._errors_log[0].append(de_dW)
        self._errors_log[1].append(de_dB)
        return de_dX

    def update(self, learning_rate: float):
        """
        Модифицирует веса модели
        learning_rate: float, [0, 1] - на сколько сильно модель будет реагировать на ошибку
        """
        de_dWs, de_dBs = self._errors_log
        de_dW = np.sum(de_dWs, axis=0)
        de_dB = np.sum(de_dBs, axis=0)

        self.optimizer.step(self.weights, self.bias, learning_rate, de_dW, de_dB)

        self._errors_log = ([], [])

    def set_optimizer(self, optimizer: OptimizerBase):
        optimizer.init_params(self.weights, self.bias)
        self.optimizer = optimizer

    def __repr__(self) -> str:
        return f"<LinearLayer, {self.input_size}x{self.output_size}>"
