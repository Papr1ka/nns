from typing import Callable, Any
from math import ceil

from .base import Nums, NumsNumsToNums
from .base import np
from .layers import FunctionalNode, Learnable
from .optimizers import OptimizerBase


class NN:
    """
    Полносвязная многослойная нейронная сеть
    """

    def __init__(
        self,
        nodes: list[FunctionalNode],
        loss_function: NumsNumsToNums,
    ) -> None:
        self.nodes = nodes
        self.loss_function = loss_function  # Функция потерь

        # История значения loss_function, после train
        self.losses = []
        # Предсказания, полученные после train
        self.predicted = []

    def pop_loss(self, func: Callable[[Nums], Any]):
        loss = func(self.losses)
        self.losses.clear()
        return loss

    def forward(self, x: Nums) -> Nums:
        """
        Прямое распространение сети
        """
        current_x = x
        for layer in self.nodes:
            current_x = layer.forward(current_x)
        return current_x

    def backward(self, error: Nums) -> Nums:
        """
        Обратное распространение сети, не меняет веса
        """
        current_error = error
        # Для последнего слоя ошибку получаем в error_function
        for layer in self.nodes[::-1]:
            current_error = layer.backward(current_error)
        return current_error

    def get_learnables(self):
        params = []
        for node in self.nodes:
            if isinstance(node, Learnable):
                params.append(node.get_learnables())
        return params

    def initialize_weights(self, initialize_function):
        for node in self.nodes:
            if isinstance(node, Learnable):
                initialize_function(node)

    def train(
        self,
        xs: Nums,
        ys: Nums,
        optimizer: OptimizerBase,
        batch_size: int = 10,
    ):
        """
        Обучение сети в рамках одной эпохи
        """
        # Цикл по батчам, размера batch_size
        for batch_index in range(ceil(len(xs) / batch_size)):
            current_batch_slice = slice(
                batch_index * batch_size, (batch_index + 1) * batch_size
            )
            batch_xs = xs[current_batch_slice]
            batch_ys = ys[current_batch_slice]
            batch_losses = []  # Loss-ы одного батча

            for x, y in zip(batch_xs, batch_ys):
                predict = self.forward(x)
                error = predict - y

                # Считаем ошибку
                self.backward(error)

                E = self.loss_function(predict, y)
                self.predicted.append(predict)
                batch_losses.append(E)

            # Обновляем веса
            optimizer.step()

            # Собираем статистику
            batch_loss = np.sum(batch_losses)
            batch_losses.clear()
            self.losses.append(batch_loss)
