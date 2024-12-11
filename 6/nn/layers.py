from typing import Tuple, Callable

from .base import Nums, NumsToNums
from .base import np
from .optimizers import OptimizerBase
from .functional import FunctionalNode
from .utils import im2col, col2im

"""
Базовый класс для обучаемых слоёв, из предположения, что у всех будут веса и смещения
"""


def assert_n_dimentions(x: np.ndarray, n: int):
    assert np.ndim(x) == n, f"Ожидалась размерность {n}, получена {np.ndim(x)}"


class Learnable(FunctionalNode):
    def __init__(self, weights: Nums, bias: Nums):
        super().__init__()

        self.weights = weights
        self.bias = bias
        self.grad_W = np.zeros_like(weights)
        self.grad_B = np.zeros_like(bias)

    def get_learnables(self):
        return (self.weights, self.bias), (self.grad_W, self.grad_B)

    def get_parameters_count(self):
        return self.weights.size + self.bias.size

    def __repr__(self) -> str:
        return f"<Learnable, {self.weights.shape}, {self.bias.shape}, params={self.get_parameters_count()}>"


"""
Полносвязный линейный слой
"""


class LinearLayer(Learnable):
    """
    Реализует полносвязный линейный слой
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        """
        Args:
            input_size (int): размерность входных данных слоя
            output_size (int): размерность выходных данных слоя
        """
        self.input_size = input_size
        self.output_size = output_size

        # basic initialization
        weights = np.random.random((input_size, output_size))
        bias = np.random.random((1, output_size))
        super().__init__(weights, bias)

    def forward(self, x: Nums) -> Nums:
        # линейное преобразование
        t = x @ self.weights + self.bias
        # Записываем вход для обратного распространения
        self.cache = x
        return t

    def backward(self, error: Nums) -> Nums:
        """
        Обратное распространение ошибки
        """
        x = self.cache

        de_dW = x.T @ error
        de_dB = error
        de_dX = error @ self.weights.T

        self.grad_W += de_dW
        self.grad_B += de_dB
        return de_dX

    def __repr__(self) -> str:
        return f"<LinearLayer, {self.input_size}x{self.output_size}>"


"""
Свёрточный линейный слой
"""


class Conv2DLayer(Learnable):
    """
    Реализует свёрточный 2D слой
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> None:

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel
        self.strides = strides
        self.padding = padding

        weights = np.random.random((out_channels, in_channels, *kernel))
        bias = np.random.random((out_channels,))

        super().__init__(weights, bias)

    def forward(self, x: Nums):
        # Выходная размерность
        n_H = x.shape[1] + 2 * self.padding[0] - self.kernel_shape[0]
        n_W = x.shape[2] + 2 * self.padding[1] - self.kernel_shape[1]
        n_H = int(n_H / self.strides[0]) + 1
        n_W = int(n_W / self.strides[1]) + 1

        # Переводим окна в 2д матрицу
        x_col = im2col(x, self.kernel_shape, self.strides, self.padding)

        # Подготавливаем веса так, чтобы сделать матричное умножение
        weights_flatted = self.weights.reshape((self.out_channels, -1))
        bias_flatted = self.bias.reshape((-1, 1))

        convolved = (weights_flatted @ x_col) + bias_flatted
        convolved = convolved.reshape((self.out_channels, n_H, n_W))

        self.cache = (x_col, x.shape[1:])
        return convolved

    def backward(self, error: Nums):
        x_col, input_shape = self.cache

        # Переводим ошибку из оконного вида в плоский
        error_reshaped = error.reshape((self.out_channels, -1))

        de_dW = (error_reshaped @ x_col.T).reshape(self.weights.shape)
        de_dB = np.sum(error, axis=(1, 2))

        # Подготавливаем веса так, чтобы сделать матричное умножение
        weights_flatted = self.weights.reshape((self.out_channels, -1))

        error_col = weights_flatted.T @ error_reshaped

        # Сворачиваем в оконный вид
        de_dX = col2im(
            error_col,
            (self.in_channels, *input_shape),
            self.kernel_shape,
            self.strides,
            self.padding,
        )

        self.grad_W += de_dW
        self.grad_B += de_dB

        return de_dX

    def __repr__(self) -> str:
        return f"<Conv2dLayer, {self.weights.shape}>"


"""
Слой 2д пуллинга
"""


class MaxPoolingLayer(FunctionalNode):
    def __init__(
        self,
        kernel: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Tuple[int, int] = (0, 0),
    ):
        self.window = kernel
        self.strides = strides
        self.padding = padding

        self.window_size = self.window[0] * self.window[1]
        super().__init__()

    def forward(self, x):
        num_channels = x.shape[0]

        # Выходная размерность для каждого канала
        n_H = x.shape[1] + 2 * self.padding[0] - self.window[0]
        n_W = x.shape[2] + 2 * self.padding[1] - self.window[1]
        n_H = int(n_H / self.strides[0]) + 1
        n_W = int(n_W / self.strides[1]) + 1

        # Переводим окна в 2д матрицу
        x_col = im2col(x, self.window, self.strides, self.padding)
        # Для удобства, делаем 3д (каналы x размер окна x кол-во окон)
        x_col = x_col.reshape(num_channels, self.window_size, -1)

        # Делаем пулинг
        a_pol_channeled_flat = np.max(x_col, axis=1)
        a_pol = a_pol_channeled_flat.reshape((num_channels, n_H, n_W))

        self.cache = (num_channels, n_H, n_W, x.shape, x_col)
        return a_pol

    def backward(self, error):
        num_channels, n_H, n_W, input_shape, x_col = self.cache

        num_windows = n_H * n_W

        indices = np.argmax(x_col, axis=1).reshape((num_channels, n_H, n_W))

        filters_indices = np.repeat(np.arange(num_channels), num_windows)

        window_indices = np.tile(np.arange(num_windows), num_channels)

        # x_col[filters_indices, indices.ravel(), window_indices].reshape((num_channels, -1))
        col_with_error = np.zeros_like(x_col)
        np.add.at(
            col_with_error,
            (filters_indices, indices.ravel(), window_indices),
            error.ravel(),
        )

        col_with_error = col_with_error.reshape(
            (self.window_size * num_channels, num_windows)
        )
        d_X = col2im(
            col_with_error, input_shape, self.window, self.strides, self.padding
        )
        return d_X

    def __repr__(self) -> str:
        return f"<Pooling2dLayer>"


"""
Слой для перевода в одномерное пространство
"""


class FlattenLayer(FunctionalNode):

    def forward(self, x: Nums) -> Nums:
        self.cache = x.shape
        return x.reshape((1, x.size))

    def backward(self, error: Nums) -> Nums:
        shape = self.cache
        return error.reshape(shape)


"""
Дропаут
"""


class DropoutLayer(FunctionalNode):
    """
    Слой дропаута

    На каждой итерации оставляет включёнными только p долю случайных входов
    """

    def __init__(self, p: float):
        """
        Args:
            p (float): Доля входов, которые останутся включёнными
        """
        self.p = p

    def forward(self, x: Nums):
        self.rnd = np.random.uniform(low=0.0, high=1.0, size=x.shape) > self.p
        self.cache = x
        return self.rnd * x

    def backward(self, error: Nums):
        return self.rnd * error
