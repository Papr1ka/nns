from typing import Tuple

from .base import np, Nums


def get_indices(
    num_channels: int,
    shape: Tuple[int, int],
    window: Tuple[int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
):
    """Возвращает индексы для оконного просмотра входного массива

    Args:
        num_channels (int): Количество выходных каналов
        shape (Tuple[int, int]): Размерность входного массива
        window (Tuple[int, int]): Размерность окна
        strides (Tuple[int, int]): Шаги по x и y
        padding (Tuple[int, int]): Отступы по x и y

    Returns:
        Tuple[Nums, Nums, Nums]: Индексы по x, по y, по каналам
    """
    out_h = int((shape[0] + 2 * padding[0] - window[0]) / strides[0]) + 1
    out_w = int((shape[1] + 2 * padding[1] - window[1]) / strides[1]) + 1

    # i indices

    # base for each slide
    slide_i = np.repeat(np.arange(window[0]), window[1])
    # for each channel the same
    slide_channels_i = np.tile(slide_i, num_channels)
    # move horizontally
    i = slide_channels_i[:, None] + strides[0] * np.repeat(np.arange(out_h), out_w)

    # j indices

    # base for each slide
    slide_j = np.tile(np.arange(window[1]), window[0])
    # for each channel
    slide_channels_j = np.tile(slide_j, num_channels)
    # move vertically
    j = slide_channels_j[:, None] + strides[1] * np.tile(np.arange(out_w), out_h)

    # channel indices

    c = np.repeat(np.arange(num_channels), window[0] * window[1])[:, None]

    return i, j, c


def add_padding(x: Nums, padding: Tuple[int, int]) -> Nums:
    """Возвращает массив с отступами по последним двум осям в количестве padding

    Args:
        x (Nums): Входной массив, 3д
        padding (Tuple[int, int]): Отступы по x и y

    Returns:
        Nums: Массив с отступами
    """
    return np.pad(x, ((0, 0), (padding[1], padding[1]), (padding[0], padding[0])))


def im2col(
    x: Nums, window: Tuple[int, int], strides: Tuple[int, int], padding: Tuple[int, int]
) -> Nums:
    """Возвращает 2д матрицу входного массива, размерности [количество окон * количество каналов x размер окна]

    Args:
        x (Nums): 3д массив
        window (Tuple[int, int]): Размер окна
        strides (Tuple[int, int]): Шаги по x и y
        padding (Tuple[int, int]): Отступы по x и y

    Returns:
        Nums: 2д матрица
    """
    if padding[0] != 0 or padding[1] != 0:
        padded = add_padding(x, padding)
    else:
        padded = x

    i, j, c = get_indices(x.shape[0], x.shape[1:], window, strides, padding)
    cols = padded[c, i, j]
    return cols


def col2im(
    x_col: Nums,
    x_shape: Tuple[int, int, int],
    window: Tuple[int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
) -> Nums:
    """Преобразует 2д матрицу окон в исходную 3д матрицу

    Args:
        x_col (Nums): 2д матрица с окнами (как в im2col)
        x_shape (Tuple[int, int, int]): Размерность входа для выходной матрицы (размерность входа для im2col)
        window (Tuple[int, int]): Размер окна
        strides (Tuple[int, int]): Шаги по x и y
        padding (Tuple[int, int]): Отступы по x и y

    Returns:
        Nums: 3д массив
    """
    H_padded = x_shape[1] + 2 * padding[0]
    W_padded = x_shape[2] + 2 * padding[1]

    new_x = np.zeros((x_shape[0], H_padded, W_padded))

    i, j, d = get_indices(x_shape[0], x_shape[1:], window, strides, padding)
    np.add.at(new_x, (d, i, j), x_col)
    if padding[0] != 0 or padding[1] != 0:
        return new_x[:, padding[0] : -padding[0], padding[1] : -padding[1]]
    return new_x
