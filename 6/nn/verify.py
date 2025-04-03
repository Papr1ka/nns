from time import time

from .layers import Learnable

from tabulate import tabulate
from .base import np


def verify(model, sample_x, sample_y):
    """
    Выводит общую информацию о модели, проверяет ее на корректность,
    выводит время выполнения, затрачиваемого на прямое и обратное распространение
    """
    headers = ["Номер", "Слой", "Вход", "Выход", "Параметры", "forward", "backward"]
    data = []

    total_params = 0

    time_forward = []
    time_backward = []

    for i, layer in enumerate(model.nodes, 1):
        input_shape = sample_x.shape
        try:
            t0 = time()
            sample_x = layer.forward(sample_x)
            t1 = time()
        except Exception as E:
            print(f"forward: Ошибка в слое {layer} ({i}), вход {sample_x.shape}")
            print(E)
            return layer, sample_x

        if np.isnan(sample_x).any():
            print(f"forward: Ошибка в слое {layer} ({i}), x превратился в NaN")
            return layer, sample_x

        elapsed_time = t1 - t0

        output_shape = sample_x.shape

        layer_name = layer.__class__.__name__

        params = ""

        if isinstance(layer, Learnable):
            total_params += layer.get_parameters_count()
            params = f"{layer.get_parameters_count()} {layer.weights.shape} + {layer.bias.shape}"

        time_forward.append(elapsed_time)

        data.append([i, layer_name, input_shape, output_shape, params])

    error = sample_x - sample_y

    for i, layer in zip(range(len(model.nodes), 0, -1), model.nodes[::-1]):
        try:
            t0 = time()
            error = layer.backward(error)
            t1 = time()
        except Exception as E:
            print(f"backward: Ошибка в слое {layer} ({i}), вход {error.shape}")
            print(E)
            return layer, error

        if np.isnan(error).any():
            print(f"forward: Ошибка в слое {layer} ({i}), x превратился в NaN")
            return layer, sample_x

        elapsed_time = t1 - t0
        time_backward.append(elapsed_time)

    time_forward_total = sum(time_forward)
    time_backward_total = sum(time_backward)
    time_total = time_forward_total + time_backward_total

    data = [
        [
            *elems,
            f"{t_forward:.6f} ({int(t_forward / time_forward_total * 100)} %)",
            f"{t_backward:.6f} ({int(t_backward / time_backward_total * 100)} %)",
        ]
        for elems, t_forward, t_backward in zip(data, time_forward, time_backward)
    ]

    print(tabulate(data, tablefmt="pretty", headers=headers))
    print("Всего параметров", total_params)
    print("Время прямого прохода", "{:.6f}".format(time_forward_total))
    print("Время обратного прохода", "{:.6f}".format(time_backward_total))
    print("Время всего", "{:.6f}".format(time_total))
