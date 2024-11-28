# Практика 4

Тут почти то же самое.

"Показать процесс обучения радиально-базисной сети с заданным количеством нейронов в скрытом слое".

Тоже на датасете iris, чтобы можно было сравнить результаты.

Радиально-базисные сети используются в основном для классификации. Они состоят из одного скрытого слоя, в котором количество нейронов равно количеству классов.

Чтобы построить радиально-базисную сеть, нужно знать, как данные располагаются в пространстве. То есть нужно знать центроиды каждого класса и расстояния от каждой точки до этих центроидов (расстояние можно брать евклидово).

Радиально-базисные функции, использующиеся как функции активации для таких сетей, зависят от этих параметров.

В качестве рбф можно взять функцию Гаусса. 

Собственно, тут стоит расписать процесс поиска центроидов и расстояний, функцию Гаусса, процесс прямого распространения с применением функции Гаусса к данным, процесс обратного.