# Практика 3

Я бы поставила задачу так: "Показать процесс обратного распространения ошибки в полносвязной сети из двух (или одного, или больше) скрытых слоёв"

Датасет я бы взяла iris (из sklearn загружается через datasets.load_iris())

Обратное распространение ошибки - это часть процесса обучения нейросетей через поиск точки минимума функции ошибки.

Вообще процесс обучения состоит из прямого распространения (данные проходят через сеть) и обратного (вычисляется градиент ошибки и веса сети изменяются в соответствии с этим значением).

Наверное, тут стоит расписать матричные вычисления для прямого распространения, поиск производной функции активации, изменение весов в соответствии с ошибкой и производной.

Ну и обучить такую сеть на классификацию iris(чтобы сделать классификацию, требуется перекодировать метки классов в формат one-hot, т.е. целевым вектором для обучения будет вектор, где только одна единица, а все остальные нули)
