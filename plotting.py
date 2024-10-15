##
# @file plotting.py
# Содержит основные функции для построения данных

from pyqtgraph import mkPen
from PyQt5.QtWidgets import QWidget

## Глобальные переменные, используемые для построения графиков
PEN_RED = mkPen(color='r', width=2)   # Красная линия шириной 2
PEN_BLUE = mkPen(color='b', width=2)  # Синяя линия шириной 2
PEN_GREEN = mkPen(color='g', width=2) # Зеленая линия шириной 2


## Строит графики распределений на предоставленном виджете.
#  @param graph_widget Виджет QWidget, на котором будут построены графики.
#  @param x_array Общие данные для оси x для всех распределений.
#  @param y_truth Данные для оси y для графика истинного распределения.
#  @param y_edf Данные для оси y для графика эмпирической функции распределения.
#  @param y_kernel Данные для оси y для графика ядерного распределения.
#  @note Используемые перья: RED для истинного распределения, BLUE для EDF и GREEN для ядерного распределения.
def plot_distributions(graph_widget: QWidget, x_array, y_truth, y_edf, y_kernel) -> None:
    graph_widget.plot(x_array, y_truth, pen=PEN_RED)
    graph_widget.plot(x_array, y_edf, pen=PEN_BLUE)
    graph_widget.plot(x_array, y_kernel, pen=PEN_GREEN)


## Строит графики гистограмм на предоставленном виджете.
#  @param graph_widget Виджет, на котором будут построены гистограммы.
#  @param x_array Массив данных x, общий для всех гистограмм.
#  @param p_1 Данные y для первой гистограммы.
#  @param p_2 Данные y для второй гистограммы.
#  @param p_3 Данные y для третьей гистограммы.
#  @note Используются перья: RED для первой гистограммы, BLUE для второй и GREEN для третьей.
def plot_histograms(graph_widget: QWidget, x_array,p_1, p_2, p_3) -> None:
    graph_widget.plot(x_array, p_1, pen=PEN_RED)
    graph_widget.plot(x_array, p_2, pen=PEN_BLUE)
    graph_widget.plot(x_array, p_3, pen=PEN_GREEN)
