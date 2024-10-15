##
#  @file estimations.py
#  Этот файл содержит определения абстрактного базового класса и его конкретных реализаций для статистических оценок.
#  Он включает в себя классы для вычисления эмпирической функции распределения (EDF) и гистограммы.

from abc import ABC
from typing import Optional

import numpy as np


## Абстрактный базовый класс для статистических оценок на основе выборки.
#  Этот класс служит базой для всех оценок, которые происходят из выборок данных.
class Estimation(ABC):
    ## Конструктор класса Estimation.
    #  @param sample Массив образцов данных.
    def __init__(self, sample: np.ndarray) -> None:
        self.sample = sample


## Класс для вычисления эмпирической функции распределения (EDF).
#  EDF - это статистическая оценка, которая назначает одинаковую вероятность каждой точке данных в выборке.
class EDF(Estimation):
    ## @brief Функция Хевисайда, используемая при вычислении EDF.
    #  @param x Значение для оценки.
    #  @return Возвращает 1, если x > 0, иначе 0.
    @staticmethod
    def heaviside_function(x: float) -> int:
        return 1 if x > 0 else 0

    ## Вычисляет значение EDF в указанных точках.
    #  @param x_array Массив точек, в которых вычисляется EDF.
    #  @return Массив значений EDF, соответствующих входным точкам.
    def value(self, x_array: np.ndarray) -> np.ndarray:
        return np.mean(np.vectorize(EDF.heaviside_function)(x_array - self.sample))


## @brief Класс для создания гистограммы по выборке данных.
#  Разбивает диапазон выборки на интервалы и подсчитывает количество точек в каждом интервале.
class Histogram(Estimation):
    ## Представляет интервал в гистограмме.
    class Interval:
        ## Инициализирует интервал с заданными границами.
        #  @param a Нижняя граница интервала.
        #  @param b Верхняя граница интервала.
        def __init__(self, a: float, b: float) -> None:
            self.left_boundary_of_interval = a
            self.right_boundary_of_intervals = b

        ## Проверяет, попадает ли значение в интервал.
        #  @param x Значение для проверки.
        #  @return True, если x находится внутри интервала, иначе False.
        def is_in(self, x: float) -> bool:
            return self.left_boundary_of_interval <= x < self.right_boundary_of_intervals

        ## Возвращает строковое представление интервала.
        #  @return Строка, представляющая границы интервала.
        def __repr__(self) -> str:
            return f'({self.left_boundary_of_interval}, {self.right_boundary_of_intervals})'

    ## Конструктор класса Histogram.
    #  @param sample Массив образцов данных.
    #  @param m Количество интервалов для создания.
    def __init__(self, sample: np.ndarray, m: int) -> None:
        super().__init__(sample)
        self.m = m
        self.init_intervals()

    ## Инициализирует интервалы на основе выборки данных и желаемого количества интервалов.
    def init_intervals(self) -> None:
        left_boundary_of_intervals = np.linspace(np.min(self.sample), np.max(self.sample), self.m + 1)[:-1]
        right_boundary_of_intervals = np.concatenate((left_boundary_of_intervals[1:], [np.max(self.sample)]))

        self.intervals = [Histogram.Interval(a, b) for a, b in
                          zip(left_boundary_of_intervals, right_boundary_of_intervals)]

        self.sub_interval_width = right_boundary_of_intervals[0] - left_boundary_of_intervals[0]

    ## Получает интервал, содержащий определенное значение.
    #  @param x Значение, для которого необходимо найти интервал.
    #  @return Интервал, содержащий x, или None, если интервал не найден.
    def get_interval(self, x: float) -> Optional[Interval]:
        for i in self.intervals:
            if i.is_in(x):
                return i
        return None

    ## Получает все точки выборки, попадающие в заданный интервал.
    #  @param interval Интервал для проверки по точкам выборки.
    #  @return Массив точек выборки внутри интервала.
    def get_sample_by_interval(self, interval: Interval) -> np.ndarray:
        return np.array(list(filter(lambda x: interval.is_in(x), self.sample)))

    ## Вычисляет значение гистограммы в определенной точке.
    #  @param x Точка, в которой вычисляется гистограмма.
    #  @return Значение гистограммы в точке x.
    def value(self, x: float) -> float:
        interval = self.get_interval(x)
        if interval is not None:
            sample_by_interval = self.get_sample_by_interval(interval)
            return len(sample_by_interval) / (self.sub_interval_width * len(self.sample))
        return 0.0
