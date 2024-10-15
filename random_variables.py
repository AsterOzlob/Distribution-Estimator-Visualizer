##
#  @file random_variables.py
#  Этот файл содержит классы для представления различных типов случайных величин,
#  включая абстрактный базовый класс и специфические распределения, такие как нормальное, равномерное,
#  экспоненциальное, Лапласа и Коши.

from abc import ABC, abstractmethod
from estimations import Estimation

import math
import numpy as np


## Абстрактный базовый класс для случайных величин.
#  Определяет интерфейс для всех случайных величин с основными статистическими функциями.
class RandomVariable(ABC):
    ## Вычисляет плотность вероятности в указанной точке.
    #  @param x Точка,в которой вычисляется плотность вероятности.
    #  @return Значение плотности вероятности в точке x.
    @abstractmethod
    def probability_density_function(self, x: float) -> float:
        pass

    ## Вычисляет кумулятивную функцию распределения в указанной точке.
    #  @param x Точка,в которой вычисляется кумулятивная функция распределения.
    #  @return Значение кумулятивной функции распределения в точке x.
    @abstractmethod
    def cumulative_distribution_function(self, x: float) -> float:
        pass

    ## Вычисляет квантильную функцию для заданного уровня вероятности.
    #  @param alpha Уровень вероятности.
    #  @return Квантиль, соответствующий уровню вероятности alpha.
    @abstractmethod
    def quantile(self, alpha: float) -> float:
        pass


## Представляет сглаженную версию случайной величины с использованием оценки плотности ядра.
#  Этот класс расширяет RandomVariable и использует выборку для оценки pdf и cdf.
class SmoothedRandomVariable(RandomVariable, Estimation):
    ## Инициализирует SmoothedRandomVariable с выборкой и шириной полосы.
    #  @param sample Массив выборочных точек данных.
    #  @param h Ширина полосы, используемая в оценке плотности ядра.
    def __init__(self, sample: np.ndarray, h: float) -> None:
        super().__init__(sample)
        self.h = h

    def probability_density_function(self, x: float) -> float:
        return np.mean([SmoothedRandomVariable._k((x - y) / self.h) for y in self.sample]) / self.h

    def cumulative_distribution_function(self, x: float) -> float:
        return np.mean([SmoothedRandomVariable._K((x - y) / self.h) for y in self.sample])

    ## Квантильная функция для сглаженной случайной величины.
    #  Не реализована в данном классе.
    #  @param alpha Уровень вероятности квантиля.
    #  @throws NotImplementedError Поскольку метод не реализован.
    def quantile(self, alpha: float) -> None:
        raise NotImplementedError

    ## Вспомогательная функция ядра для оценки плотности.
    #  Использует квадратичное ядро.
    #  @param x Нормализованное значение разности.
    #  @return Значение функции ядра в точке x.
    @staticmethod
    def _k(x: float) -> float:
        if abs(x) <= 1:
            return 0.75 * (1 - x * x)
        else:
            return 0

    ## Вспомогательная функция интеграла ядра для оценки кумулятивной функции распределения.
    #  Использует кусочно-линейную аппроксимацию квадратичного ядра.
    #  @param x Нормализованное значение разности.
    #  @return Интегральное значение функции ядра в точке x.
    @staticmethod
    def _K(x: float) -> float:
        if x < -1:
            return 0
        elif -1 <= x < 1:
            return 0.5 + 0.75 * (x - x ** 3 / 3)
        else:
            return 1


## @brief Представляет нормальное (гауссово) распределение.
#  Характеризуется параметрами местоположения (среднее) и масштаба (стандартное отклонение).
class NormalRandomVariable(RandomVariable):
    ## Конструктор нормальной случайной величины.
    #  @param location Среднее значение распределения.
    #  @param scale Стандартное отклонение распределения.
    def __init__(self, location: float = 0, scale: float = 1) -> None:
        super().__init__()
        self.location = location
        self.scale = scale

    def probability_density_function(self, x: float) -> float:
        z = (x - self.location) / self.scale
        return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self.scale)

    def cumulative_distribution_function(self, x: float) -> float:
        z = (x - self.location) / self.scale
        if z <= 0:
            return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

    def quantile(self, alpha: float) -> float:
        return self.location + 4.91 * self.scale * (math.pow(alpha, 0.14) - math.pow(1 - alpha, 0.14))


## Представляет равномерное распределение.
#  Характеризуется параметрами a (минимальное значение) и b (максимальное значение).
class UniformRandomVariable(RandomVariable):
    ## Конструктор равномерно распределенной случайной величины.
    #  @param a Минимальное значение диапазона.
    #  @param b Максимальное значение диапазона.
    def __init__(self, a: float = 0, b: float = 1) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def probability_density_function(self, x: float) -> float:
        return 1.0 / (self.b - self.a) if self.a <= x <= self.b else 0

    def cumulative_distribution_function(self, x: float) -> float:
        if x < self.a:
            return 0
        elif x > self.b:
            return 1
        else:
            return (x - self.a) / (self.b - self.a)

    def quantile(self, alpha: float) -> float:
        return self.a + alpha * (self.b - self.a)


## Представляет экспоненциальное распределение.
#  Это распределение характеризуется параметром скорости.
class ExponentialRandomVariable(RandomVariable):
    ## Инициализирует экспоненциальную случайную величину с заданной скоростью.
    #  @param rate Параметр скорости экспоненциального распределения.
    #  @throws ValueError если скорость не положительная.
    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self.rate = rate

    def probability_density_function(self, x: float) -> float:
        if x < 0:
            return 0
        return self.rate * math.exp(-self.rate * x)

    def cumulative_distribution_function(self, x: float) -> float:
        if x < 0:
            return 0
        return 1 - math.exp(-self.rate * x)

    def quantile(self, alpha: float) -> float:
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        return -math.log(1 - alpha) / self.rate


## Представляет распределение Лапласа.
#  Характеризуется параметром местоположения и параметром масштаба.
class LaplaceRandomVariable(RandomVariable):
    ## Инициализирует случайную величину распределения Лапласа с заданным местоположением и масштабом.
    #  @param location Параметр местоположения распределения Лапласа.
    #  @param scale Параметр масштаба распределения Лапласа.
    def __init__(self, location: float = 0, scale: float = 1) -> None:
        super().__init__()
        self.location = location
        self.scale = scale

    def probability_density_function(self, x: float) -> float:
        return 0.5 * math.exp(-abs(x - self.location) / self.scale) / self.scale

    def cumulative_distribution_function(self, x: float) -> float:
        if x < self.location:
            return 0.5 * math.exp((x - self.location) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.location) / self.scale)

    def quantile(self, alpha: float) -> float:
        if alpha < 0.5:
            return self.location + self.scale * math.log(2 * alpha)
        return self.location - self.scale * math.log(2 * (1 - alpha))


## Представляет распределение Коши.
#  Характеризуется параметром местоположения и параметром масштаба.
class CauchyRandomVariable(RandomVariable):
    ## Инициализирует случайную величину распределения Коши с заданным местоположением и масштабом.
    #  @param location Параметр местоположения распределения Коши.
    #  @param scale Параметр масштаба распределения Коши.
    def __init__(self, location: float = 0, scale: float = 1) -> None:
        self.location = location
        self.scale = scale

    def probability_density_function(self, x: float) -> float:
        return 1 / (math.pi * self.scale * (1 + ((x - self.location) / self.scale) ** 2))

    def cumulative_distribution_function(self, x: float) -> float:
        return 0.5 + math.atan((x - self.location) / self.scale) / math.pi

    def quantile(self, alpha: float) -> float:
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        return self.location + self.scale * math.tan(math.pi * (alpha - 0.5))
