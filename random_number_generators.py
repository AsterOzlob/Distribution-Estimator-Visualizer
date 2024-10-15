## @file random_number_generators.py
#  Этот файл содержит определения абстрактного базового класса для генераторов случайных чисел
#  и его конкретной реализации, использующей преобразование квантилей.

from abc import ABC, abstractmethod
from random_variables import RandomVariable

import numpy as np


## Абстрактный базовый класс для генераторов случайных чисел.
#  Определяет интерфейс для всех генераторов случайных чисел.
#  Подклассы должны реализовать метод get().
class RandomNumberGenerator(ABC):
    ## Конструктор класса RandomNumberGenerator.
    #  @param random_variable Случайная величина, используемая для генерации чисел.
    def __init__(self, random_variable: RandomVariable) -> None:
        self.random_variable = random_variable

    ## Абстрактный метод для генерации случайных чисел.
    #  @param size Количество генерируемых случайных чисел.
    #  @return Массив случайных чисел.
    @abstractmethod
    def get(self, size: int) -> np.ndarray:
        pass


## Простой генератор случайных чисел, использующий преобразование квантилей.
#  Эта реализация использует равномерно распределённые числа и преобразует их с помощью функции квантилей.
class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable) -> None:
        super().__init__(random_variable)

    def get(self, size: int) -> np.ndarray:
        us = np.random.uniform(0, 1, size)
        return np.vectorize(self.random_variable.quantile)(us)
