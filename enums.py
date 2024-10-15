##
# @file enums.py
# Этот файл хранит все перечисления приложения

from enum import Enum


## Перечисление, определяющее значения для комбо-бокса распределений.
#  Это перечисление предоставляет варианты для различных типов распределений, доступных для выбора в комбо-боксе.
class DistComboBoxValues(Enum):
    NormalDist = "Нормальное распределение"
    UniformDist = "Равномерное распределение"
    ExponentialDist = "Экспоненциальное распределение"
    LaplaceDist = "Распределение Лапласа"
    CauchyDist = "Распределение Коши"
