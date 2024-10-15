##
# @file main.py
# Содержит основные классы и функции приложения для демонстрации расчетов распределений и построения графиков.

from random_variables import (RandomVariable, SmoothedRandomVariable, NormalRandomVariable, UniformRandomVariable,
                              LaplaceRandomVariable, ExponentialRandomVariable, CauchyRandomVariable)
from random_number_generators import SimpleRandomNumberGenerator
from estimations import EDF, Histogram
from enums import DistComboBoxValues
from plotting import plot_distributions, plot_histograms
from PyQt5 import QtWidgets


import sys
import numpy as np
import gui


## Класс главного окна приложения для демонстрации расчетов распределений и построения графиков.
#  Этот класс интегрирует взаимодействие с интерфейсом пользователя с вычислениями статистических распределений,
#  обеспечивая динамическую визуализацию на основе пользовательского ввода.
class MyApplication(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    ## Инициализация окна приложения и подключение сигналов к слотам.
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.connect_signals()

    ## Подключает сигналы к слотам.
    #  Метод устанавливает связь между элементами GUI и соответствующими функциями обработчиками,
    #  которые вызываются при наступлении определенных событий.
    def connect_signals(self):
        self.DistComboBox.currentIndexChanged.connect(self.on_combobox_changed)
        self.CalcButton.clicked.connect(self.calculate)

    ## Обработчик событий изменения выбора в комбо-боксе.
    #  Обновляет интерфейс в зависимости от выбранного распределения.
    def on_combobox_changed(self) -> None:
        selected_text = self.DistComboBox.currentText()

        if selected_text == DistComboBoxValues.UniformDist.value:
            lower_label, upper_label = ("a:", "b:")
        elif selected_text == DistComboBoxValues.ExponentialDist.value:
            lower_label, upper_label = ("lambda:", "")
            self.UpperBoundLabel.setEnabled(False)
            self.UpperBoundLineEdit.setEnabled(False)
        else:
            lower_label, upper_label = ("Сдвиг:", "Масштаб:")
            self.UpperBoundLabel.setEnabled(True)
            self.UpperBoundLineEdit.setEnabled(True)

        self.LowerBoundLabel.setText(lower_label)
        self.UpperBoundLabel.setText(upper_label)

    ## Вычисляет распределения и строит графики на основе пользовательских данных.
    #  Выполняет расчеты и визуализацию распределений.
    def calculate(self) -> None:
        self.CDFGraphWidget.clear()
        self.PDFGraphWidget.clear()

        location, scale, size, bandwidth, num_intervals = self.get_input_parameters()

        random_variable = self.get_selected_random_variable(location, scale)

        generator = SimpleRandomNumberGenerator(random_variable)
        sample = generator.get(size)

        num_points = 100
        x_array = np.linspace(np.min(sample), np.max(sample), num_points)
        y_truth = np.vectorize(random_variable.cumulative_distribution_function)(x_array)

        edf = EDF(sample)
        y_edf = np.vectorize(edf.value)(x_array)

        smoothed_random_variable = SmoothedRandomVariable(sample, bandwidth)
        y_kernel = np.vectorize(smoothed_random_variable.cumulative_distribution_function)(x_array)

        plot_distributions(self.CDFGraphWidget, x_array, y_truth, y_edf, y_kernel)

        histogram = Histogram(sample, num_intervals)
        pdf_theoretical = np.vectorize(random_variable.probability_density_function)(x_array)
        pdf_histogram = np.vectorize(histogram.value)(x_array)
        pdf_kernel = np.vectorize(smoothed_random_variable.probability_density_function)(x_array)

        plot_histograms(self.PDFGraphWidget, x_array, pdf_theoretical, pdf_histogram, pdf_kernel)

    ## Извлекает входные параметры из GUI.
    #  Возвращает значения, введенные пользователем в интерфейсе.
    #  @return кортеж с параметрами распределения.
    def get_input_parameters(self) -> tuple:
        return (
            float(self.LowerBoundLineEdit.text()),
            float(self.UpperBoundLineEdit.text()),
            int(self.SizeLineEdit.text()),
            float(self.BandwithLineEdit.text()),
            int(self.IntervalsLineEdit.text())
        )

    ## Возвращает объект случайной величины в зависимости от выбранного пользователем распределения.
    #  @param location Сдвиг распределения.
    #  @param scale Масштаб распределения.
    #  @return объект RandomVariable соответствующего типа.
    def get_selected_random_variable(self, location: float, scale: float) -> RandomVariable:
        selected_text = self.DistComboBox.currentText()

        if selected_text == DistComboBoxValues.NormalDist.value:
            result = NormalRandomVariable(location, scale)
        if selected_text == DistComboBoxValues.UniformDist.value:
            result = UniformRandomVariable(location, scale)
        elif selected_text == DistComboBoxValues.ExponentialDist.value:
            result = ExponentialRandomVariable(location)
        elif selected_text == DistComboBoxValues.LaplaceDist.value:
            result = LaplaceRandomVariable(location, scale)
        elif selected_text == DistComboBoxValues.CauchyDist.value:
            result = CauchyRandomVariable(location, scale)

        return result


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApplication()
    window.show()
    app.exec_()
