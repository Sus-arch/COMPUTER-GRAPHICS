import numpy as np
import tkinter as tk
from abc import ABC, abstractmethod


class Shape3D(ABC):
    """Абстрактный базовый класс для 3D фигур"""

    def __init__(self, name):
        self.name = name
        self.vertices = np.array([])  # вершины (координаты)
        self.edges = []  # ребра (пары индексов вершин)
        self.faces = []  # грани (группы индексов вершин)

    @abstractmethod
    def generate_vertices(self, size=1.0):
        pass

    def apply_transform(self, transformation_matrix):
        """Применяет матрицу преобразования к вершинам"""
        self.vertices = np.dot(self.vertices, transformation_matrix.T)  # перемножение матриц

    def get_bounding_box(self):
        """Возвращает ограничивающий параллелепипед"""
        if len(self.vertices) == 0:
            return np.array([[-1, -1, -1], [1, 1, 1]])
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])


class Rotation3DApp:
    def __init__(self):
        # Создаем основное окно
        self.root = tk.Tk()
        self.root.title("Поворот объемного тела относительно осей координат на заданный угол.")
        self.root.geometry("1000x800")

        self.setup_gui()

    def setup_gui(self):
        # Создаем фрейм для графика
        frame_plot = tk.Frame(self.root)
        frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def run(self):
        """Запускает приложение"""
        self.root.mainloop()


if __name__ == '__main__':
    # Запуск графического приложения
    app = Rotation3DApp()
    app.run()
