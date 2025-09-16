import numpy as np
import tkinter as tk
from abc import ABC, abstractmethod


class Shape3D(ABC):
    """Абстрактный базовый класс для 3D фигур"""

    def __init__(self, name):
        self.name = name
        self.vertices = np.array([])  # вершины
        self.edges = []  # ребра
        self.faces = []  # грани


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
