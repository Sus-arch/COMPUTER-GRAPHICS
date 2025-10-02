import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

matplotlib.use('TkAgg')
import math


class BezierSurface:
    """Класс для работы с поверхностями Безье"""

    def __init__(self, control_points=None):
        self.control_points = control_points
        self.surface_points = None

    def bernstein_polynomial(self, i, n, t):
        """Вычисление полинома Бернштейна"""
        # Используем math.comb для биномиальных коэффициентов
        binom = math.comb(n, i)

        return binom * (t ** i) * ((1 - t) ** (n - i))

    def bezier_surface(self, u, v, control_points):
        """Вычисление точки на поверхности Безье для параметров u, v"""
        # Преобразуем в numpy array для гарантии правильных операций
        control_points = np.array(control_points, dtype=float)
        n = len(control_points) - 1  # степень по u
        m = len(control_points[0]) - 1  # степень по v

        point = np.zeros(3)
        for i in range(n + 1):
            for j in range(m + 1):
                bernstein_u = self.bernstein_polynomial(i, n, u)
                bernstein_v = self.bernstein_polynomial(j, m, v)
                # Убеждаемся, что все значения числовые
                point += bernstein_u * bernstein_v * control_points[i, j]

        return point

    def generate_surface(self, resolution=20):
        """Генерация точек поверхности"""
        if self.control_points is None:
            return None

        # Преобразуем контрольные точки в numpy array
        self.control_points = np.array(self.control_points, dtype=float)
        n = len(self.control_points)
        m = len(self.control_points[0])

        u_values = np.linspace(0, 1, resolution)
        v_values = np.linspace(0, 1, resolution)

        surface = np.zeros((resolution, resolution, 3))

        for i, u in enumerate(u_values):
            for j, v in enumerate(v_values):
                surface[i, j] = self.bezier_surface(u, v, self.control_points)

        self.surface_points = surface
        return surface

    def rotate_surface(self, angle_x, angle_y):
        """Поворот поверхности вокруг осей X и Y"""
        if self.surface_points is None:
            return None

        # Матрица поворота вокруг оси X
        theta_x = np.radians(angle_x)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        # Матрица поворота вокруг оси Y
        theta_y = np.radians(angle_y)
        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # Комбинированный поворот
        rotated_surface = self.surface_points.copy()
        shape = rotated_surface.shape
        rotated_surface = rotated_surface.reshape(-1, 3)

        # Применяем поворот сначала вокруг Y, затем вокруг X
        rotated_surface = np.dot(rotated_surface, rot_y.T)
        rotated_surface = np.dot(rotated_surface, rot_x.T)

        return rotated_surface.reshape(shape)


class ControlPointsManager:
    """Класс для управления опорными точками"""

    @staticmethod
    def create_plane_surface(size=2, points_u=4, points_v=4):
        """Создание плоской поверхности"""
        control_points = np.zeros((points_u, points_v, 3))

        for i in range(points_u):
            for j in range(points_v):
                u = i / (points_u - 1) if points_u > 1 else 0
                v = j / (points_v - 1) if points_v > 1 else 0
                control_points[i, j] = [
                    (u - 0.5) * size,
                    (v - 0.5) * size,
                    0
                ]

        return control_points

    @staticmethod
    def create_cylindrical_surface(radius=1, height=2, points_u=4, points_v=4):
        """Создание цилиндрической поверхности"""
        control_points = np.zeros((points_u, points_v, 3))

        for i in range(points_u):
            for j in range(points_v):
                u = i / (points_u - 1) if points_u > 1 else 0
                v = j / (points_v - 1) if points_v > 1 else 0

                angle = u * 2 * np.pi
                control_points[i, j] = [
                    radius * np.cos(angle),
                    (v - 0.5) * height,
                    radius * np.sin(angle)
                ]

        return control_points

    @staticmethod
    def create_spherical_surface(radius=1, points_u=4, points_v=4):
        """Создание сферической поверхности"""
        control_points = np.zeros((points_u, points_v, 3))

        for i in range(points_u):
            for j in range(points_v):
                u = i / (points_u - 1) if points_u > 1 else 0
                v = j / (points_v - 1) if points_v > 1 else 0

                theta = u * 2 * np.pi  # долгота
                phi = v * np.pi  # широта

                control_points[i, j] = [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.cos(phi),
                    radius * np.sin(phi) * np.sin(theta)
                ]

        return control_points

    @staticmethod
    def create_wave_surface(size=2, amplitude=0.5, points_u=4, points_v=4):
        """Создание волнистой поверхности"""
        control_points = np.zeros((points_u, points_v, 3))

        for i in range(points_u):
            for j in range(points_v):
                u = i / (points_u - 1) if points_u > 1 else 0
                v = j / (points_v - 1) if points_v > 1 else 0

                x = (u - 0.5) * size
                y = (v - 0.5) * size
                z = amplitude * np.sin(u * 2 * np.pi) * np.cos(v * 2 * np.pi)

                control_points[i, j] = [x, y, z]

        return control_points


class BezierSurfaceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Поверхности Безье с поворотом")
        self.root.geometry("1400x900")

        # Инициализация поверхности Безье
        self.bezier_surface = BezierSurface()
        self.current_surface_type = "plane"

        # Параметры
        self.angle_x = 0
        self.angle_y = 0
        self.resolution = 20

        self.setup_gui()
        self.load_surface("plane")

    def setup_gui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - управление
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Выбор типа поверхности
        ttk.Label(left_frame, text="Тип поверхности:", font=('Arial', 10, 'bold')).pack(pady=(0, 5))

        surface_types = {
            "Плоская": "plane",
            "Цилиндрическая": "cylinder",
            "Сферическая": "sphere",
            "Волнистая": "wave"
        }

        self.surface_var = tk.StringVar(value="Плоская")
        surface_combo = ttk.Combobox(left_frame, textvariable=self.surface_var,
                                     values=list(surface_types.keys()))
        surface_combo.pack(fill=tk.X, pady=(0, 10))
        surface_combo.bind('<<ComboboxSelected>>', self.on_surface_change)

        # Параметры поверхности
        params_frame = ttk.LabelFrame(left_frame, text="Параметры поверхности")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Количество точек по U
        ttk.Label(params_frame, text="Точек по U:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.points_u_var = tk.IntVar(value=4)
        ttk.Spinbox(params_frame, from_=2, to=8, textvariable=self.points_u_var,
                    command=self.update_surface_params).grid(row=0, column=1, sticky=tk.EW, padx=5)

        # Количество точек по V
        ttk.Label(params_frame, text="Точек по V:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.points_v_var = tk.IntVar(value=4)
        ttk.Spinbox(params_frame, from_=2, to=8, textvariable=self.points_v_var,
                    command=self.update_surface_params).grid(row=1, column=1, sticky=tk.EW, padx=5)

        # Размер
        ttk.Label(params_frame, text="Размер:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.size_var = tk.DoubleVar(value=2.0)
        ttk.Scale(params_frame, from_=0.5, to=5.0, variable=self.size_var,
                  orient=tk.HORIZONTAL, command=self.update_surface_params).grid(row=2, column=1, sticky=tk.EW, padx=5)

        # Амплитуда (для волнистой поверхности)
        ttk.Label(params_frame, text="Амплитуда:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.amplitude_var = tk.DoubleVar(value=0.5)
        ttk.Scale(params_frame, from_=0.1, to=2.0, variable=self.amplitude_var,
                  orient=tk.HORIZONTAL, command=self.update_surface_params).grid(row=3, column=1, sticky=tk.EW, padx=5)

        params_frame.columnconfigure(1, weight=1)

        # Управление поворотом
        rotation_frame = ttk.LabelFrame(left_frame, text="Управление поворотом")
        rotation_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(rotation_frame, text="Угол X:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.slider_x = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.slider_x.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.slider_x.set(0)
        self.slider_x.bind('<Motion>', self.update_rotation)

        ttk.Label(rotation_frame, text="Угол Y:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.slider_y = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.slider_y.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.slider_y.set(0)
        self.slider_y.bind('<Motion>', self.update_rotation)

        rotation_frame.columnconfigure(1, weight=1)

        # Кнопки управления
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(button_frame, text="Сброс поворота", command=self.reset_rotation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Обновить поверхность", command=self.update_surface).pack(fill=tk.X, pady=2)

        # Настройки отображения
        display_frame = ttk.LabelFrame(left_frame, text="Настройки отображения")
        display_frame.pack(fill=tk.X, pady=(0, 10))

        self.show_control_points = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Показывать опорные точки",
                        variable=self.show_control_points, command=self.update_display).pack(anchor=tk.W)

        self.show_wireframe = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Показывать каркас",
                        variable=self.show_wireframe, command=self.update_display).pack(anchor=tk.W)

        self.show_surface = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Показывать поверхность",
                        variable=self.show_surface, command=self.update_display).pack(anchor=tk.W)

        ttk.Label(display_frame, text="Разрешение:").pack(anchor=tk.W)
        self.resolution_var = tk.IntVar(value=20)
        ttk.Scale(display_frame, from_=10, to=50, variable=self.resolution_var,
                  orient=tk.HORIZONTAL, command=self.update_resolution).pack(fill=tk.X)

        # Информация
        info_frame = ttk.LabelFrame(left_frame, text="Информация")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.info_text, wraplength=250).pack(padx=5, pady=5)

        # Правая панель - график
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Создаем 3D график
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Встраиваем график в Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_info()

    def on_surface_change(self, event=None):
        """Обработчик изменения типа поверхности"""
        surface_types = {
            "Плоская": "plane",
            "Цилиндрическая": "cylinder",
            "Сферическая": "sphere",
            "Волнистая": "wave"
        }
        surface_type = surface_types[self.surface_var.get()]
        self.load_surface(surface_type)

    def load_surface(self, surface_type, **kwargs):
        """Загрузка поверхности указанного типа"""
        self.current_surface_type = surface_type

        points_u = self.points_u_var.get()
        points_v = self.points_v_var.get()
        size = self.size_var.get()
        amplitude = self.amplitude_var.get()

        if surface_type == "plane":
            control_points = ControlPointsManager.create_plane_surface(size, points_u, points_v)
        elif surface_type == "cylinder":
            control_points = ControlPointsManager.create_cylindrical_surface(size / 2, size, points_u, points_v)
        elif surface_type == "sphere":
            control_points = ControlPointsManager.create_spherical_surface(size / 2, points_u, points_v)
        elif surface_type == "wave":
            control_points = ControlPointsManager.create_wave_surface(size, amplitude, points_u, points_v)
        else:
            control_points = ControlPointsManager.create_plane_surface(size, points_u, points_v)

        self.bezier_surface.control_points = control_points
        self.bezier_surface.generate_surface(self.resolution_var.get())
        self.update_plot()
        self.update_info()

    def update_surface_params(self, event=None):
        """Обновление параметров поверхности"""
        self.load_surface(self.current_surface_type)

    def update_surface(self):
        """Обновление поверхности"""
        self.load_surface(self.current_surface_type)

    def update_resolution(self, event=None):
        """Обновление разрешения поверхности"""
        self.resolution = self.resolution_var.get()
        self.bezier_surface.generate_surface(self.resolution)
        self.update_plot()

    def update_rotation(self, event=None):
        """Обновление поворота"""
        self.angle_x = self.slider_x.get()
        self.angle_y = self.slider_y.get()
        self.update_plot()

    def update_display(self):
        """Обновление отображения"""
        self.update_plot()

    def reset_rotation(self):
        """Сброс поворота"""
        self.slider_x.set(0)
        self.slider_y.set(0)
        self.angle_x = 0
        self.angle_y = 0
        self.update_plot()

    def update_info(self):
        """Обновление информации о поверхности"""
        if self.bezier_surface.control_points is not None:
            points_u = len(self.bezier_surface.control_points)
            points_v = len(self.bezier_surface.control_points[0])
            info = f"Тип: {self.surface_var.get()}\n"
            info += f"Опорных точек: {points_u} × {points_v}\n"
            info += f"Степень поверхности: ({points_u - 1}, {points_v - 1})\n"
            info += f"Разрешение: {self.resolution} × {self.resolution}\n"
            info += f"Углы: X={self.angle_x:.1f}°, Y={self.angle_y:.1f}°"
            self.info_text.set(info)

    def update_plot(self):
        """Обновление графика"""
        self.ax.clear()

        if self.bezier_surface.surface_points is None:
            return

        # Получаем повернутую поверхность
        rotated_surface = self.bezier_surface.rotate_surface(self.angle_x, self.angle_y)

        # Отображаем поверхность
        if self.show_surface.get() and rotated_surface is not None:
            X = rotated_surface[:, :, 0]
            Y = rotated_surface[:, :, 1]
            Z = rotated_surface[:, :, 2]
            self.ax.plot_surface(X, Y, Z, alpha=0.7, color='lightblue', linewidth=0)

        # Отображаем каркас
        if self.show_wireframe.get() and rotated_surface is not None:
            X = rotated_surface[:, :, 0]
            Y = rotated_surface[:, :, 1]
            Z = rotated_surface[:, :, 2]
            self.ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5, alpha=0.5)

        # Отображаем опорные точки
        if self.show_control_points.get() and self.bezier_surface.control_points is not None:
            control_points = np.array(self.bezier_surface.control_points, dtype=float)

            # Поворачиваем опорные точки
            theta_x = np.radians(self.angle_x)
            theta_y = np.radians(self.angle_y)

            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])

            rot_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])

            rotated_control_points = np.dot(control_points.reshape(-1, 3), rot_y.T)
            rotated_control_points = np.dot(rotated_control_points, rot_x.T)
            rotated_control_points = rotated_control_points.reshape(control_points.shape)

            # Отображаем точки
            self.ax.scatter(rotated_control_points[:, :, 0],
                            rotated_control_points[:, :, 1],
                            rotated_control_points[:, :, 2],
                            color='red', s=50, alpha=0.8)

            # Отображаем сетку опорных точек
            for i in range(rotated_control_points.shape[0]):
                self.ax.plot(rotated_control_points[i, :, 0],
                             rotated_control_points[i, :, 1],
                             rotated_control_points[i, :, 2], 'r-', alpha=0.3)

            for j in range(rotated_control_points.shape[1]):
                self.ax.plot(rotated_control_points[:, j, 0],
                             rotated_control_points[:, j, 1],
                             rotated_control_points[:, j, 2], 'r-', alpha=0.3)

        # Настройки графика
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Поверхность Безье - {self.surface_var.get()}\n'
                          f'Поворот: X={self.angle_x:.1f}°, Y={self.angle_y:.1f}°')

        # Автоматическое масштабирование
        if self.bezier_surface.surface_points is not None:
            all_points = self.bezier_surface.surface_points.reshape(-1, 3)
            max_range = np.max(np.abs(all_points)) * 1.2
            self.ax.set_xlim([-max_range, max_range])
            self.ax.set_ylim([-max_range, max_range])
            self.ax.set_zlim([-max_range, max_range])

        self.ax.grid(True)
        self.canvas.draw()

    def run(self):
        """Запуск приложения"""
        self.root.mainloop()


if __name__ == "__main__":
    app = BezierSurfaceApp()
    app.run()
