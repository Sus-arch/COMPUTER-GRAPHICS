import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox
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
        if hasattr(math, 'comb'):
            binom = math.comb(n, i)
        else:
            binom = math.factorial(n) // (math.factorial(i) * math.factorial(n - i))
        return binom * (t ** i) * ((1 - t) ** (n - i))

    def bezier_surface(self, u, v, control_points):
        """Вычисление точки на поверхности Безье для параметров u, v"""
        control_points = np.array(control_points, dtype=float)
        n = len(control_points) - 1  # степень по u
        m = len(control_points[0]) - 1  # степень по v

        point = np.zeros(3)
        for i in range(n + 1):
            for j in range(m + 1):
                bernstein_u = self.bernstein_polynomial(i, n, u)
                bernstein_v = self.bernstein_polynomial(j, m, v)
                point += bernstein_u * bernstein_v * control_points[i, j]

        return point

    def generate_surface(self, resolution=20):
        """Генерация точек поверхности"""
        if self.control_points is None or len(self.control_points) == 0:
            return None

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

        theta_x = np.radians(angle_x)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        theta_y = np.radians(angle_y)
        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        rotated_surface = self.surface_points.copy()
        shape = rotated_surface.shape
        rotated_surface = rotated_surface.reshape(-1, 3)

        rotated_surface = np.dot(rotated_surface, rot_y.T)
        rotated_surface = np.dot(rotated_surface, rot_x.T)

        return rotated_surface.reshape(shape)


class PointEditor:
    """Класс для редактирования опорных точек"""

    def __init__(self, parent, app, points_u=4, points_v=4):
        self.parent = parent
        self.app = app
        self.points_u = points_u
        self.points_v = points_v
        self.control_points = self.create_default_points()
        self.tree = None

    def create_default_points(self):
        """Создание точек по умолчанию (плоская поверхность)"""
        points = np.zeros((self.points_u, self.points_v, 3))
        for i in range(self.points_u):
            for j in range(self.points_v):
                points[i, j] = [i - self.points_u / 2 + 0.5,
                                j - self.points_v / 2 + 0.5, 0]
        return points.tolist()

    def create_editor_frame(self, parent):
        """Создание интерфейса для редактирования точек"""
        frame = ttk.LabelFrame(parent, text="Редактор опорных точек")

        # Параметры сетки
        grid_frame = ttk.Frame(frame)
        grid_frame.pack(fill=tk.X, pady=5)

        ttk.Label(grid_frame, text="Точек по U:").grid(row=0, column=0, padx=5)
        self.u_var = tk.IntVar(value=self.points_u)
        u_spin = ttk.Spinbox(grid_frame, from_=2, to=8, textvariable=self.u_var, width=5)
        u_spin.grid(row=0, column=1, padx=5)

        ttk.Label(grid_frame, text="Точек по V:").grid(row=0, column=2, padx=5)
        self.v_var = tk.IntVar(value=self.points_v)
        v_spin = ttk.Spinbox(grid_frame, from_=2, to=8, textvariable=self.v_var, width=5)
        v_spin.grid(row=0, column=3, padx=5)

        ttk.Button(grid_frame, text="Обновить сетку",
                   command=self.update_grid).grid(row=0, column=4, padx=5)

        # Таблица точек
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.create_point_table(table_frame)

        # Кнопки управления
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Загрузить пример",
                   command=self.load_example).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Сбросить",
                   command=self.reset_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Применить",
                   command=self.apply_and_update).pack(side=tk.RIGHT, padx=2)

        return frame

    def create_point_table(self, parent):
        """Создание таблицы для редактирования точек"""
        # Создаем Treeview
        columns = ("point", "x", "y", "z")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=15)

        # Настраиваем заголовки
        self.tree.heading("point", text="Точка")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("z", text="Z")

        # Настраиваем ширину колонок
        self.tree.column("point", width=80)
        self.tree.column("x", width=80)
        self.tree.column("y", width=80)
        self.tree.column("z", width=80)

        # Добавляем прокрутку
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Упаковываем
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Заполняем таблицу данными
        self.update_table_data()

        # Биндим двойной клик для редактирования
        self.tree.bind("<Double-1>", self.on_double_click)

    def update_table_data(self):
        """Обновление данных в таблице"""
        if self.tree is None:
            return

        # Очищаем таблицу
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Заполняем данными
        for i in range(self.points_u):
            for j in range(self.points_v):
                point = self.control_points[i][j]
                self.tree.insert("", "end", values=(
                    f"P({i},{j})",
                    f"{point[0]:.2f}",
                    f"{point[1]:.2f}",
                    f"{point[2]:.2f}"
                ))

    def on_double_click(self, event):
        """Обработчик двойного клика для редактирования"""
        item = self.tree.selection()[0]
        column = self.tree.identify_column(event.x)
        current_values = self.tree.item(item, "values")

        # Исправляем индексы колонок
        if column == "#2":  # Колонка X (индекс 1 в values)
            self.edit_cell(item, 1, "X", current_values)
        elif column == "#3":  # Колонка Y (индекс 2 в values)
            self.edit_cell(item, 2, "Y", current_values)
        elif column == "#4":  # Колонка Z (индекс 3 в values)
            self.edit_cell(item, 3, "Z", current_values)

    def edit_cell(self, item, value_index, column_name, current_values):
        """Редактирование ячейки таблицы"""
        # Создаем окно редактирования
        edit_window = tk.Toplevel(self.parent)
        edit_window.title(f"Редактирование {column_name}")
        edit_window.geometry("300x150")
        edit_window.transient(self.parent)
        edit_window.grab_set()

        # Получаем индекс точки из названия
        point_name = current_values[0]
        point_parts = point_name[2:-1].split(',')
        i, j = int(point_parts[0]), int(point_parts[1])

        ttk.Label(edit_window, text=f"Редактирование {column_name} для точки {point_name}",
                  font=('Arial', 10, 'bold')).pack(pady=10)

        current_value = tk.StringVar(value=current_values[value_index])
        entry = ttk.Entry(edit_window, textvariable=current_value, font=('Arial', 12), width=15)
        entry.pack(pady=10)
        entry.select_range(0, tk.END)
        entry.focus()

        def save_changes():
            try:
                new_value = float(current_value.get())
                # Обновляем данные (исправляем индекс координаты)
                coord_index = value_index - 1  # 1->0 (X), 2->1 (Y), 3->2 (Z)
                self.control_points[i][j][coord_index] = new_value
                # Обновляем таблицу
                self.update_table_data()
                # Автоматически применяем изменения
                self.apply_and_update()
                edit_window.destroy()
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректное числовое значение")

        def on_enter(event):
            save_changes()

        entry.bind("<Return>", on_enter)

        btn_frame = ttk.Frame(edit_window)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Сохранить", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=edit_window.destroy).pack(side=tk.LEFT, padx=5)

    def update_grid(self):
        """Обновление сетки точек"""
        self.points_u = self.u_var.get()
        self.points_v = self.v_var.get()
        self.control_points = self.create_default_points()
        self.update_table_data()
        self.apply_and_update()

    def reset_points(self):
        """Сброс точек к значениям по умолчанию"""
        self.control_points = self.create_default_points()
        self.update_table_data()
        self.apply_and_update()

    def load_example(self):
        """Загрузка примера многогранника"""
        examples = {
            "Сфера": self.create_sphere_example(),
            "Волна": self.create_wave_example(),
            "Седло": self.create_saddle_example()
        }

        # Диалог выбора примера
        example_window = tk.Toplevel(self.parent)
        example_window.title("Выбор примера")
        example_window.geometry("300x200")

        ttk.Label(example_window, text="Выберите пример многогранника:").pack(pady=10)

        for name, points in examples.items():
            ttk.Button(example_window, text=name,
                       command=lambda p=points: self.load_example_points(p, example_window)).pack(pady=2)

    def create_sphere_example(self):
        """Пример сферического многогранника (улучшенная сфера)"""
        points_u = 20  # больше точек по долготе
        points_v = 20  # больше точек по широте
        points = np.zeros((points_u, points_v, 3))
        radius = 1.5

        for i in range(points_u):
            for j in range(points_v):
                u = i / (points_u - 1) * 2 * np.pi  # долгота [0, 2π]
                v = j / (points_v - 1) * np.pi  # широта [0, π]

                points[i, j] = [
                    radius * np.sin(v) * np.cos(u),  # X
                    radius * np.sin(v) * np.sin(u),  # Y
                    radius * np.cos(v)  # Z
                ]
        return points.tolist()

    def create_wave_example(self):
        """Пример волнистой поверхности (более выразительной)"""
        points = np.zeros((4, 4, 3))
        for i in range(4):
            for j in range(4):
                x = (i - 1.5) * 1.5  # Увеличиваем разброс по X
                y = (j - 1.5) * 1.5  # Увеличиваем разброс по Y
                # Более выраженная волна с большей амплитудой
                z = 1.5 * np.sin(x * 1.2) * np.cos(y * 1.2)
                points[i, j] = [x, y, z]
        return points.tolist()

    def create_saddle_example(self):
        """Пример седловой поверхности"""
        points = np.zeros((4, 4, 3))
        for i in range(4):
            for j in range(4):
                x = (i - 1.5) * 1.2
                y = (j - 1.5) * 1.2
                points[i, j] = [x, y, 0.8 * (x ** 2 - y ** 2)]
        return points.tolist()

    def load_example_points(self, points, window):
        """Загрузка выбранного примера"""
        self.control_points = points
        self.points_u = len(points)
        self.points_v = len(points[0])
        self.u_var.set(self.points_u)
        self.v_var.set(self.points_v)
        self.update_table_data()
        self.apply_and_update()
        window.destroy()

    def apply_changes(self):
        """Применение изменений точек (возвращает новые точки)"""
        # Данные уже обновлены через редактирование таблицы
        return self.control_points

    def apply_and_update(self):
        """Применение изменений и обновление поверхности"""
        control_points = self.apply_changes()
        self.app.update_surface_from_points(control_points)


class BezierSurfaceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Поверхность Безье - Редактор многогранника")
        self.root.geometry("1600x900")

        # Инициализация поверхности Безье и редактора точек
        self.bezier_surface = BezierSurface()
        self.point_editor = PointEditor(self.root, self)

        # Параметры
        self.angle_x = 0
        self.angle_y = 0
        self.resolution = 20

        self.setup_gui()
        self.update_surface_from_points(self.point_editor.control_points)

    def setup_gui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - редактор точек
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Редактор точек
        editor_frame = self.point_editor.create_editor_frame(left_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        # Управление поворотом
        rotation_frame = ttk.LabelFrame(left_frame, text="Управление поворотом")
        rotation_frame.pack(fill=tk.X, pady=(10, 0))

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

        ttk.Button(rotation_frame, text="Сброс поворота",
                   command=self.reset_rotation).grid(row=2, column=0, columnspan=2, pady=5)

        # Настройки отображения
        display_frame = ttk.LabelFrame(left_frame, text="Настройки отображения")
        display_frame.pack(fill=tk.X, pady=(10, 0))

        self.show_control_points = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Показывать опорные точки",
                        variable=self.show_control_points, command=self.update_display).pack(anchor=tk.W)

        self.show_wireframe = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Показывать каркас",
                        variable=self.show_wireframe, command=self.update_display).pack(anchor=tk.W)

        self.show_surface = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Показывать поверхность",
                        variable=self.show_surface, command=self.update_display).pack(anchor=tk.W)

        ttk.Label(display_frame, text="Разрешение поверхности:").pack(anchor=tk.W)
        self.resolution_var = tk.IntVar(value=20)
        ttk.Scale(display_frame, from_=10, to=50, variable=self.resolution_var,
                  orient=tk.HORIZONTAL, command=self.update_resolution).pack(fill=tk.X)

        # Информация
        info_frame = ttk.LabelFrame(left_frame, text="Информация")
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_text = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.info_text, wraplength=250).pack(padx=5, pady=5)

        # Правая панель - график
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Создаем 3D график
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Встраиваем график в Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_info()

    def update_surface_from_points(self, control_points):
        """Обновление поверхности из переданных точек"""
        self.bezier_surface.control_points = control_points
        self.bezier_surface.generate_surface(self.resolution_var.get())
        self.update_plot()
        self.update_info()

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
            info = f"Размер сетки: {points_u} × {points_v}\n"
            info += f"Степень поверхности: ({points_u - 1}, {points_v - 1})\n"
            info += f"Разрешение: {self.resolution} × {self.resolution}\n"
            info += f"Углы: X={self.angle_x:.1f}°, Y={self.angle_y:.1f}°\n"
            info += f"Опорных точек: {points_u * points_v}"
            self.info_text.set(info)

    def update_plot(self):
        """Обновление графика"""
        self.ax.clear()

        if self.bezier_surface.surface_points is None:
            self.ax.set_title("Нет данных для отображения")
            self.canvas.draw()
            return

        # Получаем повернутую поверхность
        rotated_surface = self.bezier_surface.rotate_surface(self.angle_x, self.angle_y)

        # Отображаем поверхность
        if self.show_surface.get() and rotated_surface is not None:
            X = rotated_surface[:, :, 0]
            Y = rotated_surface[:, :, 1]
            Z = rotated_surface[:, :, 2]
            self.ax.plot_surface(X, Y, Z, alpha=0.7, color='lightblue',
                                 linewidth=0, antialiased=True)

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
        self.ax.set_title(f'Поверхность Безье\nПоворот: X={self.angle_x:.1f}°, Y={self.angle_y:.1f}°')

        # Автоматическое масштабирование
        if self.bezier_surface.surface_points is not None:
            all_points = self.bezier_surface.surface_points.reshape(-1, 3)
            if len(all_points) > 0:
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
    print("=== ПОВЕРХНОСТЬ БЕЗЬЕ - РЕДАКТОР МНОГОГРАННИКА ===")
    print("Инструкция:")
    print("1. Задайте сетку опорных точек (по умолчанию 4x4)")
    print("2. Двойной клик по ячейке для редактирования координат")
    print("3. Изменения применяются автоматически")
    print("4. Используйте слайдеры для вращения")
    print("5. Загрузите готовые примеры из меню 'Загрузить пример'")
    print("\nЗапуск приложения...")
    app = BezierSurfaceApp()
    app.run()