import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.interpolate import CubicSpline
import matplotlib

matplotlib.use('TkAgg')


class SplineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Кубический сплайн - Лабораторная работа")
        self.root.geometry("1200x800")

        # Исходные точки (x, y, z)
        self.points = np.array([
            [1, 2, 0],
            [2, 3, 1],
            [3, 1, 2],
            [4, 4, 3],
            [5, 2, 4],
            [6, 5, 5]
        ])

        self.rotation_x = 0
        self.rotation_y = 0

        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - управление
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Правая панель - график
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Заголовок управления
        ttk.Label(control_frame, text="Управление точками",
                  font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        # Фрейм для таблицы точек
        points_frame = ttk.LabelFrame(control_frame, text="Координаты точек")
        points_frame.pack(fill=tk.X, pady=(0, 10))

        # Таблица точек
        self.create_points_table(points_frame)

        # Фрейм для добавления/удаления точек
        edit_frame = ttk.Frame(control_frame)
        edit_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(edit_frame, text="Добавить точку",
                   command=self.add_point).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(edit_frame, text="Удалить точку",
                   command=self.delete_point).pack(side=tk.LEFT)

        # Фрейм для поворота
        rotation_frame = ttk.LabelFrame(control_frame, text="Поворот")
        rotation_frame.pack(fill=tk.X, pady=(0, 10))

        # Поворот вокруг X
        ttk.Label(rotation_frame, text="Поворот X (°):").pack(anchor=tk.W)
        self.rotation_x_var = tk.DoubleVar(value=0)
        rotation_x_scale = ttk.Scale(rotation_frame, from_=-180, to=180,
                                     variable=self.rotation_x_var,
                                     command=self.on_rotation_change)
        rotation_x_scale.pack(fill=tk.X, pady=(0, 5))

        # Поворот вокруг Y
        ttk.Label(rotation_frame, text="Поворот Y (°):").pack(anchor=tk.W)
        self.rotation_y_var = tk.DoubleVar(value=0)
        rotation_y_scale = ttk.Scale(rotation_frame, from_=-180, to=180,
                                     variable=self.rotation_y_var,
                                     command=self.on_rotation_change)
        rotation_y_scale.pack(fill=tk.X, pady=(0, 5))

        # Кнопка сброса поворота
        ttk.Button(rotation_frame, text="Сбросить поворот",
                   command=self.reset_rotation).pack(fill=tk.X)

        # Информация
        info_frame = ttk.LabelFrame(control_frame, text="Информация")
        info_frame.pack(fill=tk.X)

        info_text = """Лабораторная работа по компьютерной графике

Кубический сплайн по 6 точкам:
• Красные точки - исходные точки
• Синяя ломаная - соединение точек
• Зеленая кривая - кубический сплайн
• Используется интерполяция CubicSpline"""

        ttk.Label(info_frame, text=info_text, justify=tk.LEFT,
                  font=('Arial', 9)).pack(padx=5, pady=5)

        # Создание графика
        self.create_plot(plot_frame)

    def create_points_table(self, parent):
        # Создаем Treeview для отображения точек
        columns = ('#', 'X', 'Y', 'Z')
        self.tree = ttk.Treeview(parent, columns=columns, show='headings', height=8)

        # Настраиваем заголовки
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=60)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Заполняем таблицу данными
        self.update_points_table()

        # Привязываем событие редактирования
        self.tree.bind('<Double-1>', self.on_cell_double_click)

    def update_points_table(self):
        # Очищаем таблицу
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Заполняем новыми данными
        for i, point in enumerate(self.points):
            self.tree.insert('', tk.END, values=(i + 1, *point))

    def create_plot(self, parent):
        # Создаем фигуру matplotlib
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Настраиваем canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        self.ax.clear()

        # Применяем поворот
        rotated_points = self.apply_rotation(self.points)

        # Сортируем точки по X для корректного построения сплайна
        sorted_indices = np.argsort(rotated_points[:, 0])
        sorted_points = rotated_points[sorted_indices]

        # Создаем параметр t для сплайна
        t = np.arange(len(sorted_points))

        # Создаем более плотную сетку для гладкого сплайна
        t_dense = np.linspace(0, len(sorted_points) - 1, 100)

        try:
            # Строим кубические сплайны для каждой координаты
            cs_x = CubicSpline(t, sorted_points[:, 0])
            cs_y = CubicSpline(t, sorted_points[:, 1])
            cs_z = CubicSpline(t, sorted_points[:, 2])

            # Вычисляем точки сплайна
            spline_x = cs_x(t_dense)
            spline_y = cs_y(t_dense)
            spline_z = cs_z(t_dense)

            # Рисуем сплайн
            self.ax.plot(spline_x, spline_y, spline_z, 'g-', linewidth=2, label='Кубический сплайн')

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить сплайн: {str(e)}")

        # Рисуем исходные точки
        self.ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                        c='red', s=50, label='Исходные точки')

        # Рисуем ломаную линию
        self.ax.plot(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                     'b--', alpha=0.7, label='Ломаная')

        # Настройки графика
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Кубический сплайн (Поворот X: {self.rotation_x:.1f}°, Y: {self.rotation_y:.1f}°)')
        self.ax.legend()

        # Автоматическое масштабирование
        self.ax.set_box_aspect([1, 1, 1])

        self.canvas.draw()

    def apply_rotation(self, points):
        # Вычисляем центр объекта для вращения вокруг центра
        center = np.mean(points, axis=0)

        # Преобразуем углы в радианы
        theta_x = np.radians(self.rotation_x)
        theta_y = np.radians(self.rotation_y)

        # Матрица поворота вокруг X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        # Матрица поворота вокруг Y
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # Комбинированная матрица поворота
        R = Ry @ Rx

        # Поворачиваем точки относительно центра
        points_centered = points - center
        rotated_centered = points_centered @ R.T
        rotated = rotated_centered + center

        return rotated

    def on_cell_double_click(self, event):
        # Обработка двойного клика для редактирования ячейки
        item = self.tree.selection()[0]
        column = self.tree.identify_column(event.x)

        if column in ('#2', '#3', '#4'):  # Колонки X, Y, Z
            col_index = int(column[1]) - 2  # Преобразуем в 0,1,2
            current_value = self.tree.item(item, 'values')[col_index + 1]

            # Создаем окно редактирования
            self.create_edit_window(item, col_index, current_value)

    def create_edit_window(self, item, col_index, current_value):
        edit_win = tk.Toplevel(self.root)
        edit_win.title("Редактирование координаты")
        edit_win.geometry("300x100")
        edit_win.transient(self.root)
        edit_win.grab_set()

        ttk.Label(edit_win, text=f"Новое значение для {'XYZ'[col_index]}:").pack(pady=5)

        new_value_var = tk.StringVar(value=current_value)
        entry = ttk.Entry(edit_win, textvariable=new_value_var)
        entry.pack(pady=5)
        entry.select_range(0, tk.END)
        entry.focus()

        def save_change():
            try:
                new_val = float(new_value_var.get())
                # Обновляем точку
                point_index = int(self.tree.item(item, 'values')[0]) - 1
                self.points[point_index][col_index] = new_val

                self.update_points_table()
                self.update_plot()
                edit_win.destroy()

            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректное числовое значение")

        ttk.Button(edit_win, text="Сохранить", command=save_change).pack(pady=5)
        edit_win.bind('<Return>', lambda e: save_change())

    def add_point(self):
        # Добавляем новую точку в конец
        new_point = self.points[-1] + np.array([1, 1, 1])
        self.points = np.vstack([self.points, new_point])

        self.update_points_table()
        self.update_plot()

    def delete_point(self):
        if len(self.points) > 3:  # Минимум 3 точки для сплайна
            self.points = self.points[:-1]  # Удаляем последнюю точку
            self.update_points_table()
            self.update_plot()
        else:
            messagebox.showwarning("Предупреждение", "Нельзя удалить точку. Минимум 3 точки требуется для сплайна.")

    def on_rotation_change(self, event=None):
        self.rotation_x = self.rotation_x_var.get()
        self.rotation_y = self.rotation_y_var.get()
        self.update_plot()

    def reset_rotation(self):
        self.rotation_x_var.set(0)
        self.rotation_y_var.set(0)
        self.rotation_x = 0
        self.rotation_y = 0
        self.update_plot()


def main():
    root = tk.Tk()
    app = SplineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()