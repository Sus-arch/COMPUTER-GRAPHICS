import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import art3d
import json
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


class Cube(Shape3D):
    """Куб"""
    
    def __init__(self, size=1.0):
        super().__init__("Куб")
        self.generate_vertices(size)
    
    def generate_vertices(self, size=1.0):
        self.vertices = np.array([
            [-size, -size, -size],  # 0
            [size, -size, -size],   # 1
            [size, size, -size],    # 2
            [-size, size, -size],   # 3
            [-size, -size, size],   # 4
            [size, -size, size],    # 5
            [size, size, size],     # 6
            [-size, size, size]     # 7
        ])
        
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # нижняя грань
            (4, 5), (5, 6), (6, 7), (7, 4),  # верхняя грань
            (0, 4), (1, 5), (2, 6), (3, 7)   # вертикальные ребра
        ]
        
        self.faces = [
            [0, 1, 2, 3],  # нижняя
            [4, 5, 6, 7],  # верхняя
            [0, 1, 5, 4],  # передняя
            [2, 3, 7, 6],  # задняя
            [0, 3, 7, 4],  # левая
            [1, 2, 6, 5]   # правая
        ]


class Pyramid(Shape3D):
    """Пирамида (тетраэдр)"""
    
    def __init__(self, size=1.0):
        super().__init__("Пирамида")
        self.generate_vertices(size)
    
    def generate_vertices(self, size=1.0):
        self.vertices = np.array([
            [0, size, 0],          # 0 - вершина
            [-size, -size, -size], # 1
            [size, -size, -size],  # 2
            [0, -size, size]       # 3
        ])
        
        self.edges = [
            (0, 1), (0, 2), (0, 3),  # ребра от вершины
            (1, 2), (2, 3), (3, 1)   # ребра основания
        ]
        
        self.faces = [
            [0, 1, 2],  # грань 1
            [0, 2, 3],  # грань 2
            [0, 3, 1],  # грань 3
            [1, 2, 3]   # основание
        ]


class Cylinder(Shape3D):
    """Цилиндр"""
    
    def __init__(self, radius=1.0, height=2.0, segments=12):
        super().__init__("Цилиндр")
        self.generate_vertices(radius, height, segments)
    
    def generate_vertices(self, radius=1.0, height=2.0, segments=12):
        # Вершины для верхнего и нижнего оснований
        vertices = []
        
        # Нижнее основание
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            vertices.append([x, -height/2, z])
        
        # Верхнее основание
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            vertices.append([x, height/2, z])
        
        self.vertices = np.array(vertices)
        
        # Ребра
        self.edges = []
        
        # Ребра оснований
        for i in range(segments):
            self.edges.append((i, (i+1)%segments))
            self.edges.append((i+segments, (i+1)%segments+segments))
        
        # Боковые ребра
        for i in range(segments):
            self.edges.append((i, i+segments))
        
        # Грани
        self.faces = []
        
        # Боковые грани
        for i in range(segments):
            next_i = (i+1)%segments
            self.faces.append([i, next_i, next_i+segments, i+segments])
        
        # Основания
        self.faces.append(list(range(segments)))  # нижнее
        self.faces.append(list(range(segments, 2*segments)))  # верхнее


class ShapeFactory:
    """Фабрика для создания фигур"""
    
    @staticmethod
    def create_shape(shape_type, **kwargs):
        if shape_type == "cube":
            return Cube(kwargs.get('size', 1.0))
        elif shape_type == "pyramid":
            return Pyramid(kwargs.get('size', 1.0))
        elif shape_type == "cylinder":
            return Cylinder(kwargs.get('radius', 1.0), kwargs.get('height', 2.0), kwargs.get('segments', 12))
        else:
            raise ValueError(f"Неизвестный тип фигуры: {shape_type}")


class Rotation3DApp:
    def __init__(self):
        # Создаем основное окно
        self.root = tk.Tk()
        self.root.title("Поворот объемного тела относительно осей координат на заданный угол.")
        self.root.geometry("1000x800")

        # Доступные фигуры
        self.available_shapes = {
            "Куб": ("cube", {"size": 1.0}),
            "Пирамида": ("pyramid", {"size": 1.0}),
            "Цилиндр": ("cylinder", {"radius": 1.0, "height": 2.0, "segments": 12})
        }

        # Текущая фигура
        self.current_shape = None
        self.original_vertices = None

        # Углы поворота
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0

        # Параметры отображения
        self.show_edges = tk.BooleanVar(value=True)
        self.show_faces = tk.BooleanVar(value=False)
        self.face_alpha = tk.DoubleVar(value=0.3)

        self.setup_gui()
        self.load_shape("Куб")

    def setup_gui(self):
        # Создаем фрейм для графика
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - управление
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Выбор фигуры
        ttk.Label(left_frame, text="Выбор фигуры:", font=('Arial', 10, 'bold')).pack(pady=(0, 5))

        shape_frame = ttk.Frame(left_frame)
        shape_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.shape_var = tk.StringVar(value="Куб")
        shape_combo = ttk.Combobox(shape_frame, textvariable=self.shape_var, 
                                  values=list(self.available_shapes.keys()))
        shape_combo.pack(fill=tk.X)
        shape_combo.bind('<<ComboboxSelected>>', self.on_shape_change)

        # Параметры фигуры
        self.params_frame = ttk.Frame(left_frame)
        self.params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Управление поворотом
        ttk.Label(left_frame, text="Управление поворотом:", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        rotation_frame = ttk.Frame(left_frame)
        rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(rotation_frame, text="Угол X:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.slider_x = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.slider_x.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.slider_x.bind('<Motion>', self.update_rotation)
        
        ttk.Label(rotation_frame, text="Угол Y:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.slider_y = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.slider_y.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.slider_y.bind('<Motion>', self.update_rotation)
        
        ttk.Label(rotation_frame, text="Угол Z:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.slider_z = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.slider_z.grid(row=2, column=1, sticky=tk.EW, padx=5)
        self.slider_z.bind('<Motion>', self.update_rotation)
        
        rotation_frame.columnconfigure(1, weight=1)
        
        # Кнопки управления
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Сброс", command=self.reset_rotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Сохранить", command=self.save_state).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Загрузить", command=self.load_state).pack(side=tk.LEFT, padx=2)
        
        # Настройки отображения
        ttk.Label(left_frame, text="Настройки отображения:", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        display_frame = ttk.Frame(left_frame)
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(display_frame, text="Показывать ребра", variable=self.show_edges, 
                       command=self.update_display).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(display_frame, text="Показывать грани", variable=self.show_faces, 
                       command=self.update_display).grid(row=1, column=0, sticky=tk.W, padx=5)
        
        ttk.Label(display_frame, text="Прозрачность:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Scale(display_frame, from_=0.1, to=1.0, variable=self.face_alpha, 
                 orient=tk.HORIZONTAL, command=lambda x: self.update_display()).grid(row=2, column=1, sticky=tk.EW, padx=5)
        
        display_frame.columnconfigure(1, weight=1)
        
        # Информация о фигуре
        info_frame = ttk.LabelFrame(left_frame, text="Информация о фигуре")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.info_text).pack(padx=5, pady=5)
        
        # Правая панель - график
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Создаем 3D график
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Встраиваем график в Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Обновляем интерфейс параметров
        self.update_shape_params()

    def update_shape_params(self):
        """Обновляет интерфейс параметров для текущей фигуры"""
        # Очищаем фрейм параметров
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        shape_name = self.shape_var.get()
        if shape_name in self.available_shapes:
            shape_type, params = self.available_shapes[shape_name]
            
            ttk.Label(self.params_frame, text="Параметры фигуры:", font=('Arial', 9, 'bold')).pack(anchor=tk.W)
            
            self.param_vars = {}
            row = 1
            for param_name, default_value in params.items():
                ttk.Label(self.params_frame, text=f"{param_name}:").pack(anchor=tk.W)
                
                if isinstance(default_value, float):
                    var = tk.DoubleVar(value=default_value)
                    scale = ttk.Scale(self.params_frame, from_=0.1, to=5.0, variable=var, orient=tk.HORIZONTAL)
                    scale.pack(fill=tk.X, pady=(0, 5))
                else:
                    var = tk.IntVar(value=default_value)
                    spinbox = ttk.Spinbox(self.params_frame, from_=1, to=20, textvariable=var)
                    spinbox.pack(fill=tk.X, pady=(0, 5))
                
                self.param_vars[param_name] = var
            
            ttk.Button(self.params_frame, text="Применить параметры", 
                      command=self.apply_shape_params).pack(pady=(5, 0))
    
    def apply_shape_params(self):
        """Применяет новые параметры к фигуре"""
        shape_name = self.shape_var.get()
        if shape_name in self.available_shapes:
            shape_type, _ = self.available_shapes[shape_name]
            params = {name: var.get() for name, var in self.param_vars.items()}
            self.load_shape(shape_name, **params)
    
    def on_shape_change(self, event=None):
        """Обработчик изменения выбранной фигуры"""
        self.update_shape_params()
        self.apply_shape_params()
    
    def load_shape(self, shape_name, **kwargs):
        """Загружает указанную фигуру"""
        if shape_name in self.available_shapes:
            shape_type, default_params = self.available_shapes[shape_name]
            # Объединяем параметры по умолчанию с переданными
            params = {**default_params, **kwargs}
            self.current_shape = ShapeFactory.create_shape(shape_type, **params)
            self.original_vertices = self.current_shape.vertices.copy()
            self.reset_rotation()
            self.update_info()
    
    def update_info(self):
        """Обновляет информацию о текущей фигуре"""
        if self.current_shape:
            bbox = self.current_shape.get_bounding_box()
            info = f"Фигура: {self.current_shape.name}\n"
            info += f"Вершин: {len(self.current_shape.vertices)}\n"
            info += f"Ребер: {len(self.current_shape.edges)}\n"
            info += f"Граней: {len(self.current_shape.faces)}\n"
            info += f"Размер: {bbox[1] - bbox[0]}"
            self.info_text.set(info)

    def rotation_matrix_x(self, angle):
        theta = np.radians(angle)
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def rotation_matrix_y(self, angle):
        theta = np.radians(angle)
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    def rotation_matrix_z(self, angle):
        theta = np.radians(angle)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    def rotate_object(self):
        """Применяет поворот к объекту"""
        if not self.current_shape:
            return None
        
        # Восстанавливаем исходные вершины
        rotated_vertices = self.original_vertices.copy()
        
        # Получаем углы из слайдеров
        angle_x = self.slider_x.get()
        angle_y = self.slider_y.get()
        angle_z = self.slider_z.get()
        
        # Применяем повороты
        if angle_x != 0:
            rotated_vertices = np.dot(rotated_vertices, self.rotation_matrix_x(angle_x).T)
        if angle_y != 0:
            rotated_vertices = np.dot(rotated_vertices, self.rotation_matrix_y(angle_y).T)
        if angle_z != 0:
            rotated_vertices = np.dot(rotated_vertices, self.rotation_matrix_z(angle_z).T)
            
        return rotated_vertices
    
    def update_rotation(self, event=None):
        """Обновляет отображение при изменении углов"""
        self.update_plot()
    
    def update_display(self):
        """Обновляет отображение при изменении настроек"""
        self.update_plot()
    
    def update_plot(self):
        """Обновляет 3D график"""
        self.ax.clear()
        
        if not self.current_shape:
            return
        
        # Получаем повернутые вершины
        rotated_vertices = self.rotate_object()
        
        # Рисуем грани
        if self.show_faces.get() and hasattr(self.current_shape, 'faces'):
            for face in self.current_shape.faces:
                points = rotated_vertices[face]
                poly = plt.Polygon(points[:, :2], alpha=self.face_alpha.get(), color='lightblue')
                self.ax.add_patch(poly)
                art3d.pathpatch_2d_to_3d(poly, z=0, zdir='z')
        
        # Рисуем ребра
        if self.show_edges.get():
            for edge in self.current_shape.edges:
                points = rotated_vertices[list(edge)]
                self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'blue', linewidth=2)
        
        # Рисуем вершины
        self.ax.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], 
                       c='red', s=50, alpha=0.7)
        
        # Настройки графика
        bbox = self.current_shape.get_bounding_box()
        max_dim = max(np.max(np.abs(bbox[0])), np.max(np.abs(bbox[1]))) * 1.2
        
        self.ax.set_xlim([-max_dim, max_dim])
        self.ax.set_ylim([-max_dim, max_dim])
        self.ax.set_zlim([-max_dim, max_dim])
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        angles = f"X={self.slider_x.get():.1f}°, Y={self.slider_y.get():.1f}°, Z={self.slider_z.get():.1f}°"
        self.ax.set_title(f'{self.current_shape.name} - Поворот: {angles}')
        
        self.ax.grid(True)
        self.canvas.draw()
    
    def reset_rotation(self):
        """Сбрасывает повороты"""
        self.slider_x.set(0)
        self.slider_y.set(0)
        self.slider_z.set(0)
        self.update_plot()
    
    def save_state(self):
        """Сохраняет текущее состояние"""
        state = {
            'shape': self.shape_var.get(),
            'params': {name: var.get() for name, var in self.param_vars.items()},
            'angles': {
                'x': self.slider_x.get(),
                'y': self.slider_y.get(),
                'z': self.slider_z.get()
            }
        }
        
        try:
            with open('rotation_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            print("Состояние сохранено")
        except Exception as e:
            print(f"Ошибка сохранения: {e}")
    
    def load_state(self):
        """Загружает сохраненное состояние"""
        try:
            with open('rotation_state.json', 'r') as f:
                state = json.load(f)
            
            self.shape_var.set(state['shape'])
            self.update_shape_params()
            
            # Устанавливаем параметры
            for name, value in state['params'].items():
                if name in self.param_vars:
                    self.param_vars[name].set(value)
            
            # Применяем параметры и углы
            self.apply_shape_params()
            self.slider_x.set(state['angles']['x'])
            self.slider_y.set(state['angles']['y'])
            self.slider_z.set(state['angles']['z'])
            
            self.update_plot()
            print("Состояние загружено")
            
        except FileNotFoundError:
            print("Файл состояния не найден")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
    
    def run(self):
        """Запускает приложение"""
        self.root.mainloop()


if __name__ == '__main__':
    # Запуск графического приложения
    app = Rotation3DApp()
    app.run()
