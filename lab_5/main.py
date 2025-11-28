import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import random


def create_cube(position=(0, 0, 0), size=5.0, color=(0.2, 0.5, 0.9)):
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s]
    ]) + position
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
    ]
    return {'vertices': vertices, 'faces': faces, 'color': np.array(color), 'name': 'Куб'}


def create_tetrahedron(position=(0, 0, 0), size=6.0, color=(0.9, 0.2, 0.2)):
    s = size / np.sqrt(8)
    vertices = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]) * s + position
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    return {'vertices': vertices, 'faces': faces, 'color': np.array(color), 'name': 'Тетраэдр'}


def create_octahedron(position=(0, 0, 0), size=5.0, color=(0.2, 0.8, 0.3)):
    vertices = np.array([
        [0, 0, size], [0, 0, -size], [size, 0, 0],
        [-size, 0, 0], [0, size, 0], [0, -size, 0]
    ]) + position
    faces = [
        [0, 4, 2], [0, 2, 5], [0, 5, 3], [0, 3, 4],
        [1, 2, 4], [1, 4, 3], [1, 3, 5], [1, 5, 2]
    ]
    return {'vertices': vertices, 'faces': faces, 'color': np.array(color), 'name': 'Октаэдр'}


def create_pyramid(position=(0, 0, 0), size=6.0, color=(0.8, 0.6, 0.1)):
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [0, 0, s]
    ]) + position
    faces = [
        [0, 1, 2, 3],
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
    ]
    return {'vertices': vertices, 'faces': faces, 'color': np.array(color), 'name': 'Пирамида'}


class VisibilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная №5 — Построчное сканирование")
        self.root.geometry("1350x900")

        self.objects = []

        self.rotation_x_var = tk.DoubleVar(value=0)
        self.rotation_y_var = tk.DoubleVar(value=0)
        self.rotation_z_var = tk.DoubleVar(value=0)

        self.wireframe_var = tk.BooleanVar(value=True)
        self.culling_var = tk.BooleanVar(value=True)

        self.canvas_width = 950
        self.canvas_height = 750

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame, width=360)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        control_frame.pack_propagate(False)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(control_frame, text="Управление сценой", font=('Arial', 14, 'bold')).pack(pady=(0, 20))

        add_frame = ttk.LabelFrame(control_frame, text="Добавить фигуру")
        add_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(add_frame, text="Добавить куб", command=self.add_cube).pack(fill=tk.X, pady=2)
        ttk.Button(add_frame, text="Добавить тетраэдр", command=self.add_tetrahedron).pack(fill=tk.X, pady=2)
        ttk.Button(add_frame, text="Добавить октаэдр", command=self.add_octahedron).pack(fill=tk.X, pady=2)
        ttk.Button(add_frame, text="Добавить пирамиду", command=self.add_pyramid).pack(fill=tk.X, pady=2)
        ttk.Button(add_frame, text="Загрузить сцену с пересечениями", command=self.load_preloaded_scene).pack(fill=tk.X, pady=10, ipadx=10, ipady=5)

        list_frame = ttk.LabelFrame(control_frame, text="Объекты в сцене")
        list_frame.pack(fill=tk.X, pady=(0, 12))
        columns = ('№', 'Тип', 'Цвет')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=9)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

        ttk.Button(control_frame, text="Удалить выбранный", command=self.delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Очистить сцену", command=self.clear_scene).pack(fill=tk.X, pady=2)

        rot_frame = ttk.LabelFrame(control_frame, text="Поворот сцены (по 3 осям)")
        rot_frame.pack(fill=tk.X, pady=(0, 12))
        for label, var in [('X', self.rotation_x_var), ('Y', self.rotation_y_var), ('Z', self.rotation_z_var)]:
            ttk.Label(rot_frame, text=f"{label} (°):").pack(anchor=tk.W)
            ttk.Scale(rot_frame, from_=-180, to=180, variable=var, command=lambda e: self.update_display()).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(rot_frame, text="Сбросить поворот", command=self.reset_rotation).pack(fill=tk.X, pady=4)

        view_frame = ttk.LabelFrame(control_frame, text="Настройки отображения")
        view_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Checkbutton(view_frame, text="Каркас поверх (wireframe overlay)", variable=self.wireframe_var, command=self.update_display).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(view_frame, text="Отсечение задних граней (back-face culling)", variable=self.culling_var, command=self.update_display).pack(anchor=tk.W, pady=2)

        info_frame = ttk.LabelFrame(control_frame, text="Информация")
        info_frame.pack(fill=tk.X, expand=True)

        self.canvas_label = tk.Label(plot_frame, bg='#dddddd')
        self.canvas_label.pack(fill=tk.BOTH, expand=True)

        self.update_display()

    def rasterize_polygon(self, proj_verts, depths, face_color, zbuffer, colorbuffer):
        n = len(proj_verts)
        if n < 3: return

        ys = proj_verts[:, 1]
        y_min = max(0, int(np.min(ys)))
        y_max = min(self.canvas_height - 1, int(np.max(ys)))

        for y in range(y_min, y_max + 1):
            inters = []
            for i in range(n):
                j = (i + 1) % n
                y1, y2 = ys[i], ys[j]
                if y1 > y2:
                    y1, y2 = y2, y1
                    x1, x2 = proj_verts[j, 0], proj_verts[i, 0]
                    z1, z2 = depths[j], depths[i]
                else:
                    x1, x2 = proj_verts[i, 0], proj_verts[j, 0]
                    z1, z2 = depths[i], depths[j]
                if y < y1 or y > y2 or y1 == y2:
                    continue
                t = (y - y1) / (y2 - y1 + 1e-12)
                x_int = x1 + t * (x2 - x1)
                z_int = z1 + t * (z2 - z1)
                inters.append((x_int, z_int))

            if len(inters) < 2: continue
            inters.sort(key=lambda p: p[0])

            for k in range(0, len(inters)-1, 2):
                xl, zl = inters[k]
                xr, zr = inters[k+1]
                xl = int(np.ceil(xl))
                xr = int(np.floor(xr))
                if xl > xr: continue
                steps = xr - xl + 1
                if steps <= 0: continue
                dz = (zr - zl) / steps
                z_curr = zl + dz * (xl - xl + 0.5)

                for x in range(max(0, xl), min(self.canvas_width, xr + 1)):
                    if z_curr > zbuffer[y, x]:
                        colorbuffer[y, x] = face_color
                        zbuffer[y, x] = z_curr
                    z_curr += dz

    def draw_wireframe(self, rotated_verts_list, proj_verts_list, colorbuffer):
        black = np.array([0, 0, 0], dtype=np.uint8)
        for obj_idx, proj_verts in enumerate(proj_verts_list):
            verts_rot = rotated_verts_list[obj_idx]
            for face in self.objects[obj_idx]['faces']:
                for k in range(len(face)):
                    i1 = face[k]
                    i2 = face[(k + 1) % len(face)]
                    x1, y1 = int(proj_verts[i1, 0]), int(proj_verts[i1, 1])
                    x2, y2 = int(proj_verts[i2, 0]), int(proj_verts[i2, 1])
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    sx = 1 if x1 < x2 else -1
                    sy = 1 if y1 < y2 else -1
                    err = dx - dy
                    while True:
                        if 0 <= x1 < self.canvas_width and 0 <= y1 < self.canvas_height:
                            colorbuffer[y1, x1] = black
                        if x1 == x2 and y1 == y2: break
                        e2 = 2 * err
                        if e2 > -dy:
                            err -= dy
                            x1 += sx
                        if e2 < dx:
                            err += dx
                            y1 += sy

    def update_display(self):
        if not self.objects:
            self.canvas_label.configure(image='')
            return

        rx = np.radians(self.rotation_x_var.get())
        ry = np.radians(self.rotation_y_var.get())
        rz = np.radians(self.rotation_z_var.get())
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        rotated_verts_list = [obj['vertices'] @ R.T for obj in self.objects]

        all_rot = np.vstack(rotated_verts_list)
        minv = all_rot.min(axis=0)
        maxv = all_rot.max(axis=0)
        size = maxv - minv + 1e-8
        max_range = max(size[0], size[1])
        scale = min(self.canvas_width, self.canvas_height) * 0.45 / max_range
        cx = self.canvas_width // 2
        cy = self.canvas_height // 2

        zbuffer = np.full((self.canvas_height, self.canvas_width), -np.inf)
        colorbuffer = np.full((self.canvas_height, self.canvas_width, 3), 240, dtype=np.uint8)

        light_dir = np.array([0.3, 0.4, 1.0])
        light_dir /= np.linalg.norm(light_dir)

        proj_verts_list = []
        for verts_rot in rotated_verts_list:
            proj = np.zeros((len(verts_rot), 2))
            proj[:, 0] = verts_rot[:, 0] * scale + cx
            proj[:, 1] = -verts_rot[:, 1] * scale + cy
            proj_verts_list.append(proj)

        for obj_idx, verts_rot in enumerate(rotated_verts_list):
            proj = proj_verts_list[obj_idx]
            base_color = self.objects[obj_idx]['color'] * 255.0

            for face in self.objects[obj_idx]['faces']:
                poly_rot = verts_rot[face]
                poly_proj = proj[face]
                depths = verts_rot[face, 2]

                if len(poly_rot) < 3:
                    continue

                vec1 = poly_rot[1] - poly_rot[0]
                vec2 = poly_rot[2] - poly_rot[0]
                normal = np.cross(vec1, vec2)
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-8:
                    continue
                normal /= norm_len

                if self.culling_var.get() and np.dot(normal, [0.0, 0.0, 1.0]) <= 0.0:
                    continue

                cos_angle = np.dot(normal, light_dir)
                intensity = 0.3 + 0.7 * max(cos_angle, 0.0)
                face_color = (base_color * intensity).astype(np.uint8)

                self.rasterize_polygon(poly_proj, depths, face_color, zbuffer, colorbuffer)

        if self.wireframe_var.get():
            self.draw_wireframe(rotated_verts_list, proj_verts_list, colorbuffer)

        img = Image.fromarray(colorbuffer)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas_label.configure(image=self.photo)

    def update_objects_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, obj in enumerate(self.objects):
            r, g, b = (int(c * 255) for c in obj['color'])
            self.tree.insert('', tk.END, values=(i + 1, obj['name'], f"({r},{g},{b})"))

    def add_cube(self):      self.objects.append(create_cube(position=(random.uniform(-15,15), random.uniform(-15,15), random.uniform(-15,15)))); self.finalize_add()
    def add_tetrahedron(self): self.objects.append(create_tetrahedron(position=(random.uniform(-15,15), random.uniform(-15,15), random.uniform(-15,15)))); self.finalize_add()
    def add_octahedron(self):  self.objects.append(create_octahedron(position=(random.uniform(-15,15), random.uniform(-15,15), random.uniform(-15,15)))); self.finalize_add()
    def add_pyramid(self):     self.objects.append(create_pyramid(position=(random.uniform(-15,15), random.uniform(-15,15), random.uniform(-15,15)))); self.finalize_add()

    def finalize_add(self):
        self.update_objects_table()
        self.update_display()

    def load_preloaded_scene(self):
        self.clear_scene()
        self.objects.append(create_cube(position=(-4, -4, -4), size=12, color=(1.0, 0.3, 0.3)))
        self.objects.append(create_cube(position=(4, 4, 4), size=12, color=(0.3, 0.8, 1.0)))
        self.objects.append(create_tetrahedron(position=(0, 0, 0), size=18, color=(0.8, 0.8, 0.2)))
        self.update_objects_table()
        self.update_display()

    def delete_selected(self):
        sel = self.tree.selection()
        if sel:
            idx = self.tree.index(sel[0])
            del self.objects[idx]
            self.update_objects_table()
            self.update_display()

    def clear_scene(self):
        self.objects.clear()
        self.update_objects_table()
        self.update_display()

    def reset_rotation(self):
        self.rotation_x_var.set(0)
        self.rotation_y_var.set(0)
        self.rotation_z_var.set(0)
        self.update_display()


if __name__ == "__main__":
    root = tk.Tk()
    app = VisibilityApp(root)
    root.mainloop()
