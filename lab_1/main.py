import tkinter as tk


class Rotation3DApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Поворот объемного тела относительно осей координат на заданный угол.")
        self.root.geometry("1000x800")

        self.setup_gui()

    def setup_gui(self):
        frame_plot = tk.Frame(self.root)
        frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = Rotation3DApp()
    app.run()
