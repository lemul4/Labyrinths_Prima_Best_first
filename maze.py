"""
Модуль для реализации класса Maze.
Генерация лабиринта с использованием алгоритма Прима.
Решение лабиринта с использованием алгоритма Best First Search.
"""
import random
from queue import PriorityQueue
from typing import List, Tuple

from PIL import Image, ImageDraw


class Maze:
    """
    Класс для генерации, решения и манипулирования лабиринтами.

    Args:
        rows (int): Количество строк в лабиринте.
        cols (int): Количество столбцов в лабиринте.

    Attributes:
        rows_fixed (int): Фиксированное количество строк в лабиринте (с учетом стен).
        cols_fixed (int): Фиксированное количество столбцов в лабиринте (с учетом стен).
        seed (int): Семя для генерации лабиринта.
        maze (List[List[int]]): Двумерный список, представляющий структуру лабиринта.
        path (List[List[int]]): Путь, найденный в процессе решения лабиринта.
    """

    def __init__(self, rows: int = 1, cols: int = 1, seed: int = 42) -> None:
        """
        Инициализация объекта Maze.

        Args:
            rows (int): Количество строк в лабиринте.
            cols (int): Количество столбцов в лабиринте.

        Returns:
            None
        """
        self.rows_fixed = rows + 2
        self.cols_fixed = cols + 2
        self.random_seed = seed
        self.maze = None
        self.path = None

    def generate_maze(self) -> None:
        """
        Генерация лабиринта с использованием алгоритма Прима.

        Returns:
            None
        """

        def get_walls_around(maze: List[List[int]], x: int, y: int) -> List[Tuple[int, int]]:
            # Вспомогательная функция для получения стен вокруг заданной ячейки
            removable_wall = 2
            if not ((0 <= x < len(maze)) and (0 <= y < len(maze[0]))):
                raise IndexError
            around = []
            for i in range(max(x - 1, 0), min(x + 2, len(maze))):
                for j in range(max(y - 1, 0), min(y + 2, len(maze[i]))):
                    if maze[i][j] == removable_wall:
                        around.append((i, j))
            return around

        random.seed(self.random_seed)
        non_visited_cell = 0
        visited_cell = -1
        removable_wall = 2
        non_removable_wall = 1

        # Инициализация лабиринта, где каждая ячейка с четными координатами - несъемная стена,
        # а каждая ячейка с нечетными координатами - проход
        self.maze = [[int(not (((i % 2) * (j % 2)) == 1)) for i in range(self.cols_fixed)] for j in
                     range(self.rows_fixed)]

        if self.rows_fixed < 2 or self.cols_fixed < 2:
            return  # Возвращаемся, если размеры лабиринта меньше 2x2

        # Устанавливаем несъемные стены по периметру лабиринта
        for x in range(self.rows_fixed):
            self.maze[x][0] = non_removable_wall
            self.maze[x][-1] = non_removable_wall
        for y in range(self.cols_fixed):
            self.maze[0][y] = non_removable_wall
            self.maze[-1][y] = non_removable_wall

        # Размещаем съемные стены внутри лабиринта
        for x in range(1, self.rows_fixed - 1):
            for y in range(1, self.cols_fixed - 1):
                if self.maze[x][y] and ((x % 2) == 1 or (y % 2) == 1) and not (((x % 2) * (y % 2)) == 1):
                    self.maze[x][y] = removable_wall

        walls_stack = []

        self.maze[1][1] = visited_cell
        # Добавляем стены вокруг начальной ячейки в стек
        for w in get_walls_around(self.maze, 1, 1):
            walls_stack.append(w)

        # Основной цикл алгоритма Прима
        while walls_stack:
            i = random.randint(0, len(walls_stack) - 1)
            wall = walls_stack.pop(i)

            x, y = wall

            unvisited_around_wall = 0
            # Подсчет количества непосещенных ячеек вокруг стены
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if self.maze[i][j] == non_visited_cell:
                        unvisited_around_wall += 1
                        self.maze[i][j] = visited_cell
                        # Добавление стен вокруг новой посещенной ячейки в стек
                        for w in get_walls_around(self.maze, i, j):
                            walls_stack.append(w)

            if unvisited_around_wall:
                self.maze[x][y] = visited_cell

        # Очистка лабиринта от временных меток, установка несъемных стен и возврат
        for x in range(self.rows_fixed):
            for y in range(self.cols_fixed):
                if self.maze[x][y] == -1 or self.maze[x][y] == 3 or self.maze[x][y] == 0:
                    self.maze[x][y] = non_visited_cell
                else:
                    self.maze[x][y] = non_removable_wall
        return

    def print_maze(self) -> None:
        """
        Вывод лабиринта в консоль.

        Returns:
            None
        """
        for row in self.maze:
            for elem in row:
                if elem:
                    print("[]", end="")
                else:
                    print("  ", end="")
            print()
        print()

    def print_solved_maze(self) -> None:
        """
        Вывод решенного лабиринта в консоль.

        Returns:
            None
        """
        for row_idx, row in enumerate(self.maze):
            for col_idx, elem in enumerate(row):
                pos = [row_idx, col_idx]
                if pos in self.path:
                    print("🐾", end="")
                elif elem:
                    print("[]", end="")
                else:
                    print("  ", end="")
            print()

    def solve_maze(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        """
        Решение лабиринта от заданной начальной позиции к конечной.

        Args:
            start (Tuple[int, int]): Начальная позиция в виде кортежа (строка, столбец).
            end (Tuple[int, int]): Конечная позиция в виде кортежа (строка, столбец).

        Returns:
            None
        """
        # Получение размеров лабиринта
        rows, cols = len(self.maze), len(self.maze[0])

        # Проверка, что стартовая и конечная позиции в пределах лабиринта
        if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not (0 <= end[0] < rows and 0 <= end[1] < cols):
            raise ValueError("Недопустимая начальная или конечная позиция")

        # Множество для отслеживания посещенных точек в лабиринте
        visited = set()

        # Приоритетная очередь для управления порядком обхода точек
        priority_queue = PriorityQueue()

        # Добавление стартовой точки в очередь с приоритетом, начальный путь - пустой список
        priority_queue.put((0, start, []))

        # Цикл продолжается, пока очередь не станет пустой
        while not priority_queue.empty():
            # Извлечение элемента из очереди с наименьшим приоритетом
            _, current_pos, path = priority_queue.get()

            # Если текущая позиция совпадает с конечной, сохранение найденного пути
            if current_pos == end:
                self.path = [list(p) for p in path] + [list(end)]

            # Если текущая позиция уже посещена, пропуск этой итерации цикла
            if current_pos in visited:
                continue

            # Добавление текущей позиции в список посещенных
            visited.add(current_pos)

            # Извлечение координат текущей позиции
            row, col = current_pos

            # Определение соседей текущей позиции
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

            # Обход соседей текущей позиции
            for neighbor in neighbors:
                n_row, n_col = neighbor

                # Проверка, что сосед находится в пределах лабиринта и не был посещен
                if rows > n_row >= 0 == self.maze[n_row][n_col] and 0 <= n_col < cols and neighbor not in visited:
                    # Вычисление приоритета для соседа с использованием эвристики
                    priority = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])

                    # Добавление соседа в очередь с приоритетом с обновленным путем
                    priority_queue.put((priority, neighbor, path + [current_pos]))

    def import_maze_from_file(self, filename: str) -> None:
        """
        Импорт лабиринта из текстового файла.

        Args:
            filename (str): Имя текстового файла, содержащего лабиринт.

        Returns:
            None
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                maze_data = [list(map(int, line.strip())) for line in file.readlines()]
                self.maze = maze_data
                self.rows_fixed = len(maze_data)
                self.cols_fixed = len(maze_data[0])

        except FileNotFoundError:
            print(f"Файл {filename} не найден.")

    def import_maze_from_image(self, filename: str) -> None:
        """
        Импорт лабиринта из изображения.

        Args:
            filename (str): Имя изображения, содержащего лабиринт.

        Returns:
            None
        """
        wall_color = (0, 0, 0)
        path_color = (255, 255, 255)
        try:
            image = Image.open(filename)
            width, height = image.size

            maze_data = []
            for y in range(0, height, 21):
                row = []
                for x in range(0, width, 21):
                    pixel = image.getpixel((x, y))
                    if pixel == wall_color:
                        row.append(1)
                    elif pixel == path_color:
                        row.append(0)
                    else:
                        raise ValueError("Неизвестный цвет пикселя на изображении")
                maze_data.append(row)
            self.maze = maze_data
            self.rows_fixed = len(maze_data)
            self.cols_fixed = len(maze_data[0])
        except FileNotFoundError:
            print(f"Файл {filename} не найден.")

    def export_maze_to_file(self, filename: str) -> None:
        """
        Экспорт лабиринта в текстовый файл.

        Args:
            filename (str): Имя текстового файла для сохранения лабиринта.

        Returns:
            None
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for row in self.maze:
                file.write(''.join(map(str, row)) + '\n')

    def create_maze_png(self, maze: List[List[int]]) -> Image.Image:
        """
        Создание изображения лабиринта с отмеченным путем, если он существует.

        Args:
            maze (List[List[int]]): Структура лабиринта.

        Returns:
            Image.Image: Изображение лабиринта в формате PIL.
        """
        cell_size = 20
        wall_color = (0, 0, 0)
        path_color = (255, 255, 255)
        find_color = (0, 255, 0)

        width = self.cols_fixed * cell_size
        height = self.rows_fixed * cell_size
        img = Image.new('RGB', (width, height), path_color)
        draw = ImageDraw.Draw(img)

        for i in range(self.rows_fixed):
            for j in range(self.cols_fixed):
                if maze[i][j] == 1:
                    draw.rectangle(((j * cell_size, i * cell_size),
                                    ((j + 1) * cell_size, (i + 1) * cell_size)), fill=wall_color)
        if self.path:
            for position in self.path:
                draw.rectangle(((position[1] * cell_size, position[0] * cell_size),
                                ((position[1] + 1) * cell_size, (position[0] + 1) * cell_size)), fill=find_color)
        return img
