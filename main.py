"""
CLI-модуль для взаимодействия с классом Maze.
Генерация лабиринта, его решение, импорт лабиринта из файла|изображения(png), экспорт лабиринта в файл|изображение.
"""
import argparse

from maze import Maze


def main():
    """
    Основная функция для работы с классом Maze в интерфейсе командной строки.

    Returns:
    """
    parser = argparse.ArgumentParser(description="Генератор и решатель лабиринтов CLI")
    parser.add_argument("--size", type=str, help="Размер лабиринта в формате 'строки,столбцы'")
    parser.add_argument("--seed", type=str, help="Семя для генерации лабиринта")
    parser.add_argument("--solve_indecies", type=str,
                        help="Индексы для решения лабиринта в формате "
                             "'начало_строка,начало_столбец,конец_строка,конец_столбец'")
    parser.add_argument("--import_file", type=str,
                        help="Путь к файлу для импорта (используйте .png для изображений и .txt для текста)")
    parser.add_argument("--filename", type=str, help="Имя выходных файлов")
    parser.add_argument("--console_output", action="store_true", help="Вывести лабиринт в консоль")
    parser.add_argument("--text_output", action="store_true", help="Вывести лабиринт в текстовый файл")
    parser.add_argument("--image_output", action="store_true", help="Вывести лабиринт в изображение")

    args = parser.parse_args()
    maze = None

    if args.size:
        size = args.size.split(",")
        if len(size) != 2:
            print("Ошибка: Укажите размеры в формате 'строки,столбцы'.")
            return

        rows, cols = map(int, size)
        if args.seed:
            seed = int(args.seed)
            maze = Maze(rows, cols, seed)
        else:
            maze = Maze(rows, cols)
        maze.generate_maze()

    if args.import_file:
        maze = Maze()
        if args.import_file.endswith(".png"):
            maze.import_maze_from_image(args.import_file)
        elif args.import_file.endswith(".txt"):
            maze.import_maze_from_file(args.import_file)
        else:
            print("Ошибка: Неподдерживаемый формат файла для импорта."
                  " Используйте .png для изображений или .txt для текста.")
            return
    # если лабиринт не создан, последующие функции бессмысленны
    if maze is None:
        print("Ошибка: Укажите размер лабиринта или импортируйте лабиринт для решения.")
        return

    solve_indecies = args.solve_indecies.split(",")

    if len(solve_indecies) != 4:
        print("Ошибка: Укажите координаты для решения в формате"
              " 'начало_строка,начало_столбец,конец_строка,конец_столбец'.")
        return
    start, end = tuple(map(int, solve_indecies[:2])), tuple(map(int, solve_indecies[2:]))
    maze.solve_maze(start, end)

    if args.console_output:
        maze.print_maze()
        if maze.path:
            maze.print_solved_maze()

    if args.filename:
        if args.text_output:
            maze.export_maze_to_file(args.filename + ".txt")
        if args.image_output:
            maze.create_maze_png(maze.maze).save(args.filename + ".png", "PNG")


if __name__ == "__main__":
    main()

#python main.py --size 11,11 --solve_indecies 1,1,11,11 --filename maze1 --console_output --text_output --image_output