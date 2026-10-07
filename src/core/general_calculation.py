from utils.output_rich import Rich


# Вариант 331

# Численный метод решения системы уравнений - метод ортогонализации

# Элементы матрицы А
# 0,405  0,05     0,04    0      0,09
# -0,061 0,53     0,073   0,11   -0,06
# 0,07   -0,036   0,38    0,03   0,02
# -0,05  0        0,066   0,58   0,23
# 0      0,081    -0,05   0      0,41

# Элементы вектора b
# -1,475
# 2,281
# 0,296
# 0,492
# 1,454

class OrthogonalizationMethod:
    """
    Класс для вычисления решения СЛАУ методом ортогонализации
    """

    def __init__(self):

        # массив элементов матрицы А
        self.matrix_A: list[list] = \
            [
                [0.405, 0.05, 0.04, 0, 0.09],
                [-0.061, 0.53, 0.073, 0.11, -0.06],
                [0.07, -0.036, 0.38, 0.03, 0.02],
                [-0.05, 0, 0.066, 0.58, 0.23],
                [0, 0.081, -0.05, 0, 0.41]
            ]

        # Элементы вектора b
        self.vector_b: list = \
            [
                -1.475,
                2.281,
                0.296,
                0.492,
                1.454
            ]


    def calculate_orthogonalization_method(self):
        """
        Функция для выполнения вычислений с матрицей методом ортогонализации

        :return:
        """

        Rich.print_spacer()
        Rich.debug_log("Исходная матрица А:")
        for row in self.matrix_A:
            # выравнивание вправо с точностью до тысячных и заданной шириной "ячейки"
            Rich.debug_log("".join(f"{element:>8.3f}" for element in row))
        Rich.print_spacer()

        Rich.debug_log("Элементы вектора b:")
        for element in self.vector_b:
            Rich.debug_log(f"   {element}")
        Rich.print_spacer()


    def run(self):
        Rich.simple_log("Запуск вычислений решения СЛАУ методом ортогонализации")

        self.calculate_orthogonalization_method()


    @staticmethod
    def scalar_product(vector1: list, vector2: list) -> float:
        """
        Функция для вычисления скалярного произведения двух векторов

        :param vector1: массив с координатами первого вектора
        :param vector2: массив с координатами второго вектора

        :return: float - результат вычисленного скалярного произведения
        """

        return sum(coord_v1 * coord_v2 for coord_v1, coord_v2 in zip(vector1, vector2))


    @staticmethod
    def multiplication_by_scal(vector: list, scal: float) -> list:
        """
        Функция для умножения вектора на скаляр

        :param vector: list - принимает вектор.
        :param scal: float - принимает число, на которое будет умножаться вектор

        :return: list - вернёт массив с координатами вектора,
         но уже умноженными на скаляр
        """

        return [scal * coord for coord in vector]


    @staticmethod
    def calculating_difference_of_vectors(vector1: list, vector2: list) -> list :
        """
        Функция для вычисления разности двух векторов

        :param vector1: массив с координатами первого вектора
        :param vector2: массив с координатами второго вектора

        :return:вернёт массив с вычисленным вектором-разностью
        """

        return [coord_v1 - coord_v2 for coord_v1,coord_v2 in zip(vector1, vector2)]
