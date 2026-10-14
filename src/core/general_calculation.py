import math

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


    def print_system_equations(self):
        """
        Функция для вывода системы уравнений вида а11*х1+...+b1=0

        :return: None
        """

        Rich.debug_log("Вывод системы однородных уравнений:")

        for i, row in enumerate(self.matrix_A):
            terms = []
            for j, a in enumerate(row):
                x_index = j + 1
                if a < 0:
                    sign = "-"
                else:
                    sign = "+"
                coef = abs(a)
                terms.append(f"{sign} {coef:.3f}·x{x_index}")

            # первое слагаемое без первого "+"
            equation = " ".join(terms).lstrip("+ ").strip()
            equation += f" + ({-self.vector_b[i]:.3f}) = 0"

            Rich.debug_log(equation)

        Rich.print_spacer()

    def calculate_orthogonalization_method(self):
        """
        Функция для выполнения вычислений с матрицей методом ортогонализации

        :return: None
        """

        n = len(self.matrix_A)

        # векторы ai
        vectors_ai = [
            self.matrix_A[i] + [-self.vector_b[i]]
            for i in range(n)
        ]
        vectors_ai.append([0.0] * n + [1.0])  # an+1

        Rich.simple_log(f"Сформировано {len(vectors_ai)} расширенных векторов "
                        f"(матрица A + столбец -b + служебный вектор)")

        vectors_v = []

        for idx, a in enumerate(vectors_ai):
            label = f"a{idx + 1}" if idx < n else "a(n+1) (служебный)"

            Rich.print_spacer()
            Rich.debug_log(f"Шаг {idx + 1}: обрабатываем вектор {label} = "
                           f"[{', '.join(f'{c:.4f}' for c in a)}]")

            u = a[:]

            if not vectors_v:
                Rich.debug_log("  Это первый вектор — проекции вычитать не на что, "
                               "u = a")
            else:
                for j, v in enumerate(vectors_v, 1):
                    ratio = self.scalar_product(u, v)
                    proj = self.multiplication_by_scal(v, ratio)

                    Rich.debug_log(f"  Вычитаем проекцию на v{j}: "
                                   f"<u, v{j}> = {ratio: .6f}")

                    u = self.calculating_difference_of_vectors(u, proj)

                    Rich.debug_log(f"  u = u - {ratio: .6f}·v{j} = "
                                   f"[{', '.join(f'{c:.6f}' for c in u)}]")

            normalize = math.sqrt(self.scalar_product(u, u))

            Rich.debug_log(f"  |u| = {normalize:.6f}")

            v = self.multiplication_by_scal(u, 1.0 / normalize)
            vectors_v.append(v)

            Rich.debug_log(f"  v{idx + 1} = u / |u| = "
                           f"[{', '.join(f'{c:.6f}' for c in v)}]")

            # Проверим ортогональность с предыдущими v — полезно для наглядности
            if vectors_v[:-1]:
                checks = [
                    f"<v{idx + 1}, v{k}> = {self.scalar_product(v, prev): .2e}"
                    for k, prev in enumerate(vectors_v[:-1], 1)
                ]
                Rich.debug_log("  Проверка ортогональности: " + "; ".join(checks))

        v_last = vectors_v[-1]

        Rich.print_spacer()
        Rich.debug_log("Последний вектор v(n+1) = "
                       f"[{', '.join(f'{c:.6f}' for c in v_last)}]")
        Rich.debug_log("Решение: x_i = v(n+1)[i] / v(n+1)[n]")

        x = [v_last[i] / v_last[n] for i in range(n)]

        Rich.success_log("Решение системы:")
        for i, val in enumerate(x, 1):
            Rich.simple_log(f"  x{i} = {val:.6f}")

        residual = [
            sum(self.matrix_A[i][j] * x[j] for j in range(n)) - self.vector_b[i]
            for i in range(n)
        ]

        Rich.print_table(
            title="Невязка r = A·x - b",
            cols=["i", "r_i"],
            rows=[[i, f"{r:.3e}"] for i, r in enumerate(residual, 1)],
        )

        max_res = max(abs(r) for r in residual)
        if max_res < 1e-10:
            Rich.success_log(f"Максимум |r_i| = {max_res:.3e} — решение корректно")
        else:
            Rich.warning_log(f"Максимум |r_i| = {max_res:.3e} — возможна ошибка")


    def run(self):
        Rich.simple_log("Запуск вычислений решения СЛАУ методом ортогонализации")

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

        self.print_system_equations()

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
