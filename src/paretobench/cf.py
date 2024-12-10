import numpy as np
from itertools import count

from .problem import Problem, ProblemWithFixedPF, ProblemWithPF
from .utils import triangle_grid, triangle_grid_count
from .containers import Population


def get_pf_cf9_cf10(n, b):
    sub_n = int((np.sqrt(1 + 4 * n * b) - 1) / 2 / b) + 1

    # Add "line"
    f2s = [np.linspace(0, 1, sub_n)]
    f3s = [np.sqrt(1 - f2s[-1] ** 2)]
    f1s = [np.zeros(sub_n)]

    # Add surfaces
    n_triangle = next(n for n in count() if triangle_grid_count(n) > sub_n**2)
    for i in range(1, b + 1):
        f1, f3 = triangle_grid(n_triangle)
        start = np.sqrt((2 * i - 1) / 2 / b * (1 - f3**2))
        end = np.sqrt(2 * i / 2 / b * (1 - f3**2))
        f1 = f1 / np.maximum(1e-9, 1 - f3) * np.maximum(1e-9, end - start) + start
        f2 = np.sqrt(np.maximum(0, 1 - f1**2 - f3**2))
        f1s, f2s, f3s = f1s + [f1], f2s + [f2], f3s + [f3]
    return np.vstack((np.concatenate(f1s), np.concatenate(f2s), np.concatenate(f3s)))


class CFx(Problem):
    @property
    def reference(self):
        return (
            "Zhang, Q., Zhou, A., Zhao, S., Suganthan, P. N., Liu, W., & Tiwari, S. (2009). "
            "Multiobjective optimization Test Instances for the CEC 2009 Special Session "
            "and Competition. 31."
        )


class CF1(CFx, ProblemWithFixedPF):
    """The parameter N has be renamed to b to fit into python naming convention"""

    n: int = 10
    a: float = 1.0
    b: int = 10

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(2, self.n + 1)
        summand = (x[1:, :] - np.power(x[:1], 0.5 * (1.0 + 3 * (j[:, None] - 2) / (self.n - 2)))) ** 2
        f = np.vstack(
            (
                x[0] + 2 / j[1::2].size * np.sum(summand[1::2], axis=0),
                1 - x[0] + 2 / j[::2].size * np.sum(summand[::2], axis=0),
            )
        )
        g = np.vstack((f[0] + f[1] - self.a * np.abs(np.sin(self.b * np.pi * (f[0] - f[1] + 1))) - 1,))
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.zeros(self.n)

    @property
    def var_upper_bounds(self):
        return np.ones(self.n)

    def get_pareto_front(self):
        f1 = np.linspace(0, 1, 2 * self.b + 1)
        f2 = 1 - f1
        return np.vstack((f1, f2)).T


class CF2(CFx, ProblemWithPF):
    """The parameter N has be renamed to b to fit into python naming convention"""

    n: int = 10
    a: float = 1.0
    b: int = 2

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(2, self.n + 1)
        i = j % 2
        summand = (x[1:, :] - np.cos(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n - np.pi / 2 * i[:, None])) ** 2
        f = np.vstack(
            (
                x[0] + 2 / j[1::2].size * np.sum(summand[1::2], axis=0),
                1 - np.sqrt(x[0]) + 2 / j[::2].size * np.sum(summand[::2], axis=0),
            )
        )
        t = f[1] + np.sqrt(f[0]) - self.a * np.sin(self.b * np.pi * (np.sqrt(f[0]) - f[1] + 1)) - 1
        g = np.vstack((t / (1 + np.exp(4 * np.abs(t))),))
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -1 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.ones(self.n)

    def get_pareto_front(self, n):
        ranges = [(((2 * i - 1) / 2 / self.b) ** 2, (i / self.b) ** 2) for i in range(1, self.b + 1)]
        total_range = sum(stop - start for start, stop in ranges)
        f1 = np.concatenate(
            [np.linspace(start, stop, int(n * (stop - start) / total_range + 0.5)) for start, stop in ranges]
        )
        f2 = 1 - np.sqrt(f1)
        return np.vstack((f1, f2)).T


class CF3(CFx, ProblemWithPF):
    """The parameter N has be renamed to b to fit into python naming convention"""

    n: int = 10
    a: float = 1.0
    b: int = 2

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Checked - 9/1/2020
        j = np.arange(2, self.n + 1)
        y = x[1:, :] - np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n)

        summand = y**2
        prod = np.cos(20 * y * np.pi / np.sqrt(j[:, None]))

        f = np.vstack(
            (
                x[0] + 2 / j[1::2].size * (4 * np.sum(summand[1::2], axis=0) - 2 * np.prod(prod[1::2], axis=0) + 2),
                1
                - x[0] ** 2
                + 2 / j[::2].size * (4 * np.sum(summand[::2], axis=0) - 2 * np.prod(prod[::2], axis=0) + 2),
            )
        )
        g = np.vstack((f[1] + f[0] ** 2 - self.a * np.sin(self.b * np.pi * (f[0] ** 2 - f[1] + 1)) - 1,))
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -2 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1], 2 * np.ones(self.n - 1)))

    def get_pareto_front(self, n):
        ranges = [(np.sqrt((2 * i - 1) / 2 / self.b), np.sqrt(i / self.b)) for i in range(1, self.b + 1)]
        total_range = sum(stop - start for start, stop in ranges)
        f1 = np.concatenate(
            [np.linspace(start, stop, int(n * (stop - start) / total_range + 0.5)) for start, stop in ranges]
        )
        f2 = 1 - f1**2
        return np.vstack((f1, f2)).T


class CF4(CFx, ProblemWithPF):
    n: int = 10

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Checked - 9/1/2020
        j = np.arange(2, self.n + 1)
        y = x[1:, :] - np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n)

        summand = y**2
        summand[0] = np.abs(y[0])
        ind = y[0] >= 3 / 2 * (1 - np.sqrt(2) / 2)
        summand[0, ind] = 0.125 + (y[0, ind] - 1) ** 2

        f = np.vstack(
            (
                x[0] + np.sum(summand[1::2], axis=0),
                1 - x[0] + np.sum(summand[::2], axis=0),
            )
        )

        t = x[1] - np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n) - 0.5 * x[0] + 0.25
        g = np.vstack((t / (1 + np.exp(4 * np.abs(t))),))
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -2 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1], 2 * np.ones(self.n - 1)))

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = 1 - f1[f1 <= 0.5]
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)] = -f1[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)] / 2 + 3 / 4
        f2[f1 > 3 / 4] = 1 - f1[f1 > 3 / 4] + 1 / 8
        return np.vstack((f1, f2)).T


class CF5(CFx, ProblemWithPF):
    n: int = 10

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(2, self.n + 1)
        i = j % 2
        y = x[1:, :] - 0.8 * x[:1] * np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n + np.pi / 2 * i[:, None])

        summand = 2.0 * y**2 - np.cos(4.0 * np.pi * y) + 1.0
        summand[0] = np.abs(y[0])
        ind = y[0] >= 3 / 2 * (1 - np.sqrt(2) / 2)
        summand[0, ind] = 0.125 + (y[0, ind] - 1) ** 2

        f = np.vstack(
            (
                x[0] + np.sum(summand[1::2], axis=0),
                1 - x[0] + np.sum(summand[::2], axis=0),
            )
        )

        g = np.vstack((x[1] - 0.8 * x[0] * np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n) - 0.5 * x[0] + 0.25,))

        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -2 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1], 2 * np.ones(self.n - 1)))

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = 1 - f1[f1 <= 0.5]
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)] = -f1[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)] / 2 + 3 / 4
        f2[f1 > 3 / 4] = 1 - f1[f1 > 3 / 4] + 1 / 8
        return np.vstack((f1, f2)).T


class CF6(CFx, ProblemWithPF):
    n: int = 10

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(2, self.n + 1)
        i = j % 2
        y = x[1:, :] - 0.8 * x[:1] * np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n + np.pi / 2 * i[:, None])

        f = np.vstack(
            (
                x[0] + np.sum(y[1::2] ** 2, axis=0),
                (1 - x[0]) ** 2 + np.sum(y[::2] ** 2, axis=0),
            )
        )

        g1 = (
            x[1]
            - 0.8 * x[0] * np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n)
            - np.sign(0.5 * (1 - x[0]) - (1 - x[0]) ** 2) * np.sqrt(np.abs(0.5 * (1 - x[0]) - (1 - x[0]) ** 2))
        )
        g2 = (
            x[3]
            - 0.8 * x[0] * np.sin(6 * np.pi * x[0] + 4 * np.pi / self.n)
            - np.sign(0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0]))
            * np.sqrt(np.abs(0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0])))
        )
        g = np.vstack((g1, g2))

        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -2 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1], 2 * np.ones(self.n - 1)))

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = (1 - f1[f1 <= 0.5]) ** 2
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)] = (1 - f1[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)]) / 2
        f2[f1 > 3 / 4] = np.sqrt(1 - f1[f1 > 3 / 4]) / 4
        return np.vstack((f1, f2)).T


class CF7(CFx, ProblemWithPF):
    n: int = 10

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(2, self.n + 1)
        i = j % 2
        y = x[1:, :] - np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n + np.pi / 2 * i[:, None])
        h = 2 * y**2 - np.cos(4 * np.pi * y) + 1
        h[0] = y[0] ** 2
        h[2] = y[2] ** 2

        f = np.vstack((x[0] + np.sum(h[1::2], axis=0), (1 - x[0]) ** 2 + np.sum(h[::2], axis=0)))

        g1 = (
            x[1]
            - np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n)
            - np.sign(0.5 * (1 - x[0]) - (1 - x[0]) ** 2) * np.sqrt(np.abs(0.5 * (1 - x[0]) - (1 - x[0]) ** 2))
        )
        g2 = (
            x[3]
            - np.sin(6 * np.pi * x[0] + 4 * np.pi / self.n)
            - np.sign(0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0]))
            * np.sqrt(np.abs(0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0])))
        )
        g = np.vstack((g1, g2))

        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -2 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1], 2 * np.ones(self.n - 1)))

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = (1 - f1[f1 <= 0.5]) ** 2
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)] = (1 - f1[np.bitwise_and(f1 > 0.5, f1 <= 3 / 4)]) / 2
        f2[f1 > 3 / 4] = np.sqrt(1 - f1[f1 > 3 / 4]) / 4
        return np.vstack((f1, f2)).T


class CF8(CFx, ProblemWithPF):
    """The parameter N has be renamed to b to fit into python naming convention"""

    n: int = 10
    a: float = 4.0
    b: int = 2

    @property
    def m(self):
        return 3

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(3, self.n + 1)
        summand = (x[2:] - 2 * x[1][None, :] * np.sin(2 * np.pi * x[0][None, :] + j[:, None] * np.pi / self.n)) ** 2

        f = np.vstack(
            (
                np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi)
                + 2 / j[1::3].size * np.sum(summand[1::3], axis=0),
                np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi)
                + 2 / j[2::3].size * np.sum(summand[2::3], axis=0),
                np.sin(0.5 * x[0] * np.pi) + 2 / j[::3].size * np.sum(summand[::3], axis=0),
            )
        )

        g = np.vstack(
            (
                (f[0] ** 2 + f[1] ** 2) / (1 - f[2] ** 2)
                - self.a * np.abs(np.sin(self.b * np.pi * ((f[0] ** 2 - f[1] ** 2) / (1 - f[2] ** 2) + 1)))
                - 1,
            )
        )

        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0, 0], -4 * np.ones(self.n - 2)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1, 1], 4 * np.ones(self.n - 2)))

    def get_pareto_front(self, n):
        sub_n = n // (2 * self.b + 1)
        f3 = np.repeat(np.linspace(0, 1, sub_n), 2 * self.b + 1)
        f1 = np.concatenate([np.sqrt(i / 2 / self.b * (1 - f3[:sub_n] ** 2)) for i in range(2 * self.b + 1)])
        f2 = np.sqrt(np.maximum(0, 1 - f1**2 - f3**2))
        return np.vstack((f1, f2, f3)).T


class CF9(CFx, ProblemWithPF):
    """The parameter N has be renamed to b to fit into python naming convention"""

    n: int = 10
    a: float = 3.0
    b: int = 2

    @property
    def m(self):
        return 3

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(3, self.n + 1)
        summand = (x[2:] - 2 * x[1][None, :] * np.sin(2 * np.pi * x[0][None, :] + j[:, None] * np.pi / self.n)) ** 2

        f = np.vstack(
            (
                np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi)
                + 2 / j[1::3].size * np.sum(summand[1::3], axis=0),
                np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi)
                + 2 / j[2::3].size * np.sum(summand[2::3], axis=0),
                np.sin(0.5 * x[0] * np.pi) + 2 / j[::3].size * np.sum(summand[::3], axis=0),
            )
        )

        g = np.vstack(
            (
                (f[0] ** 2 + f[1] ** 2) / (1 - f[2] ** 2)
                - self.a * np.sin(self.b * np.pi * ((f[0] ** 2 - f[1] ** 2) / (1 - f[2] ** 2) + 1))
                - 1,
            )
        )

        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0, 0], -2 * np.ones(self.n - 2)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1, 1], 2 * np.ones(self.n - 2)))

    def get_pareto_front(self, n):
        return get_pf_cf9_cf10(n, self.b).T


class CF10(CFx, ProblemWithPF):
    """The parameter N has be renamed to b to fit into python naming convention"""

    n: int = 10
    a: float = 1.0
    b: int = 2

    @property
    def m(self):
        return 3

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        j = np.arange(3, self.n + 1)
        y = x[2:] - 2 * x[1][None, :] * np.sin(2 * np.pi * x[0][None, :] + j[:, None] * np.pi / self.n)
        summand = 4 * y**2 - np.cos(8 * np.pi * y) + 1

        f = np.vstack(
            (
                np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi)
                + 2 / j[1::3].size * np.sum(summand[1::3], axis=0),
                np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi)
                + 2 / j[1::3].size * np.sum(summand[2::3], axis=0),
                np.sin(0.5 * x[0] * np.pi) + 2 / j[::3].size * np.sum(summand[1::3], axis=0),
            )
        )

        g = np.vstack(
            (
                (f[0] ** 2 + f[1] ** 2) / (1 - f[2] ** 2)
                - self.a * np.sin(self.b * np.pi * ((f[0] ** 2 - f[1] ** 2) / (1 - f[2] ** 2) + 1))
                - 1,
            )
        )

        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0, 0], -2 * np.ones(self.n - 2)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1, 1], 2 * np.ones(self.n - 2)))

    def get_pareto_front(self, n):
        return get_pf_cf9_cf10(n, self.b).T
