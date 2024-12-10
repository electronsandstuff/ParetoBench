import numpy as np

from .problem import Problem
from .containers import Population


class SCH(Problem):
    """
    Description: Convex

    Other name: Schaffer function N. 1
    """

    @property
    def n(self):
        return 1

    @property
    def m(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        return Population(f=np.vstack((x[0] ** 2, (x[0] - 2) ** 2)).T)

    @property
    def var_lower_bounds(self):
        return -1e3 * np.ones(self.n)

    @property
    def var_upper_bounds(self):
        return 1e3 * np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class FON(Problem):
    """
    Description: non-convex

    Other name: Fonseca-Fleming Function
    """

    @property
    def n(self):
        return 3

    @property
    def m(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        term1 = -((x[0] - 1 / np.sqrt(3)) ** 2) - (x[1] - 1 / np.sqrt(3)) ** 2 - (x[2] - 1 / np.sqrt(3)) ** 2
        term2 = -((x[0] + 1 / np.sqrt(3)) ** 2) - (x[1] + 1 / np.sqrt(3)) ** 2 - (x[2] + 1 / np.sqrt(3)) ** 2
        return Population(
            f=np.array(
                [
                    1 - np.exp(term1),
                    1 - np.exp(term2),
                ]
            ).T
        )

    @property
    def var_lower_bounds(self):
        return -4 * np.ones(self.n)

    @property
    def var_upper_bounds(self):
        return 4 * np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class POL(Problem):
    """
    Description: non-convex, disconnected

    Other name: Poloni’s two objective function
    """

    @property
    def n(self):
        return 2

    @property
    def m(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
        a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
        b1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
        b2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
        return Population(
            f=np.array(
                [
                    1 + (a1 - b1) ** 2 + (a2 - b2) ** 2,
                    (x[0] + 3) ** 2 + (x[1] + 1) ** 2,
                ]
            ).T
        )

    @property
    def var_lower_bounds(self):
        return -np.pi * np.ones(self.n)

    @property
    def var_upper_bounds(self):
        return np.pi * np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class KUR(Problem):
    """
    Description: non-convex

    Other name: Kursawe’s Function
    """

    n: int = 3

    @property
    def m(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        return Population(
            f=np.array(
                [
                    np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1] ** 2 + x[1:] ** 2)), axis=0),
                    np.sum(np.abs(x) ** 0.8 + 5 * np.sin(x**3), axis=0),
                ]
            ).T
        )

    @property
    def var_lower_bounds(self):
        return -5 * np.ones(self.n)

    @property
    def var_upper_bounds(self):
        return 5 * np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class CONSTR(Problem):
    @property
    def n(self):
        return 2

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f = np.array([x[0], (1 + x[1]) / x[0]])
        g = np.array([x[1] + 9 * x[0] - 6, -x[1] + 9 * x[0] - 1])
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.array([0.1, 0.0])

    @property
    def var_upper_bounds(self):
        return np.array([[1.0, 5.0]])

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class SRN(Problem):
    @property
    def n(self):
        return 2

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f = np.array([(x[0] - 2) ** 2 + (x[1] - 1) ** 2 + 2, 9 * x[0] - (x[1] - 1) ** 2])
        g = np.array([225 - (x[0] ** 2 + x[1] ** 2), -10 - (x[0] - 3 * x[1])])
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return -20 * np.ones(self.n)

    @property
    def var_upper_bounds(self):
        return 20 * np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class TNK(Problem):
    @property
    def n(self):
        return 2

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 2

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f = np.array([x[0], x[1]])
        g = np.array(
            [
                -(-(x[0] ** 2) - x[1] ** 2 + 1 + 0.1 * np.cos(16 * np.arctan(x[0] / x[1]))),
                0.5 - ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2),
            ]
        )
        return Population(f=f.T, g=g.T)

    @property
    def var_bounds(self):
        return np.array([[0.0, 0.0], [np.pi, np.pi]])

    @property
    def var_lower_bounds(self):
        return np.zeros(self.n)

    @property
    def var_upper_bounds(self):
        return np.pi * np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )


class WATER(Problem):
    @property
    def n(self):
        return 3

    @property
    def m(self):
        return 5

    @property
    def n_constraints(self):
        return 7

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f = np.array(
            [
                106780.37 * (x[1] + x[2]) + 61704.67,
                3000.0 * x[0],
                305700 * 2289 * x[1] / (0.06 * 2289) ** 0.65,
                250 * 2289 * np.exp(-39.75 * x[1] + 9.9 * x[2] + 2.74),
                25 * (1.39 / (x[0] * x[1]) + 4940 * x[2] - 80),
            ]
        )
        g = np.array(
            [
                1.0 - (0.00139 / (x[0] * x[1]) + 4.94 * x[2] - 0.08),
                1.0 - (0.000306 / (x[0] * x[1]) + 1.082 * x[2] - 0.0986),
                50000.0 - (12.307 / (x[0] * x[1]) + 49408.25 * x[2] + 4051.02),
                16000.0 - (2.098 / (x[0] * x[1]) + 8046.33 * x[2] - 696.71),
                10000.0 - (2.138 / (x[0] * x[1]) + 7883.39 * x[2] - 705.04),
                2000.0 - (0.417 / (x[0] * x[1]) + 1721.26 * x[2] - 136.54),
                550.0 - (0.164 / (x[0] * x[1]) + 631.13 * x[2] - 54.48),
            ]
        )
        return Population(f=f.T, g=g.T)

    @property
    def var_lower_bounds(self):
        return np.array([0.01, 0.01, 0.01])

    @property
    def var_upper_bounds(self):
        return np.array([0.45, 0.10, 0.10])

    @property
    def reference(self):
        return (
            "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic "
            "algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197."
        )
