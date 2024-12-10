import numpy as np

from .problem import Problem
from .utils import rastrigin
from .containers import Population


class CTPx(Problem):
    @property
    def reference(self):
        return (
            "Zitzler, E. (Ed.). (2001). Evolutionary multi-criterion optimization: First international conference, "
            "EMO 2001, Zurich, Switzerland, March 2001: proceedings. Springer."
        )

    @property
    def var_lower_bounds(self):
        return np.concatenate(([0], -5.12 * np.ones(self.n - 1)))

    @property
    def var_upper_bounds(self):
        return np.concatenate(([1], 5.12 * np.ones(self.n - 1)))


class CTP1(CTPx):
    """The parameter J has been changed to j to correspond to python naming conventoins"""

    n: int = 5
    j: int = 2

    def __init__(self, **data):
        super().__init__(**data)

        # Calculate the parameters
        self._a = np.ones(self.j + 1)
        self._b = np.ones(self.j + 1)
        delta = 1 / (self.j + 1)
        x = delta
        for i in range(self.j):
            y = self._a[i] * np.exp(-self._b[i] * x)
            self._a[i + 1] = (self._a[i] + y) / 2
            self._b[i + 1] = -1 / x * np.log(y / self._a[i + 1])
            x = x + delta
        self._a = self._a[1:]
        self._b = self._b[1:]

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return self.j

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        rast = rastrigin(x[1:])
        f = np.vstack((x[0], rast * np.exp(-x[0] / rast)))

        g = []
        for i in range(self.j):
            g.append(f[1] - self._a[i] * np.exp(-self._b[i] * f[0]))
        return Population(f=f.T, g=np.vstack(g).T)


class CTP2_7(CTPx):
    """
    This class is a parent for the problems CTP2 - CTP7 which are just slight variations of one another
    """

    n: int
    _theta: float
    _a: float
    _b: float
    _c: float
    _d: float
    _e: float

    def __init__(self, theta, a, b, c, d, e, **data):
        # Handle pydantic data
        super().__init__(**data)

        # Set all of our private attrs
        self._theta = theta
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e

    @property
    def m(self):
        return 2

    @property
    def n_constraints(self):
        return 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        rast = rastrigin(x[1:])

        f = np.vstack((x[0], rast * (1 - x[0] / rast)))

        c = np.cos(self._theta) * (f[1] - self._e) - np.sin(self._theta) * f[0]
        c -= (
            self._a
            * np.abs(
                np.sin(
                    self._b * np.pi * (np.sin(self._theta) * (f[1] - self._e) + np.cos(self._theta) * f[0]) ** self._c
                )
            )
            ** self._d
        )
        g = np.vstack((c,))

        return Population(f=f.T, g=g.T)


class CTP2(CTP2_7):
    n: int = 5

    def __init__(self, **data):
        super().__init__(-0.2 * np.pi, 0.2, 10.0, 1.0, 6.0, 1.0, **data)


class CTP3(CTP2_7):
    n: int = 5

    def __init__(self, **data):
        super().__init__(-0.2 * np.pi, 0.1, 10.0, 1.0, 0.5, 1.0, **data)


class CTP4(CTP2_7):
    n: int = 5

    def __init__(self, **data):
        super().__init__(-0.2 * np.pi, 0.75, 10.0, 1.0, 0.5, 1.0, **data)


class CTP5(CTP2_7):
    n: int = 5

    def __init__(self, **data):
        super().__init__(-0.2 * np.pi, 0.1, 10.0, 2.0, 0.5, 1.0, **data)


class CTP6(CTP2_7):
    n: int = 5

    def __init__(self, **data):
        super().__init__(0.1 * np.pi, 40, 0.5, 1.0, 2.0, -2.0, **data)


class CTP7(CTP2_7):
    n: int = 5

    def __init__(self, **data):
        super().__init__(-0.05 * np.pi, 40, 5.0, 1.0, 6.0, 0.0, **data)
