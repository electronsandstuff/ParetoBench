import numpy as np

from .problem import Problem
from .utils import rastrigin


class CTPx(Problem):
    def get_reference(self):
        return "Zitzler, E. (Ed.). (2001). Evolutionary multi-criterion optimization: First international conference, "\
               "EMO 2001, Zurich, Switzerland, March 2001: proceedings. Springer."


class CTP1(CTPx):
    """The parameter J has been changed to j to correspond to python naming conventoins"""
    
    n: int = 5
    j: int = 2
    
    def __init__(self, **data):
        super().__init__(**data)

        # Calculate the parameters
        self._a = np.ones(self.j+1)
        self._b = np.ones(self.j+1)
        delta = 1/(self.j + 1)
        x = delta
        for i in range(self.j):
            y = self._a[i]*np.exp(-self._b[i]*x)
            self._a[i+1] = (self._a[i] + y)/2
            self._b[i+1] = -1/x*np.log(y/self._a[i+1])
            x = x + delta
        self._a = self._a[1:]
        self._b = self._b[1:]

    @property
    def n_vars(self):
        return self.n

    @property
    def n_objs(self):
        return 2
    
    @property
    def n_constraints(self):
        return self.j
    
    def _call(self, x):
        rast = rastrigin(x[1:])
        f = np.vstack((
            x[0],
            rast*np.exp(-x[0]/rast)
        ))

        g = []
        for i in range(self.j):
            g.append(f[1] - self._a[i]*np.exp(-self._b[i]*f[0]))
        return f, np.vstack(g)

    @property
    def var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-5.12, 5.12])).T
        b[0, 0] = 0.0
        b[1, 0] = 1.0
        return b


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
    def n_vars(self):
        return self.n

    @property
    def n_objs(self):
        return 2
    
    @property
    def n_constraints(self):
        return 1
    
    def _call(self, x):
        rast = rastrigin(x[1:])

        f = np.vstack((
            x[0],
            rast*(1 - x[0]/rast)
        ))

        c = np.cos(self._theta)*(f[1] - self._e) - np.sin(self._theta)*f[0]
        c -= self._a*np.abs(np.sin(self._b*np.pi*(np.sin(self._theta)*(f[1] - self._e) \
             + np.cos(self._theta)*f[0])**self._c))**self._d
        g = np.vstack((c,))
        
        return f, g

    @property
    def var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-5.12, 5.12])).T
        b[0, 0] = 0.0
        b[1, 0] = 1.0
        return b


class CTP2(CTP2_7):
    n: int = 5
    
    def __init__(self, **data):
        super().__init__(-0.2*np.pi, 0.2, 10.0, 1.0, 6.0, 1.0, **data)


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
