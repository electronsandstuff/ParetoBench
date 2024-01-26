import numpy as np

from .problem import Problem
from .utils import rastrigin


class CTPx(Problem):
    def get_reference(self):
        return "Zitzler, E. (Ed.). (2001). Evolutionary multi-criterion optimization: First international conference, "\
               "EMO 2001, Zurich, Switzerland, March 2001: proceedings. Springer."


class CTP1(CTPx):
    def __init__(self, n=5, J=2):
        self.n = int(n)
        self.J = int(J)

        # Calculate the parameters
        j = 0
        self.a = np.ones(J+1)
        self.b = np.ones(J+1)
        delta = 1/(J + 1)
        x = delta
        for i in range(J):
            y = self.a[i]*np.exp(-self.b[i]*x)
            self.a[i+1] = (self.a[i] + y)/2
            self.b[i+1] = -1/x*np.log(y/self.a[i+1])
            x = x + delta
        self.a = self.a[1:]
        self.b = self.b[1:]

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 2
    
    @property
    def n_constraints(self):
        return self.J
    
    def _call(self, x):
        rast = rastrigin(x[1:])
        f = np.vstack((
            x[0],
            rast*np.exp(-x[0]/rast)
        ))

        g = []
        for i in range(self.J):
            g.append(f[1] - self.a[i]*np.exp(-self.b[i]*f[0]))
        return f, np.vstack(g)

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-5.12, 5.12])).T
        b[0, 0] = 0.0
        b[1, 0] = 1.0
        return b


class CTP2_7(CTPx):
    """
    This class is a parent for the problems CTP2 - CTP7 which are just slight variations of one another
    """
    def __init__(self, n, theta, a, b, c, d, e):
        self.n = int(n)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
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

        c = np.cos(self.theta)*(f[1] - self.e) - np.sin(self.theta)*f[0]
        c -= self.a*np.abs(np.sin(self.b*np.pi*(np.sin(self.theta)*(f[1] - self.e) + np.cos(self.theta)*f[0])**self.c))**self.d
        g = np.vstack((c,))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-5.12, 5.12])).T
        b[0, 0] = 0.0
        b[1, 0] = 1.0
        return b


class CTP2(CTP2_7):
    def __init__(self, n=5):
        super().__init__(n, -0.2*np.pi, 0.2, 10.0, 1.0, 6.0, 1.0)


class CTP3(CTP2_7):
    def __init__(self, n=5):
        super().__init__(n, -0.2 * np.pi, 0.1, 10.0, 1.0, 0.5, 1.0)


class CTP4(CTP2_7):
    def __init__(self, n=5):
        super().__init__(n, -0.2 * np.pi, 0.75, 10.0, 1.0, 0.5, 1.0)


class CTP5(CTP2_7):
    def __init__(self, n=5):
        super().__init__(n, -0.2 * np.pi, 0.1, 10.0, 2.0, 0.5, 1.0)


class CTP6(CTP2_7):
    def __init__(self, n=5):
        super().__init__(n, 0.1 * np.pi, 40, 0.5, 1.0, 2.0, -2.0)


class CTP7(CTP2_7):
    def __init__(self, n=5):
        super().__init__(n, -0.05 * np.pi, 40, 5.0, 1.0, 6.0, 0.0)
