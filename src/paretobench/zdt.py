import numpy as np

from .problem import Problem, ProblemWithPF
from .utils import weighted_chunk_sizes


class ZDTx(Problem, ProblemWithPF):
    n: int = 30
    
    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 2
    
    def get_reference(self):
        return "Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of Multiobjective Evolutionary Algorithms: Empirical Results. "\
               "Evolutionary Computation, 8(2), 173–195. https://doi.org/10.1162/106365600568202"


class ZDT1(ZDTx):
    def _call(self, x):
        g = 1 + 9 * np.sum(x[1:], axis=0) / (self.n - 1)
        return np.array([
            x[0],
            g * (1 - np.sqrt(x[0] / g)),
        ])

    @property
    def decision_var_bounds(self):
        return (np.ones((self.n, 2)) * np.array([0, 1])).T

    def get_pareto_front(self, n):
        x = np.linspace(0, 1, n)
        return np.array([x, 1 - np.sqrt(x)])


class ZDT2(ZDTx):
    def _call(self, x):
        g = 1 + 9 * np.sum(x[1:], axis=0) / (self.n - 1)
        return np.array([
            x[0],
            g * (1 - (x[0] / g) ** 2),
        ])

    @property
    def decision_var_bounds(self):
        return (np.ones((self.n, 2)) * np.array([0, 1])).T

    def get_pareto_front(self, n):
        x = np.linspace(0, 1, n)
        return np.array([x, 1 - x**2])
    

class ZDT3(ZDTx):
    def _call(self, x):
        g = 1 + 9 * np.sum(x[1:], axis=0) / (self.n - 1)
        return np.array([
            x[0],
            g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0])),
        ])

    @property
    def decision_var_bounds(self):
        return (np.ones((self.n, 2)) * np.array([0, 1])).T

    def get_pareto_front(self, n):
        # The non-dominated regions from dev_notebooks\zdt3_pareto_front.ipynb
        regs = [
            (0.0, 0.08300152991598536),
            (0.18222872351573624, 0.25776236490626825),
            (0.4093136698086569, 0.453882099086383),
            (0.6183967894392438, 0.6525116988130034),
            (0.8233317933269128, 0.851832862382744)
        ]
        
        # Evaluate along the non-dominated region
        f = []
        for r, my_n in zip(regs, weighted_chunk_sizes(n, [r[1] - r[0] for r in regs])):
            x = np.linspace(*r, my_n)
            f.append([x, 1 - np.sqrt(x) - x*np.sin(10*np.pi*x)])
        return np.concatenate(f, axis=1)


class ZDT4(ZDTx):
    n: int = 10

    def _call(self, x):
        g = 1 + 10 * (self.n - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]), axis=0)
        return np.array([
            x[0],
            g * (1 - np.sqrt(x[0] / g)),
        ])

    @property
    def decision_var_bounds(self):
        b = np.ones((self.n, 2)) * np.array([-5.0, 5.0])
        b[0, 0] = 0.0
        b[0, 1] = 1.0
        return b.T
    
    def get_pareto_front(self, n):
        x = np.linspace(0, 1, n)
        return np.array([x, 1 - np.sqrt(x)])


class ZDT6(ZDTx):
    n: int = 10

    def _call(self, x):
        f1 = 1 - np.exp(-4*x[0])*(np.sin(6*np.pi*x[0]))**6
        g = 1 + 9 * (np.sum(x[1:], axis=0) / (self.n - 1))**0.25
        return np.array([
            f1,
            g * (1 - (f1 / g) ** 2),
        ])

    @property
    def decision_var_bounds(self):
        return (np.ones((self.n, 2)) * np.array([0, 1])).T

    def get_pareto_front(self, n):
        # From dev_notebooks\zdt6_pareto_front.ipynb
        x = np.linspace(0.28077531881538886, 1, n)
        return np.array([x, 1 - np.power(x, 2)])
