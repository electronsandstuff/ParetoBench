import numpy as np

from .problem import Problem


class SCH(Problem):
    """
    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic
    algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

    Description: Convex

    Other name: Schaffer function N. 1
    """
    @property
    def n_decision_vars(self):
        return 1

    @property
    def n_objectives(self):
        return 2
    
    def _call(self, x):
        return np.vstack((
            x[0] ** 2,
            (x[0] - 2) ** 2
        ))

    @property
    def decision_var_bounds(self):
        return np.array([
            [-1e3, ], [1e3, ],
        ])


class FON(Problem):
    """
    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic
    algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

    Description: non-convex

    Other name: Fonseca-Fleming Function
    """
    @property
    def n_decision_vars(self):
        return 3

    @property
    def n_objectives(self):
        return 2
    
    def _call(self, x):
        return np.array([
            1 - np.exp(-(x[0] - 1 / np.sqrt(3)) ** 2 - (x[1] - 1 / np.sqrt(3)) ** 2 - (x[2] - 1 / np.sqrt(3)) ** 2),
            1 - np.exp(-(x[0] + 1 / np.sqrt(3)) ** 2 - (x[1] + 1 / np.sqrt(3)) ** 2 - (x[2] + 1 / np.sqrt(3)) ** 2),
        ])

    @property
    def decision_var_bounds(self):
        return np.array([
            [-4.0, -4.0, -4.0],
            [4.0, 4.0, 4.0],
        ])


class POL(Problem):
    """
    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic
    algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

    Description: non-convex, disconnected

    Other name: Poloni’s two objective function
    """
    @property
    def n_decision_vars(self):
        return 2

    @property
    def n_objectives(self):
        return 2
    
    def _call(self, x):
        a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
        a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
        b1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
        b2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
        return np.array([
            1 + (a1 - b1) ** 2 + (a2 - b2) ** 2,
            (x[0] + 3) ** 2 + (x[1] + 1) ** 2,
        ])

    @property
    def decision_var_bounds(self):
        return np.array([
            [-np.pi, -np.pi],
            [np.pi, np.pi],
        ])


class KUR(Problem):
    """
    Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic
    algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

    Description: non-convex

    Other name: Kursawe’s Function
    """
    def __init__(self, n=3):
        self.n = int(n)
        
    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 2

    def _call(self, x):
        return np.array([
            np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1] ** 2 + x[1:] ** 2)), axis=0),
            np.sum(np.abs(x) ** 0.8 + 5 * np.sin(x ** 3), axis=0),
        ])

    @property
    def decision_var_bounds(self):
        return (np.ones((self.n, 2)) * np.array([-5, 5])).T


class CONSTR(Problem):
    @property
    def n_decision_vars(self):
        return 2

    @property
    def n_objectives(self):
        return 2
    
    @property
    def n_constraints(self):
        return 2
    
    def _call(self, x):
        f = np.array([
            x[0],
            (1 + x[1]) / x[0]
        ])
        g = np.array([
            x[1] + 9 * x[0] - 6,
            -x[1] + 9 * x[0] - 1
        ])
        return f, g

    @property
    def decision_var_bounds(self):
        return np.array([
            [0.1, 0.0],
            [1.0, 5.0]
        ])


class SRN(Problem):
    @property
    def n_decision_vars(self):
        return 2

    @property
    def n_objectives(self):
        return 2
    
    @property
    def n_constraints(self):
        return 2
    
    def _call(self, x):
        f = np.array([
            (x[0] - 2) ** 2 + (x[1] - 1) ** 2 + 2,
            9 * x[0] - (x[1] - 1) ** 2
        ])
        g = np.array([
            225 - (x[0] ** 2 + x[1] ** 2),
            -10 - (x[0] - 3 * x[1])
        ])
        return f, g

    @property
    def decision_var_bounds(self):
        return np.array([
            [-20.0, -20.0],
            [20.0, 20.0]
        ])


class TNK(Problem):
    @property
    def n_decision_vars(self):
        return 2

    @property
    def n_objectives(self):
        return 2
    
    @property
    def n_constraints(self):
        return 2
    
    def _call(self, x):
        f = np.array([
            x[0],
            x[1]
        ])
        g = np.array([
            -(-x[0] ** 2 - x[1] ** 2 + 1 + 0.1 * np.cos(16 * np.arctan(x[0] / x[1]))),
            0.5 - ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        ])
        return f, g

    @property
    def decision_var_bounds(self):
        return np.array([
            [0.0, 0.0],
            [np.pi, np.pi]
        ])


class WATER(Problem):
    @property
    def n_decision_vars(self):
        return 3

    @property
    def n_objectives(self):
        return 5
    
    @property
    def n_constraints(self):
        return 7
    
    def _call(self, x):
        f = np.array([
            106780.37 * (x[1] + x[2]) + 61704.67,
            3000.0 * x[0],
            305700 * 2289 * x[1] / (0.06 * 2289) ** 0.65,
            250 * 2289 * np.exp(-39.75 * x[1] + 9.9 * x[2] + 2.74),
            25 * (1.39 / (x[0] * x[1]) + 4940 * x[2] - 80)
        ])
        g = np.array([
            1.0 - (0.00139 / (x[0] * x[1]) + 4.94 * x[2] - 0.08),
            1.0 - (0.000306 / (x[0] * x[1]) + 1.082 * x[2] - 0.0986),
            50000.0 - (12.307 / (x[0] * x[1]) + 49408.25 * x[2] + 4051.02),
            16000.0 - (2.098 / (x[0] * x[1]) + 8046.33 * x[2] - 696.71),
            10000.0 - (2.138 / (x[0] * x[1]) + 7883.39 * x[2] - 705.04),
            2000.0 - (0.417 / (x[0] * x[1]) + 1721.26 * x[2] - 136.54),
            550.0 - (0.164 / (x[0] * x[1]) + 631.13 * x[2] - 54.48)
        ]) 
        return f, g

    @property
    def decision_var_bounds(self):
        return np.array([
            [0.01, 0.01, 0.01],
            [0.45, 0.10, 0.10]
        ])
