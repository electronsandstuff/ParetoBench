import numpy as np
from itertools import count

from .problem import Problem, ProblemWithFixedPF, ProblemWithPF


def triangle_grid_count(n):
    return n*(n+1)//2


def triangle_grid(n):
    # skewed points to help compensate for 
    x = np.concatenate([np.linspace(0, 1 - np.sqrt(i/(n-1)), n-i) for i in range(n)])
    y = np.concatenate([np.ones(n-i)*np.sqrt(i/(n-1)) for i in range(n)])
    return np.vstack((x, y))


def get_pf_cf9_cf10(n, N):
    sub_n = int((np.sqrt(1+4*n*N)-1)/2/N)+1

    # Add "line"
    f2s = [np.linspace(0, 1, sub_n)]
    f3s = [np.sqrt(1 - f2s[-1]**2)]
    f1s = [np.zeros(sub_n)]

    # Add surfaces
    n_triangle = next(n for n in count() if triangle_grid_count(n) > sub_n**2)
    for i in range(1, N+1):
        f1, f3 = triangle_grid(n_triangle)
        start = np.sqrt((2*i-1)/2/N*(1-f3**2))
        end = np.sqrt(2*i/2/N*(1-f3**2))
        f1 = f1/np.maximum(1e-9, 1-f3)*np.maximum(1e-9, end-start) + start
        f2 = np.sqrt(np.maximum(0, 1 - f1**2 - f3**2))
        f1s, f2s, f3s = f1s + [f1], f2s + [f2], f3s + [f3]
    return np.vstack((np.concatenate(f1s), np.concatenate(f2s), np.concatenate(f3s)))


class CF1(Problem, ProblemWithFixedPF):
    def __init__(self, n=10, N=10, a=1.0):
        # Checked - 9/1/2020
        self.n = int(n)
        self.N = int(N)
        self.a = int(a)

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
        j = np.arange(2, self.n + 1)
        summand = (x[1:, :] - np.power(x[:1], 0.5 * (1.0 + 3 * (j[:, None] - 2) / (self.n - 2)))) ** 2
        f = np.vstack((
            x[0] + 2 / j[1::2].size * np.sum(summand[1::2], axis=0),
            1 - x[0] + 2 / j[::2].size * np.sum(summand[::2], axis=0)
        ))
        g = np.vstack((
            f[0] + f[1] - self.a * np.abs(np.sin(self.N * np.pi * (f[0] - f[1] + 1))) - 1,
        ))
        return f, g

    @property
    def decision_var_bounds(self):
        return (np.ones((self.n, 2)) * np.array([0, 1])).T

    def get_pareto_front(self):
        f1 = np.linspace(0, 1, 2*self.N+1)
        f2 = 1 - f1
        return np.vstack((f1, f2))
    

class CF2(Problem, ProblemWithPF):
    def __init__(self, n=10, N=2, a=1.0):
        self.n = int(n)
        self.N = int(N)
        self.a = int(a)

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
        j = np.arange(2, self.n + 1)
        i = j % 2
        summand = (x[1:, :] - np.cos(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n - np.pi / 2 * i[:, None])) ** 2
        f = np.vstack((
            x[0] + 2 / j[1::2].size * np.sum(summand[1::2], axis=0),
            1 - np.sqrt(x[0]) + 2 / j[::2].size * np.sum(summand[::2], axis=0)
        ))
        t = f[1] + np.sqrt(f[0]) - self.a * np.sin(self.N * np.pi * (np.sqrt(f[0]) - f[1] + 1)) - 1
        g = np.vstack((
            t / (1 + np.exp(4 * np.abs(t))),
        ))
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-1, 1])).T
        b[0, 0] = 0
        return b

    def get_pareto_front(self, n):
        ranges = [(((2*i-1)/2/self.N)**2, (i/self.N)**2) for i in range(1, self.N+1)]
        total_range = sum(stop - start for start, stop in ranges)
        f1 = np.concatenate([np.linspace(start, stop, int(n*(stop - start)/total_range + 0.5)) for start, stop in ranges])
        f2 = 1 - np.sqrt(f1)
        return np.vstack((f1, f2))
    
    
class CF3(Problem, ProblemWithPF):
    def __init__(self, n=10, N=2, a=1.0):
        self.n = int(n)
        self.N = int(N)
        self.a = int(a)

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
        # Checked - 9/1/2020
        j = np.arange(2, self.n + 1)
        y = x[1:, :] - np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n)

        summand = y ** 2
        prod = np.cos(20 * y * np.pi / np.sqrt(j[:, None]))

        f = np.vstack((
            x[0] + 2 / j[1::2].size * (
                        4 * np.sum(summand[1::2], axis=0) - 2 * np.prod(prod[1::2], axis=0) + 2),
            1 - x[0] ** 2 + 2 / j[::2].size * (
                        4 * np.sum(summand[::2], axis=0) - 2 * np.prod(prod[::2], axis=0) + 2)
        ))
        g =  np.vstack((
            f[1] + f[0] ** 2 - self.a * np.sin(self.N * np.pi * (f[0] ** 2 - f[1] + 1)) - 1,
        ))
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, 0] = 0
        b[1, 0] = 1
        return b

    def get_pareto_front(self, n):
        ranges = [(np.sqrt((2*i-1)/2/self.N), np.sqrt(i/self.N)) for i in range(1, self.N+1)]
        total_range = sum(stop - start for start, stop in ranges)
        f1 = np.concatenate([np.linspace(start, stop, int(n*(stop - start)/total_range + 0.5)) for start, stop in ranges])
        f2 = 1 - f1**2
        return np.vstack((f1, f2))


class CF4(Problem, ProblemWithPF):
    def __init__(self, n=10):
        self.n = int(n)

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
        # Checked - 9/1/2020
        j = np.arange(2, self.n + 1)
        y = x[1:, :] - np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n)

        summand = y ** 2
        summand[0] = np.abs(y[0])
        ind = y[0] >= 3 / 2 * (1 - np.sqrt(2) / 2)
        summand[0, ind] = 0.125 + (y[0, ind] - 1) ** 2

        f = np.vstack((
            x[0] + np.sum(summand[1::2], axis=0),
            1 - x[0] + np.sum(summand[::2], axis=0)
        ))

        t = x[1] - np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n) - 0.5 * x[0] + 0.25
        g = np.vstack((
            t / (1 + np.exp(4 * np.abs(t))),
        ))
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, 0] = 0
        b[1, 0] = 1
        return b

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = 1 - f1[f1 <= 0.5]
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3/4)] = -f1[np.bitwise_and(f1 > 0.5, f1 <= 3/4)]/2 + 3/4
        f2[f1 > 3/4] = 1 - f1[f1 > 3/4] + 1/8
        return np.vstack((f1, f2))
    

class CF5(Problem, ProblemWithPF):
    def __init__(self, n=10):
        self.n = int(n)

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
        j = np.arange(2, self.n + 1)
        i = j % 2
        y = x[1:, :] - 0.8 * x[:1] * np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n + np.pi / 2 * i[:, None])

        summand = 2.0 * y ** 2 - np.cos(4.0 * np.pi * y) + 1.0
        summand[0] = np.abs(y[0])
        ind = y[0] >= 3 / 2 * (1 - np.sqrt(2) / 2)
        summand[0, ind] = 0.125 + (y[0, ind] - 1) ** 2

        f = np.vstack((
            x[0] + np.sum(summand[1::2], axis=0),
            1 - x[0] + np.sum(summand[::2], axis=0)
        ))

        g = np.vstack((
            x[1] - 0.8 * x[0] * np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n) - 0.5 * x[0] + 0.25,
        ))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, 0] = 0
        b[1, 0] = 1
        return b

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = 1 - f1[f1 <= 0.5]
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3/4)] = -f1[np.bitwise_and(f1 > 0.5, f1 <= 3/4)]/2 + 3/4
        f2[f1 > 3/4] = 1 - f1[f1 > 3/4] + 1/8
        return np.vstack((f1, f2))
    

class CF6(Problem, ProblemWithPF):
    def __init__(self, n=10):
        self.n = int(n)

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 2
    
    @property
    def n_constraints(self):
        return 2
    
    def _call(self, x):
        j = np.arange(2, self.n + 1)
        i = j % 2
        y = x[1:, :] - 0.8 * x[:1] * np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n + np.pi / 2 * i[:, None])

        f = np.vstack((
            x[0] + np.sum(y[1::2] ** 2, axis=0),
            (1 - x[0]) ** 2 + np.sum(y[::2] ** 2, axis=0)
        ))
        
        g1 = x[1] - 0.8 * x[0] * np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n) - np.sign(
            0.5 * (1 - x[0]) - (1 - x[0]) ** 2) * np.sqrt(np.abs(0.5 * (1 - x[0]) - (1 - x[0]) ** 2))
        g2 = x[3] - 0.8 * x[0] * np.sin(6 * np.pi * x[0] + 4 * np.pi / self.n) - np.sign(
            0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0])) * np.sqrt(np.abs(0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0])))
        g = np.vstack((
            g1,
            g2
        ))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, 0] = 0
        b[1, 0] = 1
        return b

    def get_pareto_front(self, n):
        f1 = np.linspace(0, 1, n)
        f2 = np.empty_like(f1)
        f2[f1 <= 0.5] = (1 - f1[f1 <= 0.5])**2
        f2[np.bitwise_and(f1 > 0.5, f1 <= 3/4)] = (1-f1[np.bitwise_and(f1 > 0.5, f1 <= 3/4)])/2
        f2[f1 > 3/4] = np.sqrt(1 - f1[f1 > 3/4])/4
        return np.vstack((f1, f2))
    

class CF7(Problem, ProblemWithPF):
    def __init__(self, n=10):
        self.n = int(n)

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 2
    
    @property
    def n_constraints(self):
        return 2
    
    def _call(self, x):
        j = np.arange(2, self.n + 1)
        i = j % 2
        y = x[1:, :] - np.sin(6 * np.pi * x[:1] + j[:, None] * np.pi / self.n + np.pi / 2 * i[:, None])
        h = 2 * y ** 2 - np.cos(4 * np.pi * y) + 1
        h[0] = y[0] ** 2
        h[2] = y[2] ** 2

        f = np.vstack((
            x[0] + np.sum(h[1::2], axis=0),
            (1 - x[0]) ** 2 + np.sum(h[::2], axis=0)
        ))

        g1 = x[1] - np.sin(6 * np.pi * x[0] + 2 * np.pi / self.n) - np.sign(
            0.5 * (1 - x[0]) - (1 - x[0]) ** 2) * np.sqrt(np.abs(0.5 * (1 - x[0]) - (1 - x[0]) ** 2))
        g2 = x[3] - np.sin(6 * np.pi * x[0] + 4 * np.pi / self.n) - np.sign(
            0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0])) * np.sqrt(np.abs(0.25 * np.sqrt(1 - x[0]) - 0.5 * (1 - x[0])))
        g = np.vstack((
            g1,
            g2
        ))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, 0] = 0
        b[1, 0] = 1
        return b

    def get_pareto_front(self, n):
            f1 = np.linspace(0, 1, n)
            f2 = np.empty_like(f1)
            f2[f1 <= 0.5] = (1 - f1[f1 <= 0.5])**2
            f2[np.bitwise_and(f1 > 0.5, f1 <= 3/4)] = (1-f1[np.bitwise_and(f1 > 0.5, f1 <= 3/4)])/2
            f2[f1 > 3/4] = np.sqrt(1 - f1[f1 > 3/4])/4
            return np.vstack((f1, f2))


class CF8(Problem, ProblemWithPF):
    def __init__(self, n=10, a=4, N=2):
        self.n = int(n)
        self.a = int(a)
        self.N = int(N)

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 3
    
    @property
    def n_constraints(self):
        return 1
    
    def _call(self, x):
        j = np.arange(3, self.n + 1)
        summand = (x[2:] - 2 * x[1][None, :] * np.sin(2 * np.pi * x[0][None, :] + j[:, None] * np.pi / self.n)) ** 2

        f = np.vstack((
            np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi) + 2 / j[1::3].size * np.sum(
                summand[1::3], axis=0),
            np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi) + 2 / j[2::3].size * np.sum(
                summand[2::3], axis=0),
            np.sin(0.5 * x[0] * np.pi) + 2 / j[::3].size * np.sum(summand[::3], axis=0),
        ))

        g = np.vstack((
            (f[0] ** 2 + f[1] ** 2) / (1 - f[2] ** 2) - self.a * np.abs(
                np.sin(self.N * np.pi * ((f[0] ** 2 - f[1] ** 2) / (1 - f[2] ** 2) + 1))) - 1,
        ))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-4, 4])).T
        b[0, :2] = 0
        b[1, :2] = 1
        return b

    def get_pareto_front(self, n):
        sub_n = n // (2*self.N + 1)
        f3 = np.repeat(np.linspace(0, 1, sub_n), 2*self.N + 1)
        f1 = np.concatenate([np.sqrt(i/2/self.N*(1-f3[:sub_n]**2)) for i in range(2*self.N+1)])
        f2 = np.sqrt(np.maximum(0, 1 - f1**2 - f3**2))
        return np.vstack((f1, f2, f3))


class CF9(Problem, ProblemWithPF):
    def __init__(self, n=10, a=3, N=2):
        self.n = int(n)
        self.a = int(a)
        self.N = int(N)

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 3
    
    @property
    def n_constraints(self):
        return 1
    
    def _call(self, x):
        j = np.arange(3, self.n + 1)
        summand = (x[2:] - 2 * x[1][None, :] * np.sin(2 * np.pi * x[0][None, :] + j[:, None] * np.pi / self.n)) ** 2

        f = np.vstack((
            np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi) + 2 / j[1::3].size * np.sum(
                summand[1::3], axis=0),
            np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi) + 2 / j[2::3].size * np.sum(
                summand[2::3], axis=0),
            np.sin(0.5 * x[0] * np.pi) + 2 / j[::3].size * np.sum(summand[::3], axis=0),
        ))

        g = np.vstack((
            (f[0] ** 2 + f[1] ** 2) / (1 - f[2] ** 2) - self.a * np.sin(
                self.N * np.pi * ((f[0] ** 2 - f[1] ** 2) / (1 - f[2] ** 2) + 1)) - 1,
        ))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, :2] = 0
        b[1, :2] = 1
        return b

    def get_pareto_front(self, n):
        return get_pf_cf9_cf10(n, self.N)
    

class CF10(Problem, ProblemWithPF):
    def __init__(self, n=10, a=1, N=2):
        self.n = int(n)
        self.a = int(a)
        self.N = int(N)

    @property
    def n_decision_vars(self):
        return self.n

    @property
    def n_objectives(self):
        return 3
    
    @property
    def n_constraints(self):
        return 1
    
    def _call(self, x):
        j = np.arange(3, self.n + 1)
        y = x[2:] - 2 * x[1][None, :] * np.sin(2 * np.pi * x[0][None, :] + j[:, None] * np.pi / self.n)
        summand = 4 * y ** 2 - np.cos(8 * np.pi * y) + 1

        f = np.vstack((
            np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi) + 2 / j[1::3].size * np.sum(
                summand[1::3], axis=0),
            np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi) + 2 / j[1::3].size * np.sum(
                summand[2::3], axis=0),
            np.sin(0.5 * x[0] * np.pi) + 2 / j[::3].size * np.sum(summand[1::3], axis=0),
        ))

        g = np.vstack((
            (f[0] ** 2 + f[1] ** 2) / (1 - f[2] ** 2) - self.a * np.sin(
                self.N * np.pi * ((f[0] ** 2 - f[1] ** 2) / (1 - f[2] ** 2) + 1)) - 1,
        ))
        
        return f, g

    @property
    def decision_var_bounds(self):
        b = (np.ones((self.n, 2)) * np.array([-2, 2])).T
        b[0, :2] = 0
        b[1, :2] = 1
        return b

    def get_pareto_front(self, n):
        return get_pf_cf9_cf10(n, self.N)
