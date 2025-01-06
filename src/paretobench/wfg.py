import numpy as np

from .problem import Problem, ProblemWithPF
from .utils import get_hyperplane_points, get_nondominated_inds
from .wfg_utils import (
    shape_linear,
    shape_convex,
    shape_concave,
    shape_mixed,
    shape_disconnected,
    compute_x,
    transform_bias_polynomial,
    transform_bias_flat_region,
    transform_bias_parameter_dependent,
    transform_shift_linear,
    transform_shift_deceptive,
    transform_shift_multimodal,
    transform_reduction_weighted_sum,
    transform_reduction_non_separable,
)
from .containers import Population


class WFGx(Problem, ProblemWithPF):
    n: int = 24
    k: int = 4
    m: int = 2

    def __init__(self, **data):
        # Handle pydantic data
        super().__init__(**data)

        # Set internal attributes for later
        self._d = 1.0
        self._s = np.array([2 * (i + 1) for i in range(self.m)])

    @property
    def var_lower_bounds(self):
        return np.zeros(self.n)

    @property
    def var_upper_bounds(self):
        return np.array([2 * (i + 1) for i in range(self.n)])

    @property
    def reference(self):
        return (
            "Huband, S., Hingston, P., Barone, L., & While, L. (2006). A review of multiobjective test problems and a scalable test "
            "problem toolkit. IEEE Transactions on Evolutionary Computation, 10(5), 477–506. https://doi.org/10.1109/TEVC.2005.861417"
        )


class WFG1(WFGx):
    def _call(self, x, set_pos_zero=False):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.copy(y)
        t1[self.k :] = transform_shift_linear(y[self.k :], 0.35)

        # Transition 2
        t2 = np.copy(t1)
        t2[self.k :] = transform_bias_flat_region(t1[self.k :], 0.8, 0.75, 0.85)
        if set_pos_zero:  # Because of some floating point round-off error issues
            t2[self.k :] = 0.0

        # Transition 3
        t3 = transform_bias_polynomial(np.clip(t2, 0.0, 100.0), 0.02)

        # Transition 4
        w = b
        t4 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t4.append(transform_reduction_weighted_sum(t3[start:end, :], w[start:end]))
        t4.append(transform_reduction_weighted_sum(t3[self.k :, :], w[self.k :]))
        t4 = np.vstack(t4)

        # Find x
        new_x = compute_x(t4, False)

        # Shape functions
        h = shape_convex(new_x)
        h[-1] = shape_mixed(new_x, 5, 1)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n, n_solver_guess=128, n_solver_iter=32):
        # Invert the convex shape function to find points along the reference vectors
        # f1 = (1-cos(x0))(1-cos(x1))(1-cos(x2))...(1-cos(x{n-1}))
        # fi = (1-cos(x0))(1-cos(x1))(1-cos(x2))...(1-sin(x{n-i+1}))
        # fm = (1-sin(x0))
        # Note that: fi/f1 = (1-sin(x{n-i+1}))/(1-cos(x{n-i+1}))  /  ((1-cos(x))(1-cos(x))... for remaining terms after n-i+1
        # Use (1-sin(x))/(1-cos(x)) = y is solved by x = 2*arctan((sqrt(2y) - 1)/(2y-1))
        r = get_hyperplane_points(self.m, n) + 1e-12  # Hack to avoid some numerical issues
        theta = np.zeros_like(r)
        for j in range(1, r.shape[0]):
            t = r[j] * np.prod(1 - np.cos(theta[-j:-1]), axis=0)
            theta[-j - 1] = 2 * np.arctan2((np.sqrt(2 * r[0] * t) - r[0]), (2 * t - r[0])) % (np.pi)
        x = theta / np.pi * 2

        # Solve for the value of x0 which gives the correct fm/f{m-1} where fm is shape_mixed(...) and f{m-1} is from shape_convex(...)
        t = r[-1] / r[-2] * (1 - np.sin(np.pi / 2 * x[1]))

        def solve_fun(x):
            return x + np.cos(10 * np.pi * x + np.pi / 2) / 10 / np.pi - 1 + t[None, :] * (1 - np.cos(np.pi / 2 * x))

        def solve_fun_der(x):
            return 1 - np.sin(10 * np.pi * x + np.pi / 2) + np.pi / 2 * t[None, :] * np.sin(np.pi / 2 * x)

        # Make an initial guess
        x_guess = np.linspace(0, 1, n_solver_guess)
        x[0] = x_guess[np.argmin(np.abs(solve_fun(x_guess[:, None])), axis=0)]

        # Newton Raphson iterations
        for _ in range(n_solver_iter):
            fun_prime = solve_fun_der(x[0][None, :])
            fun_prime[fun_prime == 0.0] = 1e6  # Avoid divide by zero
            x[0] = np.clip(x[0] - solve_fun(x[0][None, :]) / fun_prime, 0, 1)

        # Evaluate the objective functions
        f = shape_convex(x)
        f[-1] = shape_mixed(x, 5, 1)
        return (f * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG2(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.copy(y)
        t1[self.k :] = np.clip(transform_shift_linear(y[self.k :], 0.35), 0.0, 1.0)
        # Transition 2
        t2 = np.empty((self.k + (self.n - self.k) // 2, t1.shape[1]))
        t2[: self.k] = t1[: self.k]
        for i in range(self.k + 1, self.k + (self.n - self.k) // 2 + 1):
            # Calculate indices
            idx1 = self.k + 2 * (i - self.k) - 2
            idx2 = self.k + 2 * (i - self.k) - 1

            # Stack the transformed values
            values = np.vstack((t1[idx1], t1[idx2]))

            # Apply transformation and clipping
            t2[i - 1] = np.clip(
                transform_reduction_non_separable(values, 2),
                0.0,
                1.0,
            )

        # Transition 3
        t3 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t3.append(np.clip(transform_reduction_weighted_sum(t2[start:end, :]), 0.0, 1.0))
        t3.append(np.clip(transform_reduction_weighted_sum(t2[self.k :, :]), 0.0, 1.0))
        t3 = np.vstack(t3)

        # Find x
        new_x = compute_x(t3, False)

        # Shape functions
        h = np.empty((self.m, x.shape[1]))
        h[:-1, :] = shape_convex(new_x)[:-1, :]
        h[-1] = shape_disconnected(new_x, 5, 1.0, 1.0)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n, n_solver_guess=2048, n_solver_iter=32):
        # Invert the convex shape function to find points along the reference vectors
        # f1 = (1-cos(x0))(1-cos(x1))(1-cos(x2))...(1-cos(x{n-1}))
        # fi = (1-cos(x0))(1-cos(x1))(1-cos(x2))...(1-sin(x{n-i+1}))
        # fm = (1-sin(x0))
        # Note that: fi/f1 = (1-sin(x{n-i+1}))/(1-cos(x{n-i+1}))  /  ((1-cos(x))(1-cos(x))... for remaining terms after n-i+1
        # Use (1-sin(x))/(1-cos(x)) = y is solved by x = 2*arctan((sqrt(2y) - 1)/(2y-1))
        r = get_hyperplane_points(self.m, 2 * n) + 1e-12  # Hack to avoid some numerical issues
        theta = np.zeros_like(r)
        for j in range(1, r.shape[0]):
            t = r[j] * np.prod(1 - np.cos(theta[-j:-1]), axis=0)
            theta[-j - 1] = 2 * np.arctan2((np.sqrt(2 * r[0] * t) - r[0]), (2 * t - r[0])) % (np.pi)
        x = theta / np.pi * 2

        # Solve for the value of x0 which gives the correct fm/f{m-1} where fm is shape_mixed(...) and f{m-1} is from shape_convex(...)
        t = r[-1] / r[-2] * (1 - np.sin(np.pi / 2 * x[1]))

        def solve_fun(x):
            return x * np.cos(5 * np.pi * x) ** 2 - 1 + t[None, :] * (1 - np.cos(np.pi / 2 * x))

        def solve_fun_der(x):
            return (
                np.cos(5 * np.pi * x) ** 2
                - 10 * np.pi * x * np.cos(5 * np.pi * x) * np.sin(5 * np.pi * x)
                + np.pi / 2 * t[None, :] * np.sin(np.pi / 2 * x)
            )

        # Make an initial guess (biased toward small x)
        x_guess = np.linspace(0, 1, n_solver_guess)
        x[0] = x_guess[np.min(np.argsort(np.abs(solve_fun(x_guess[:, None])), axis=0)[:8], axis=0)]

        # Newton Raphson iterations
        for _ in range(n_solver_iter):
            fun_prime = solve_fun_der(x[0][None, :])
            fun_prime[fun_prime == 0.0] = 1e6  # Avoid divide by zero
            x[0] = np.clip(x[0] - solve_fun(x[0][None, :]) / fun_prime, 0, 1)

        # Evaluate the objective functions
        f = shape_convex(x)
        f[-1] = shape_disconnected(x, 5, 1, 1)
        f = f[:, get_nondominated_inds(f.T)]
        return (f * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG3(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.copy(y)
        t1[self.k :] = np.clip(transform_shift_linear(y[self.k :], 0.35), 0.0, 1.0)

        # Transition 2
        t2 = np.empty((self.k + (self.n - self.k) // 2, t1.shape[1]))
        t2[: self.k] = t1[: self.k]
        for i in range(self.k + 1, self.k + (self.n - self.k) // 2 + 1):
            # Calculate indices
            idx1 = self.k + 2 * (i - self.k) - 2
            idx2 = self.k + 2 * (i - self.k) - 1

            # Stack the transformed values
            values = np.vstack((t1[idx1], t1[idx2]))

            # Apply transformation and clipping
            t2[i - 1] = np.clip(
                transform_reduction_non_separable(values, 2),
                0.0,
                1.0,
            )

        # Transition 3
        t3 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t3.append(np.clip(transform_reduction_weighted_sum(t2[start:end, :]), 0.0, 1.0))
        t3.append(np.clip(transform_reduction_weighted_sum(t2[self.k :, :]), 0.0, 1.0))
        t3 = np.vstack(t3)

        # Find x
        new_x = compute_x(t3, True)

        # Shape functions
        h = shape_linear(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = np.vstack((np.linspace(0, 1, n), np.full((self.m - 2, n), 1 / 2), np.zeros((1, n))))
        f = shape_linear(f)
        return (f * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG4(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.clip(transform_shift_multimodal(y, 30, 10, 0.35), 0.0, 1.0)

        # Transition 2
        t2 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t2.append(np.clip(transform_reduction_weighted_sum(t1[start:end, :]), 0.0, 1.0))
        t2.append(np.clip(transform_reduction_weighted_sum(t1[self.k :, :]), 0.0, 1.0))
        t2 = np.vstack(t2)

        # Find x
        new_x = compute_x(t2, True)

        # Shape functions
        h = shape_concave(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0)) * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG5(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.clip(transform_shift_deceptive(y, 0.35, 0.001, 0.05), 0.0, 1.0)

        # Transition 2
        t2 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t2.append(np.clip(transform_reduction_weighted_sum(t1[start:end, :]), 0.0, 1.0))
        t2.append(np.clip(transform_reduction_weighted_sum(t1[self.k :, :]), 0.0, 1.0))
        t2 = np.vstack(t2)

        # Find x
        new_x = compute_x(t2, True)

        # Shape functions
        h = shape_concave(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0)) * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG6(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1 - checked
        t1 = np.copy(y)
        t1[self.k :] = transform_shift_linear(y[self.k :], 0.35)

        # Transition 2
        gap = self.k // (self.m - 1)
        t2 = [transform_reduction_non_separable(t1[(m - 1) * gap : (m * gap), :], gap) for m in range(1, self.m)]
        t2.append(transform_reduction_non_separable(t1[self.k :, :], self.n - self.k))
        t2 = np.vstack(t2)

        # Find x
        new_x = compute_x(t2, True)

        # Shape functions
        h = shape_concave(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0)) * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG7(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.copy(y)
        for i in range(1, self.k + 1):
            weighted_sum = transform_reduction_weighted_sum(y[i:, :])
            t1[i - 1] = np.clip(
                transform_bias_parameter_dependent(y[i - 1], weighted_sum, 0.98 / 49.98, 0.02, 50), 0.0, 1.0
            )

        # Transition 2
        t2 = np.copy(t1)
        t2[self.k :] = np.clip(transform_shift_linear(t1[self.k :], 0.35), 0.0, 1.0)

        # Transition 3
        t3 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t3.append(np.clip(transform_reduction_weighted_sum(t2[start:end, :]), 0.0, 1.0))
        t3.append(np.clip(transform_reduction_weighted_sum(t2[self.k :, :]), 0.0, 1.0))
        t3 = np.vstack(t3)

        # Find x
        new_x = compute_x(t3, True)

        # Shape functions
        h = shape_concave(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0)) * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG8(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.copy(y)
        for i in range(self.k + 1, self.n + 1):
            weighted_sum = transform_reduction_weighted_sum(y[: i - 1])
            t1[i - 1] = np.clip(
                transform_bias_parameter_dependent(y[i - 1][None, :], weighted_sum, 0.98 / 49.98, 0.02, 50),
                0.0,
                1.0,
            )

        # Transition 2
        t2 = np.copy(t1)
        t2[self.k :] = np.clip(transform_shift_linear(t2[self.k :], 0.35), 0.0, 1.0)

        # Transition 3
        t3 = []
        for i in range(1, self.m):
            start = int((i - 1) * self.k / (self.m - 1))
            end = int(i * self.k / (self.m - 1))
            t3.append(np.clip(transform_reduction_weighted_sum(t2[start:end, :]), 0.0, 1.0))
        t3.append(np.clip(transform_reduction_weighted_sum(t2[self.k :, :]), 0.0, 1.0))
        t3 = np.vstack(t3)

        # Find x
        new_x = compute_x(t3, True)

        # Shape functions
        h = shape_concave(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0)) * 2 * np.arange(1, self.m + 1)[:, None]).T


class WFG9(WFGx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        # Normalize the input
        b = np.array([2 * (i + 1) for i in range(self.n)])
        y = x / b[:, None]

        # Transition 1
        t1 = np.copy(y)
        for i in range(1, self.n):
            weighted_sum = transform_reduction_weighted_sum(y[i:])
            t1[i - 1] = np.clip(
                transform_bias_parameter_dependent(y[i - 1][None, :], weighted_sum, 0.98 / 49.98, 0.02, 50),
                0.0,
                1.0,
            )

        # Transition 2
        t2 = np.empty_like(t1)
        t2[: self.k] = np.clip(transform_shift_deceptive(t1[: self.k], 0.35, 0.001, 0.05), 0.0, 1.0)
        t2[self.k :] = np.clip(transform_shift_multimodal(t1[self.k :], 30, 95, 0.35), 0.0, 1.0)

        # Transition 3
        gap = self.k // (self.m - 1)
        t3 = [transform_reduction_non_separable(t2[(m - 1) * gap : (m * gap), :], gap) for m in range(1, self.m)]
        t3.append(transform_reduction_non_separable(t2[self.k :, :], self.n - self.k))
        t3 = np.vstack(t3)

        # Find x
        new_x = compute_x(t3, True)

        # Shape functions
        h = shape_concave(new_x)

        # Compute the objective
        obj = (self._d * new_x[-1])[None, :] + self._s[:, None] * h

        # Return it
        return Population(f=obj.T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0)) * 2 * np.arange(1, self.m + 1)[:, None]).T
