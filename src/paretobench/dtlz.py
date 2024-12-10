import numpy as np

from .problem import Problem, ProblemWithPF
from .utils import get_hyperplane_points, uniform_grid
from .dtlz_utils import g_1_3, g_2_4_5, theta_5_6, f_2_to_6
from .containers import Population


class DTLZx(Problem, ProblemWithPF):
    m: int = 3
    n: int = 10

    @property
    def var_lower_bounds(self):
        return np.zeros(self.n)

    @property
    def var_upper_bounds(self):
        return np.ones(self.n)

    @property
    def reference(self):
        return (
            "Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002). Scalable multi-objective optimization test problems. "
            "Proceedings of the 2002 Congress on Evolutionary Computation. CEC’02 (Cat. No.02TH8600), 1, 825–830 vol.1. "
            "https://doi.org/10.1109/CEC.2002.1007032"
        )

    def _call(self, x):
        raise NotImplementedError()


class DTLZ1(DTLZx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        g = g_1_3(x[self.m - 1 :, :], self.n - self.m + 1)
        f1 = np.vstack([np.prod(x[: self.m - 1 - i, :], axis=0) for i in range(0, self.m)])
        f2 = np.vstack([np.ones(x.shape[1])] + [1 - x[self.m - 1 - i, :] for i in range(1, self.m)])
        return Population(f=((1 + g) * f1 * f2 / 2).T)

    def get_pareto_front(self, n):
        return (get_hyperplane_points(self.m, n) / 2).T


class DTLZ2(DTLZx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        return Population(f=((1 + g_2_4_5(x[self.m - 1 :, :])) * f_2_to_6(x[: self.m - 1, :], self.m)).T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0))).T


class DTLZ3(DTLZx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        return Population(
            f=((1 + g_1_3(x[self.m - 1 :, :], self.n - self.m + 1)) * f_2_to_6(x[: self.m - 1, :], self.m)).T
        )

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0))).T


class DTLZ4(DTLZx):
    alpha: int = 100

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        return Population(
            f=((1 + g_2_4_5(x[self.m - 1 :, :])) * f_2_to_6(x[: self.m - 1, :], self.m, alpha=self.alpha)).T
        )

    def get_pareto_front(self, n):
        f = get_hyperplane_points(self.m, n)
        return (f / np.sqrt(np.sum(f**2, axis=0))).T


class DTLZ5(DTLZx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        g = g_2_4_5(x[self.m - 1 :, :])
        return Population(f=((1 + g) * f_2_to_6(theta_5_6(x, g, self.m), self.m)).T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(2, n)
        f = f / np.sqrt(np.sum(f**2, axis=0))[None, :]
        f = np.concatenate((np.repeat(f[:1], self.m - 1, axis=0), f[1:]), axis=0)
        return (
            f
            / np.power(
                np.sqrt(2),
                np.concatenate(([self.m - 2], np.arange(self.m - 2, -1.0, -1.0))),
            )[:, None]
        ).T


class DTLZ6(DTLZx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        g = np.sum(x[self.m - 1 :, :] ** 0.1, axis=0)
        return Population(f=((1 + g) * f_2_to_6(theta_5_6(x, g, self.m), self.m)).T)

    def get_pareto_front(self, n):
        f = get_hyperplane_points(2, n)
        f = f / np.sqrt(np.sum(f**2, axis=0))[None, :]
        f = np.concatenate((np.repeat(f[:1], self.m - 1, axis=0), f[1:]), axis=0)
        return (
            f
            / np.power(
                np.sqrt(2),
                np.concatenate(([self.m - 2], np.arange(self.m - 2, -1.0, -1.0))),
            )[:, None]
        ).T


class DTLZ7(DTLZx):
    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f1 = np.copy(x[: self.m - 1, :])
        f2 = 2 + 9 * np.sum(x[self.m - 1 :, :], axis=0) / (self.n - self.m + 1)
        f2 *= self.m - np.sum((1 + np.sin(3 * np.pi * f1)) * f1 / (1 + f2[None, :]), axis=0)
        return Population(f=np.vstack([f1, f2]).T)

    def get_pareto_front(self, n):
        # Break first m-1 dimensions into non-dominated chunks
        regs = [(0.0, 0.25141183661715344), (0.6316265267192559, 0.8594008516924949)]
        mid = (regs[0][1] - regs[0][0]) / (regs[1][1] - regs[1][0] + regs[0][1] - regs[0][0])
        x = uniform_grid(n, self.m - 1)
        x[x <= mid] = x[x <= mid] * (regs[0][1] - regs[0][0]) / mid + regs[0][0]
        x[x > mid] = (x[x > mid] - mid) * (regs[1][1] - regs[1][0]) / (1 - mid) + regs[1][0]

        # Compute final objective
        return np.concatenate(
            (
                x,
                2 * (self.m - np.sum(x / 2 * (1 + np.sin(3 * np.pi * x)), axis=0))[None, :],
            ),
            axis=0,
        ).T


class DTLZ8(DTLZx):
    @property
    def n_constraints(self):
        return self.m

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f = np.vstack(
            [
                np.mean(
                    x[np.floor(j * self.n / self.m).astype(int) : np.floor((j + 1) * self.n / self.m).astype(int)],
                    axis=0,
                )
                for j in range(self.m)
            ]
        )

        # First m-1 constraints
        g = [f[-1] + 4 * f[j] - 1 for j in range(self.m - 1)]

        # Add last constraint
        fsum = f[:-1, None, :] + f[None, :-1, :]
        fsum[np.diag_indices(2), :] = np.inf  # For i != j
        g = np.vstack(g + [2 * f[-1] + np.min(np.min(fsum, axis=0), axis=0) - 1])
        return Population(f=f.T, g=g.T)

    def get_pareto_front(self, n):
        # Break points into the "pole" feature and the lower PF. Based on dimension, number in
        # pole should scale like the square root of number in lower plane
        npole = int(1.33 * np.ceil(np.sqrt(1 + 4 * n) / 2 - 1 / 2))
        nlower = n - npole

        # Can build PF from lower 3D subspace. Our last constraint here is satisfied by the
        # hyperplane x + y + 2z = 1. Then, use the boundaries, 4x + z = 1 and 4y + z = 1. Projecting
        # Onto the xy-plane, this gives us the intersection of the lines y=7x-1, y=x/7+1/7, y=1-x.
        # The hyperplane in between the intersection of these three curves and along x + y + 2z = 1
        # should give us the "lower" part of the PF. These points are (1/6, 1/6, 2/3), (0.25, 0.75, 0),
        # and (0.75, 0.25, 0).
        # Find the affine transform [(0, 0), (1, 0), (0, 1)] -> [(1/6, 1/6), (3/4, 1/4), (1/4, 3/4)] and
        # apply to first two dimensions then rescale the last dimension.
        temp = get_hyperplane_points(3, int(np.ceil(2 * nlower / (self.m - 1))))
        temp[0], temp[1] = (
            temp[0] * (1 / 4 - 1 / 6) + temp[1] * (3 / 4 - 1 / 6) + 1 / 6,
            temp[0] * (3 / 4 - 1 / 6) + temp[1] * (1 / 4 - 1 / 6) + 1 / 6,
        )
        temp[2] = temp[2] / 3

        # Generate the remaining objectives from subspace by repeating one half of lower section
        temp = temp[:, temp[0] >= temp[1]]
        f = np.empty((self.m, 0))
        for i in range(self.m - 1):
            f = np.concatenate(
                (
                    f,
                    np.vstack([temp[1] if i == j else temp[0] for j in range(self.m - 1)] + [temp[2]]),
                ),
                axis=1,
            )
        f = np.unique(f, axis=1)

        # Add the "pole"
        pole = np.linspace(1 / 3, 1.0, max(n - f.shape[1], npole))
        pole = np.vstack((np.repeat((1 - pole[None, :]) / 4, self.m - 1, axis=0), pole))
        return np.concatenate((f, pole), axis=1).T


class DTLZ9(DTLZx):
    @property
    def n_constraints(self):
        return self.m - 1

    def _call(self, x):
        # Transpose x (this function was written before ParetoBench standardized on rows being the batched index)
        x = x.T

        f = np.vstack(
            [
                np.sum(
                    x[np.floor(j * self.n / self.m).astype(int) : np.floor((j + 1) * self.n / self.m).astype(int)]
                    ** 0.1,
                    axis=0,
                )
                for j in range(self.m)
            ]
        )
        g = np.vstack([f[-1] ** 2 + f[j] ** 2 - 1 for j in range(self.m - 1)])
        return Population(f=f.T, g=g.T)

    def get_pareto_front(self, n):
        th = np.linspace(0, 1, n)
        return np.concatenate(
            (
                np.repeat(np.cos(np.pi / 2 * th[None, :]), self.m - 1, axis=0),
                np.sin(np.pi / 2 * th[None, :]),
            ),
            axis=0,
        ).T
