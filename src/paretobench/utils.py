from itertools import combinations, chain, count
from math import comb
from typing import Union
import numpy as np

from .problem import Problem


def get_betas(m, p):
    """
    From: Das, Indraneel, and J. E. Dennis. “Normal-Boundary Intersection: A New Method for
    Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.” SIAM
    Journal on Optimization 8, no. 3 (August 1998): 631–57. https://doi.org/10.1137/S1052623496307510.
    """
    beta = np.fromiter(chain.from_iterable(combinations(range(1, p + m), m - 1)), np.float64)
    beta = beta.reshape(beta.shape[0] // (m - 1), m - 1).T
    beta = beta - np.arange(0, m - 1)[:, None] - 1
    beta1 = np.concatenate((beta, np.full((1, beta.shape[1]), p)), axis=0)
    beta2 = np.concatenate((np.zeros((1, beta.shape[1])), beta), axis=0)
    return (beta1 - beta2) / p


def get_hyperplane_points(m, n):
    """
    Returns at least n points of dimension m on the hyperplane x1 + x2 + x3 + ... = 1

    From: Das, Indraneel, and J. E. Dennis. “Normal-Boundary Intersection: A New Method for
    Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.” SIAM
    Journal on Optimization 8, no. 3 (August 1998): 631–57. https://doi.org/10.1137/S1052623496307510.
    """
    return get_betas(m, next(p for p in count() if comb(p + m - 1, m - 1) >= n))


def uniform_grid(n, m):
    """
    At lesat n evenly spread points in the hypercube of dimension m
    """
    return np.reshape(
        np.stack(np.meshgrid(*([np.linspace(0, 1, int(np.ceil(n ** (1 / m))))] * m))),
        (m, -1),
    )


def get_domination(objs, constraints=None):
    # Compare all pairs of individuals based on domination
    dom = np.bitwise_and(
        (objs[:, None, :] <= objs[None, :, :]).all(axis=-1),
        (objs[:, None, :] < objs[None, :, :]).any(axis=-1),
    )

    if constraints is not None:
        # If one individual is feasible and the other isn't, set domination
        feas = constraints >= 0.0
        ind = np.bitwise_and(feas.all(axis=1)[:, None], ~feas.all(axis=1)[None, :])
        dom[ind] = True
        ind = np.bitwise_and(~feas.all(axis=1)[:, None], feas.all(axis=1)[None, :])
        dom[ind] = False

        # If both are infeasible, then the individual with the least constraint violation wins
        constraint_violation = -np.sum(np.minimum(constraints, 0), axis=1)
        comp = constraint_violation[:, None] < constraint_violation[None, :]
        ind = ~np.bitwise_or(feas.all(axis=1)[:, None], feas.all(axis=1)[None, :])
        dom[ind] = comp[ind]
    return dom


def fast_dominated_argsort(objs, constraints=None):
    """
    Performs a dominated sort on matrix of objective function values O.  This is a numpy implementation of the algorithm
    described in Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). IEEE Transactions on Evolutionary
    Computation, 6(2), 182–197. https://doi.org/10.1109/4235.996017

    A list of ranks is returned referencing each individual by its index in the objective matrix.

    :param O: (M, N) numpy array where N is the number of individuals and M is the number of objectives
    :return: List of ranks where each rank is a list of the indices to the individuals in that rank
    """
    # Compare all pairs of individuals based on domination
    dom = get_domination(objs, constraints)

    # Create the sets of dominated individuals, domination number, and first rank
    S = [np.nonzero(row)[0].tolist() for row in dom]
    N = np.sum(dom, axis=0)
    F = [np.where(N == 0)[0].tolist()]

    i = 0
    while len(F[-1]) > 0:
        Q = []
        for p in F[i]:
            for q in S[p]:
                N[q] -= 1
                if N[q] == 0:
                    Q.append(q)
        F.append(Q)
        i += 1

    # Remove last empty set
    F.pop()

    return F


def get_nondominated_inds(objs, constraints=None):
    """
    Returns the indices of the nondominated individuals for the objectives O, an (m,n) array for the m objectives
    """
    dom = get_domination(objs, constraints)
    return np.where(np.sum(dom, axis=0) == 0)[0].tolist()


def triangle_grid_count(n):
    return n * (n + 1) // 2


def triangle_grid(n):
    # skewed points to help compensate for
    x = np.concatenate([np.linspace(0, 1 - np.sqrt(i / (n - 1)), n - i) for i in range(n)])
    y = np.concatenate([np.ones(n - i) * np.sqrt(i / (n - 1)) for i in range(n)])
    return np.vstack((x, y))


def rastrigin(x):
    """
    Compute the Rastragin function for optimization.  The bounds for optimization are [-5.12, 5.12] in all variables

    Reference: Huband, S., Hingston, P., Barone, L., & While, L. (2006). A review of multiobjective test problems and a
    scalable test problem toolkit. IEEE Transactions on Evolutionary Computation, 10(5), 477–506.
    https://doi.org/10.1109/TEVC.2005.861417

    :param x: (m,n) input for m dimensions and n separate instances
    :return: g(x), the Rastrigin function
    """
    a = 10
    return 1 + a * x.shape[0] + np.sum(x**2 - a * np.cos(2 * np.pi * x), axis=0)


def weighted_chunk_sizes(n, weights):
    """
    Break the number n into a list of numbers that sum to n, with the given
    approximate weights

    Args:
        n (int): The number to break apart
        weights (list): list of the weights of each chunk (should sum to one)

    Returns:
        list: The size of each chunk summing to n
    """
    ns = [int(np.floor(n * w / sum(weights))) for w in weights]
    for i in range(32):
        if sum(ns) < n:
            ns[i % len(ns)] += 1
        else:
            break
    return ns


def get_problem_from_obj_or_str(obj_or_str: Union[str, Problem]) -> Problem:
    """Convert input to Problem instance.

    Parameters
    ----------
    obj_or_str : Problem or str
        Input to convert. If already a Problem instance, returns as-is.
        If string, creates Problem from line format.

    Returns
    -------
    Problem
        The resulting Problem instance.

    Raises
    ------
    ValueError
        If input is neither Problem nor str type.
    """
    if isinstance(obj_or_str, Problem):
        return obj_or_str
    elif isinstance(obj_or_str, str):
        return Problem.from_line_fmt(obj_or_str)
    else:
        raise ValueError(f"Unrecognized input type: {type(obj_or_str)}")
