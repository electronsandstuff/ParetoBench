import numpy as np
from itertools import combinations, chain, count
from math import comb


def get_betas(m, p):
    '''
    From: Das, Indraneel, and J. E. Dennis. “Normal-Boundary Intersection: A New Method for
    Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.” SIAM 
    Journal on Optimization 8, no. 3 (August 1998): 631–57. https://doi.org/10.1137/S1052623496307510.
    '''
    beta = np.fromiter(chain.from_iterable(combinations(range(1, p+m), m-1)), np.float64)
    beta = beta.reshape(beta.shape[0]//(m-1), m-1).T
    beta = beta - np.arange(0, m-1)[:, None] - 1
    beta1 = np.concatenate((beta, np.full((1, beta.shape[1]), p)), axis=0)
    beta2 = np.concatenate((np.zeros((1, beta.shape[1])), beta), axis=0)
    return (beta1 - beta2)/p


def get_hyperplane_points(m, n):
    '''
    Returns at least n points of dimension m on the hyperplane x1 + x2 + x3 + ... = 1
    
    From: Das, Indraneel, and J. E. Dennis. “Normal-Boundary Intersection: A New Method for
    Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.” SIAM 
    Journal on Optimization 8, no. 3 (August 1998): 631–57. https://doi.org/10.1137/S1052623496307510.
    '''
    return get_betas(m, next(p for p in count() if comb(p+m-1, m-1) >= n))


def uniform_grid(n, m):
    '''
    At lesat n evenly spread points in the hypercube of dimension m
    '''
    return np.reshape(np.stack(np.meshgrid(*([np.linspace(0, 1, int(np.ceil(n**(1/m))))]*m))), (m, -1))


def fast_dominated_argsort(O):
    """
    Performs a dominated sort on matrix of objective function values O.  This is a numpy implementation of the algorithm
    described in Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). IEEE Transactions on Evolutionary
    Computation, 6(2), 182–197. https://doi.org/10.1109/4235.996017

    A list of ranks is returned referencing each individual by its index in the objective matrix.

    :param O: (M, N) numpy array where N is the number of individuals and M is the number of objectives
    :return: List of ranks where each rank is a list of the indices to the individuals in that rank
    """
    # Compare all pairs of individuals based on domination
    dom = np.bitwise_and((O[:, :, None] <= O[:, None, :]).all(axis=0), (O[:, :, None] < O[:, None, :]).any(axis=0))

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


def get_nondominated(O):
    """
    Returns the indices of the nondominated individuals for the objectives O, an (m,n) array for the m objectives
    """
    dom = np.bitwise_and((O[:, :, None] <= O[:, None, :]).all(axis=0), (O[:, :, None] < O[:, None, :]).any(axis=0))
    return np.where(np.sum(dom, axis=0) == 0)[0].tolist()
