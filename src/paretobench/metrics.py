import numpy as np


def get_inverse_generational_distance(O, ref):
    """
    Calculates convergence metric between a pareto front O and a reference set points ref.  This is the mean of minimum
    distances between points on the front and the reference as described in the NSGA-II paper.

    :param O: (M,N) numpy array where N is the number of individuals and M is the number of objectives
    :param ref: (M,L) numpy array of reference points on the Pareto front
    :return: T, the convergence metric
    """
    # Compute pairwise distance between every point in the front and reference
    d = np.sqrt(np.sum((ref[:, :, None] - O[:, None, :]) ** 2, axis=0))

    # Find the minimum distance for each point and average it
    d_min = np.min(d, axis=1)
    return np.mean(d_min)


def get_generational_distance(O, ref):
    """
    Calculates convergence metric between a pareto front O and a reference set points ref.  This is the mean of minimum
    distances between points on the front and the reference as described in the NSGA-II paper.

    :param O: (M,N) numpy array where N is the number of individuals and M is the number of objectives
    :param ref: (M,L) numpy array of reference points on the Pareto front
    :return: T, the convergence metric
    """
    # Compute pairwise distance between every point in the front and reference
    d = np.sqrt(np.sum((ref[:, :, None] - O[:, None, :]) ** 2, axis=0))

    # Find the minimum distance for each point and average it
    d_min = np.min(d, axis=0)
    return np.mean(d_min)