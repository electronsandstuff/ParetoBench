import numpy as np


def g_1_3(x, k):
    x -= 0.5
    return 100 * (k + np.sum(x**2 - np.cos(20 * np.pi * x), axis=0))


def g_2_4_5(x):
    return np.sum((x - 0.5) ** 2, axis=0)


def theta_5_6(x, g, m):
    th = (1 + 2 * g[None, :] * x[: m - 1, :]) / (2 + 2 * g[None, :])
    return np.vstack([x[0], th[1:, :]])


def f_2_to_6(x, m, alpha=1):
    f1 = np.vstack([np.prod(np.cos(x[: x.shape[0] - i, :] ** alpha * np.pi / 2), axis=0) for i in range(0, m)])
    f2 = np.vstack([np.ones(x.shape[1])] + [np.sin(x[x.shape[0] - i, :] ** alpha * np.pi / 2) for i in range(1, m)])
    return f1 * f2
