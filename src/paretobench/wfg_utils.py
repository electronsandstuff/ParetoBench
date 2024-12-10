import numpy as np


########################################################################################################################
# Shape functions
########################################################################################################################


def shape_linear(x):
    f = []
    f.append(np.prod(x[:-1], axis=0))
    for m in range(2, x.shape[0]):
        f.append(np.prod(x[:-m], axis=0) * (1 - x[-m]))
    if x.shape[0] > 1:
        f.append(1 - x[0])
    return np.vstack(f)


def shape_convex(x):
    f = []
    f.append(np.prod(1 - np.cos(x[:-1] * np.pi / 2), axis=0))
    for m in range(2, x.shape[0]):
        f.append(np.prod(1 - np.cos(x[:-m] * np.pi / 2), axis=0) * (1 - np.sin(x[-m] * np.pi / 2)))
    if x.shape[0] > 1:
        f.append(1 - np.sin(x[0] * np.pi / 2))
    return np.vstack(f)


def shape_concave(x):
    f = []
    f.append(np.prod(np.sin(x[:-1] * np.pi / 2), axis=0))
    for m in range(2, x.shape[0]):
        f.append(np.prod(np.sin(x[:-m] * np.pi / 2), axis=0) * (np.cos(x[-m] * np.pi / 2)))
    if x.shape[0] > 1:
        f.append(np.cos(x[0] * np.pi / 2))
    return np.vstack(f)


def shape_mixed(x, a, alpha):
    f = (1 - x[0] - np.cos(2 * a * np.pi * x[0] + np.pi / 2.0) / 2 / a / np.pi) ** alpha
    return np.vstack((f,))


def shape_disconnected(x, a, alpha, beta):
    f = 1 - x[0] ** alpha * np.cos(a * x[0] ** beta * np.pi) ** 2
    return np.vstack((f,))


def compute_x(t, degenerate):
    if degenerate:
        a = np.zeros(t.shape[0] - 1)
        a[0] = 1.0
    else:
        a = np.ones(t.shape[0] - 1)

    x = np.maximum(t[-1][None, :], a[:, None]) * (t[:-1, :] - 0.5) + 0.5
    x = np.append(x, t[-1][None, :], axis=0)
    return x


########################################################################################################################
# Transformation functions
########################################################################################################################


def transform_bias_polynomial(y, alpha):
    return y**alpha


def transform_bias_flat_region(y, a, b, c):
    return (
        a
        + np.clip(np.floor(y - b), np.finfo("d").min, 0.0) * a * (b - y) / b
        - np.clip(np.floor(c - y), np.finfo("d").min, 0.0) * (1 - a) * (y - c) / (1 - c)
    )


def transform_bias_parameter_dependent(y, u_of_y_prime, a, b, c):
    v = a - (1 - 2 * u_of_y_prime) * np.abs(np.floor(0.5 - u_of_y_prime) + a)
    return np.power(y, b + (c - b) * v)


def transform_shift_linear(y, a):
    return np.abs(y - a) / np.abs(np.floor(a - y) + a)


def transform_shift_deceptive(y, a, b, c):
    return 1 + (np.abs(y - a) - b) * (
        np.floor(y - a + b) * (1 - c + (a - b) / b) / (a - b)
        + np.floor(a + b - y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
        + 1 / b
    )


def transform_shift_multimodal(y, a, b, c):
    return (
        1
        + np.cos((4 * a + 2) * np.pi * (0.5 - np.abs(y - c) / 2 / (np.floor(c - y) + c)))
        + 4 * b * (np.abs(y - c) / 2 / (np.floor(c - y) + c)) ** 2
    ) / (b + 2)


def transform_reduction_weighted_sum(y, w=None):
    if w is None:
        w = np.ones(y.shape[0])
    return np.sum(y * w[:, None], axis=0) / np.sum(w)


def transform_reduction_non_separable(y, a):
    acc1 = np.zeros(y.shape[1])
    for j in range(y.shape[0]):
        acc1 += y[j]
        acc2 = np.zeros(y.shape[1])
        for k in range(a - 1):
            acc2 += np.abs(y[j] - y[(1 + j + k) % y.shape[0]])
        acc1 += acc2
    acc1 /= y.shape[0] / a * np.ceil(a / 2) * (1 + 2 * a - 2 * np.ceil(a / 2))
    return acc1
