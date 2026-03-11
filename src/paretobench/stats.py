import numpy as np
import math


def rankdata(a):
    """Assign ranks to data, handling ties by averaging."""
    a = np.asarray(a).ravel()
    sorter = np.argsort(a, kind="mergesort")
    ranked = np.empty(len(a), dtype=float)
    ranked[sorter] = np.arange(1, len(a) + 1, dtype=float)

    # average ties
    a_sorted = a[sorter]
    tie_mask = np.concatenate(([False], a_sorted[1:] == a_sorted[:-1], [False]))
    tie_starts = np.where(~tie_mask[:-1] & tie_mask[1:])[0]
    tie_ends = np.where(tie_mask[:-1] & ~tie_mask[1:])[0]
    for start, end in zip(tie_starts, tie_ends):
        avg = (start + end + 2) / 2.0  # ranks are 1-indexed
        ranked[sorter[start : end + 1]] = avg

    return ranked


def ranksums(x, y, alternative="two-sided"):
    """
    Compute the Wilcoxon rank-sum statistic for two samples.

    Parameters
    ----------
    x, y : array_like
        The data from the two samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Default is 'two-sided'.

    Returns
    -------
    statistic : float
    pvalue : float
    """

    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    if alternative == "two-sided":
        pvalue = 2 * _norm_sf(abs(z))
    elif alternative == "less":
        pvalue = _norm_cdf(z)
    elif alternative == "greater":
        pvalue = _norm_sf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return float(z), float(pvalue)


def _norm_sf(z):
    """Survival function of standard normal."""
    return 0.5 * (1 - _erf(z / np.sqrt(2)))


def _norm_cdf(z):
    """CDF of standard normal."""
    return 0.5 * (1 + _erf(z / np.sqrt(2)))


def _erf(x):
    """Error function using math.erf."""
    if np.ndim(x) == 0:
        return math.erf(float(x))
    return np.vectorize(math.erf)(x)
