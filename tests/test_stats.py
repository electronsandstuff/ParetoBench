import numpy as np
import pytest
from scipy.stats import ranksums as scipy_ranksums

from paretobench.stats import ranksums


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize(
    "x, y",
    [
        (np.arange(1, 11, dtype=float), np.arange(11, 21, dtype=float)),
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        (np.array([1.0, 3.0, 5.0, 7.0]), np.array([2.0, 4.0, 6.0, 8.0])),
        (np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0, 7.0, 8.0])),
    ],
)
def test_ranksums_vs_scipy(x, y, alternative):
    """ranksums statistic and p-value match scipy for all alternatives and sample types."""
    stat, pval = ranksums(x, y, alternative)
    expected_stat, expected_pval = scipy_ranksums(x, y, alternative)
    assert stat == pytest.approx(expected_stat, rel=1e-6)
    assert pval == pytest.approx(expected_pval, rel=1e-6)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ranksums_vs_scipy_random(alternative, seed):
    """ranksums matches scipy on random samples."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=20)
    y = rng.normal(0.5, 1.0, size=25)
    stat, pval = ranksums(x, y, alternative)
    expected_stat, expected_pval = scipy_ranksums(x, y, alternative)
    assert stat == pytest.approx(expected_stat, rel=1e-6)
    assert pval == pytest.approx(expected_pval, rel=1e-6)
