import numpy as np

from paretobench import Population
from paretobench.plotting.utils import get_per_point_settings_population


def test_per_point_settings_nondominated():
    """Test with a population where all points are feasible and non-dominated"""
    # Create a simple population where all points are feasible and non-dominated
    pop = Population.from_random(n_objectives=2, n_decision_vars=2, n_constraints=1, pop_size=3)
    # Make all points feasible
    pop.g[:] = 1.0
    # Make points non-dominated by setting objectives such that no point dominates another
    pop.f = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])

    settings = get_per_point_settings_population(pop, plot_dominated="all", plot_feasible="all")

    assert np.all(settings.nd_inds)  # All should be non-dominated
    assert np.all(settings.feas_inds)  # All should be feasible
    assert np.all(settings.plot_filt)  # All should be visible
    assert np.all(settings.markers == "o")  # All should have feasible marker
    assert np.all(settings.alpha == 1.0)  # All should have full opacity


def test_per_point_settings_random():
    """Test with a mixed population containing both dominated and infeasible points"""
    pop = Population.from_random(n_objectives=2, n_decision_vars=2, n_constraints=1, pop_size=5)

    # Set up specific test cases
    pop.f = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [0.5, 0.5], [1.5, 0.5]])

    pop.g = np.array([[1.0], [1.0], [-1.0], [-1.0], [1.0]])

    settings = get_per_point_settings_population(pop, plot_dominated="all", plot_feasible="all")

    # Check non-dominated status
    assert np.array_equal(settings.nd_inds, [True, False, False, False, True])

    # Check feasibility status
    assert np.array_equal(settings.feas_inds, [True, True, False, False, True])

    # Check markers
    assert np.array_equal(settings.markers, ["o", "o", "x", "x", "o"])

    # All points should be visible
    assert np.all(settings.plot_filt)

    # Test filtering
    settings = get_per_point_settings_population(pop, plot_dominated="non-dominated", plot_feasible="feasible")
    assert np.array_equal(settings.plot_filt, [True, False, False, False, True])

    # Test filtering
    settings = get_per_point_settings_population(pop, plot_dominated="dominated", plot_feasible="infeasible")
    assert np.array_equal(settings.plot_filt, [False, False, True, True, False])
