import numpy as np
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import pytest

from paretobench import Population, EmptyPopulationError
from paretobench.plotting import population_obj_scatter, PopulationObjScatterConfig
from paretobench.plotting.utils import get_per_point_settings_population
from paretobench.plotting.attainment import compute_attainment_surface_2d, compute_attainment_surface_3d


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


def test_population_obj_scatter_basic():
    """Test basic plotting functionality for both 2D and 3D cases"""
    # Test 2D case
    f_2d = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    pop_2d = Population(f=f_2d)

    fig, ax = population_obj_scatter(pop_2d)
    scatter_plots = [c for c in ax.collections if isinstance(c, PathCollection)]

    assert len(scatter_plots) == 1  # One scatter plot collection
    assert len(scatter_plots[0].get_offsets()) == 3  # All points plotted
    assert ax.get_xlabel() == "$f_1$"
    plt.close(fig)

    # Test 3D case
    f_3d = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [1.5, 1.5, 1.5]])
    pop_3d = Population(f=f_3d)

    fig, ax = population_obj_scatter(pop_3d)
    assert ax.name == "3d"
    scatter_plots = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatter_plots[0].get_offsets()) == 3
    plt.close(fig)

    # Check names get plotted
    f = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    pop = Population(f=f, names_f=["Cost", "Time"])

    # Test custom names
    fig, ax = population_obj_scatter(pop)
    assert ax.get_xlabel() == "Cost"
    assert ax.get_ylabel() == "Time"
    plt.close(fig)


def test_population_obj_scatter_edge_cases():
    """Test edge cases and error handling"""
    # Test empty population
    with pytest.raises(EmptyPopulationError):
        population_obj_scatter(Population(f=np.empty((0, 4))))

    # Test no points shown
    f = np.array([[1.0, 2.0], [2.0, 1.0]])
    pop = Population(f=f)
    settings = PopulationObjScatterConfig(show_points=False)
    fig, ax = population_obj_scatter(pop, settings=settings)
    scatter_plots = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatter_plots) == 0
    plt.close(fig)

    # No points after filtering
    settings = PopulationObjScatterConfig(plot_feasible="infeasible")
    fig, ax = population_obj_scatter(pop, settings=settings)
    scatter_plots = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatter_plots) == 0
    plt.close(fig)

    # Test plotting on existing axes
    fig, ax = plt.subplots()
    fig_return, ax_return = population_obj_scatter(pop, fig=fig, ax=ax)
    assert ax_return == ax
    assert fig_return == fig
    plt.close(fig)


def test_population_obj_scatter_attainment_and_dominated():
    """Test that attainment surface and dominated region appear in plot when requested"""
    # Create random population
    pop = Population.from_random(n_objectives=2, n_decision_vars=2, n_constraints=1, pop_size=5)

    # Test with both attainment and dominated area enabled
    settings = PopulationObjScatterConfig(plot_attainment=True, plot_dominated_area=True)

    fig, ax = population_obj_scatter(pop, settings=settings)

    # Check for attainment surface line
    lines = ax.get_lines()
    assert len(lines) > 0, "No attainment surface line found"

    # Check for dominated area fill
    fills = [c for c in ax.collections if isinstance(c, plt.matplotlib.collections.PolyCollection)]
    assert len(fills) > 0, "No dominated area fill found"

    plt.close(fig)


def test_attainment_surface_2d():
    """Test that no point on the attainment surface is dominated by the original population"""
    # Create a random population
    pop = Population.from_random(n_objectives=2, n_decision_vars=0, n_constraints=0, pop_size=50)

    # Compute attainment surface
    surface_points = compute_attainment_surface_2d(pop)

    # Check if any population point strongly dominates any surface point
    feasible_indices = pop.get_feasible_indices()
    objs = np.concatenate((pop.f[feasible_indices], surface_points), axis=0)
    strong_dom = (objs[:, None, :] < objs[None, :, :]).all(axis=-1)
    assert not strong_dom[
        : len(feasible_indices), len(feasible_indices) :
    ].any(), "Found surface points dominated by population points"


def test_attainment_surface_3d():
    """Test that no point on the attainment surface is dominated by the original population"""
    # Create a random population
    pop = Population.from_random(n_objectives=3, n_decision_vars=0, n_constraints=0, pop_size=50)

    # Compute attainment surface
    verts, _ = compute_attainment_surface_3d(pop)

    # Check if any population point strongly dominates any surface point
    feasible_indices = pop.get_feasible_indices()
    objs = np.concatenate((pop.f[feasible_indices], verts), axis=0)
    strong_dom = (objs[:, None, :] < objs[None, :, :]).all(axis=-1)
    assert not strong_dom[
        : len(feasible_indices), len(feasible_indices) :
    ].any(), "Found surface points dominated by population points"
