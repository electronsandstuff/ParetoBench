from dataclasses import dataclass
from matplotlib.colors import LightSource
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple, Literal, Union, List
import matplotlib.pyplot as plt
import numpy as np


from ..containers import Population
from ..exceptions import EmptyPopulationError, NoDecisionVarsError, NoObjectivesError
from ..problem import Problem, ProblemWithFixedPF, ProblemWithPF
from ..utils import get_problem_from_obj_or_str
from .attainment import compute_attainment_surface_2d, compute_attainment_surface_3d
from .utils import get_per_point_settings_population, alpha_scatter, selection_to_indices


@dataclass
class PopulationObjScatterConfig:
    """
    Helper class for passing settings to `population_obj_scatter`
    """

    domination_filt: Literal["all", "dominated", "non-dominated"] = "all"
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all"
    show_points: bool = True
    problem: Optional[Union[str, Problem]] = None
    n_pf: int = 1000
    pf_objectives: Optional[np.ndarray] = None
    show_attainment: bool = False
    show_dominated_area: bool = False
    dominated_area_zorder: Optional[int] = -2
    ref_point: Optional[Tuple[float, float]] = None
    ref_point_padding: float = 0.05
    label: Optional[str] = None
    legend_loc: Optional[str] = None
    show_names: bool = True
    color: Optional[str] = None


def population_obj_scatter(
    population: Population,
    fig=None,
    ax=None,
    domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
    show_points: bool = True,
    problem: Optional[Union[str, Problem]] = None,
    n_pf: int = 1000,
    pf_objectives: Optional[np.ndarray] = None,
    show_attainment: bool = False,
    show_dominated_area: bool = False,
    dominated_area_zorder: Optional[int] = -2,
    ref_point: Optional[Tuple[float, float]] = None,
    ref_point_padding: float = 0.05,
    label: Optional[str] = None,
    legend_loc: Optional[str] = None,
    show_names: bool = True,
    color: Optional[str] = None,
):
    """
    Plot the objectives in 2D and 3D.

    Parameters
    ----------
    population : paretobench Population
        The population containing data to plot
    fig : matplotlib figure, optional
        Figure to plot on, by default None
    ax : matplotlib axis, optional
        Axis to plot on, by default None
    domination_filt : Literal["all", "dominated", "non-dominated"], optional
        Plot only the dominated/non-dominated solutions, or all. Defaults to all
    feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    show_points : bool
        Whether to actually show the points (useful for only showing attainment surface or dominated region)
    problem : str/Problem, optional
        Name of the problem for Pareto front plotting, by default None
    n_pf : int, optional
        The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
    pf_objectives : array-like, optional
        User-specified Pareto front objectives. Should be a 2D array where each row represents a point
        on the Pareto front and each column represents an objective value.
    show_attainment : bool, optional
        Whether to plot the attainment surface, by default False
    show_dominated_area : bool, optional
        Plots the dominated region towards the larger values of each decision var
    dominated_area_zorder : int, optional
        What "zorder" to draw dominated region at. Mostly used internally to correctly show dominated area in history plots.
    ref_point : Union[str, Tuple[float, float]], optional
        Where to stop plotting the dominated region / attainment surface. Must be a point to the upper right (increasing
        value of objectives in 3D) of all plotted points. By default, will set to right of max of each objective plus
        padding.
    ref_point_padding : float
        Amount of padding to apply to the automatic reference point calculation.
    label : str, optional
        The label for these points, if shown in a legend
    legend_loc : str, optional
        Passed to `loc` argument of plt.legend
    show_names : bool, optional
        Whether to show the names of the objectives if provided by population
    color : str, optional
        What color should we use for the points. Defaults to selecting from matplotlib color cycler

    Returns
    -------
    matplotlib figure and matplotlib axis
        The figure and axis containing the objectives plot
    """
    # Make sure we have been given data to plot
    if not len(population):
        raise EmptyPopulationError()
    if not population.m:
        raise NoObjectivesError()

    if fig is None:
        fig = plt.figure()

    # Input validation for Pareto front specification
    pf_sources_specified = sum(x is not None for x in [problem, pf_objectives])
    if pf_sources_specified > 1:
        raise ValueError("Multiple Pareto front sources specified. Use only one of: 'problem' or 'pf_objectives'")

    # Get the Pareto front
    pf = None
    if problem is not None:
        problem = get_problem_from_obj_or_str(problem)
        if problem.m != population.f.shape[1]:
            raise ValueError(
                f'Number of objectives in problem must match number in population. Got {problem.m} objectives from problem "{problem}" '
                f"and {population.f.shape[1]} from the population."
            )
        if isinstance(problem, ProblemWithPF):
            pf = problem.get_pareto_front(n_pf)
        elif isinstance(problem, ProblemWithFixedPF):
            pf = problem.get_pareto_front()
        else:
            raise ValueError(f'Cannot get Pareto front from problem: "{problem}"')
    elif pf_objectives is not None:
        pf = np.asarray(pf_objectives)
        if pf.ndim != 2:
            raise ValueError("pf_objectives must be a 2D array")
        if pf.shape[1] != population.f.shape[1]:
            raise ValueError(
                f"Number of objectives in pf_objectives must match number in population. Got {pf.shape[1]} in pf_objectives "
                f"and {population.f.shape[1]} in population"
            )

    # Get the point settings for this plot
    ps = get_per_point_settings_population(population, domination_filt, feasibility_filt)

    # For 2D problems
    add_legend = False
    base_color = color
    if population.f.shape[1] == 2:
        # Make axis if not supplied
        if ax is None:
            ax = fig.add_subplot(111)

        # Plot the data
        if show_points:
            scatter = alpha_scatter(
                ax,
                population.f[ps.plot_filt, 0],
                population.f[ps.plot_filt, 1],
                alpha=ps.alpha[ps.plot_filt],
                marker=ps.markers[ps.plot_filt],
                color=base_color,
                s=15,
                label=label,
            )
            if scatter:
                base_color = scatter[0].get_facecolor()[0]  # Get the color that matplotlib assigned

        # Plot attainment surface if requested (using feasible solutions only)
        attainment = compute_attainment_surface_2d(population, ref_point=ref_point, padding=ref_point_padding)
        if show_attainment:
            ax.plot(attainment[:, 0], attainment[:, 1], color=base_color, alpha=0.5, zorder=-1)
        if show_dominated_area:
            # We plot white first and then the actual color so we can stack dominated areas in the history plot while
            # correctly desaturating the color so the datapoints (which have the same color) don't blend in.
            plt.fill_between(
                attainment[:, 0],
                attainment[:, 1],
                attainment[0, 1] * np.ones(attainment.shape[0]),
                color="white",
                zorder=dominated_area_zorder,
            )
            plt.fill_between(
                attainment[:, 0],
                attainment[:, 1],
                attainment[0, 1] * np.ones(attainment.shape[0]),
                color=base_color,
                alpha=0.8,
                zorder=dominated_area_zorder,
            )

        # Add in Pareto front
        if pf is not None:
            # PF goes on bottom so our points show up on top of it when we have good solutions
            ax.scatter(pf[:, 0], pf[:, 1], c="k", s=10, label="PF", zorder=min(-1, dominated_area_zorder) - 1)
            add_legend = True

        # Handle the axis labels
        if population.names_f and show_names:
            ax.set_xlabel(population.names_f[0])
            ax.set_ylabel(population.names_f[1])
        else:
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$f_2$")

    # For 3D problems
    elif population.f.shape[1] == 3:
        # Get an axis if not supplied
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")

        # Add in Pareto front
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c="k", s=10, label="PF", alpha=0.75)
            add_legend = True

        # Plot the data
        if show_points:
            scatter = alpha_scatter(
                ax,
                population.f[ps.plot_filt, 0],
                population.f[ps.plot_filt, 1],
                population.f[ps.plot_filt, 2],
                alpha=ps.alpha[ps.plot_filt],
                marker=ps.markers[ps.plot_filt],
                color=base_color,
                s=15,
                label=label,
            )
            if scatter:
                base_color = scatter[0].get_facecolor()[0]  # Get the color that matplotlib assigned

        if show_dominated_area:
            raise NotImplementedError("Cannot display dominated volume in 3D :(")

        if show_attainment:
            if base_color is None:
                base_color = ax._get_lines.get_next_color()

            vertices, faces = compute_attainment_surface_3d(population, ref_point=ref_point, padding=ref_point_padding)
            poly3d = Poly3DCollection(
                [vertices[face] for face in faces],
                shade=True,
                facecolors=base_color,
                edgecolors=base_color,
                lightsource=LightSource(azdeg=174, altdeg=-15),
            )
            ax.add_collection3d(poly3d)

        # Handle the axis labels
        if population.names_f and show_names:
            ax.set_xlabel(population.names_f[0])
            ax.set_ylabel(population.names_f[1])
            ax.set_zlabel(population.names_f[2])
        else:
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$f_2$")
            ax.set_zlabel(r"$f_3$")

    # We can't plot in 4D :(
    else:
        raise ValueError(f"Plotting supports only 2D and 3D objectives currently: n_objs={population.f.shape[1]}")

    if add_legend:
        plt.legend(loc=legend_loc)

    return fig, ax


@dataclass
class PopulationDVarPairsConfig:
    """
    Settings related to plotting decision variables from `Population`.

    domination_filt : Literal["all", "dominated", "non-dominated"], optional
        Plot only the dominated/non-dominated solutions, or all. Defaults to all
    feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    hist_bins : int, optional
        Number of bins for histograms on the diagonal, default is to let matplotlib choose
    show_names : bool, optional
        Whether to include variable names on the axes if they exist, default is True
    problem : str/Problem, optional
        The problem for plotting decision variable bounds
    lower_bounds : array-like, optional
        Lower bounds for each decision variable
    upper_bounds : array-like, optional
        Upper bounds for each decision variable
    color : str, optional
        What color should we use for the points. Defaults to selecting from matplotlib color cycler
    """

    domination_filt: Literal["all", "dominated", "non-dominated"] = "all"
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all"
    hist_bins: Optional[int] = None
    show_names: bool = True
    problem: Optional[Union[str, Problem]] = None
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    color: Optional[str] = None


def population_dvar_pairs(
    population: Population,
    dvars: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    fig=None,
    axes=None,
    settings: PopulationDVarPairsConfig = PopulationDVarPairsConfig(),
):
    """
    Creates a pairs plot (scatter matrix) showing correlations between decision variables
    and their distributions.

    Parameters
    ----------
    population : paretobench Population
        The population containing data to plot
    dvars : int, slice, List[int], or Tuple[int, int], optional
        Specifies which decision variables to plot. Can be:
        - None: All variables (default)
        - int: Single variable index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd var)
        - List[int]: Explicit list of variable indices
        - List[bool] or np.ndarray of bools: boolean mask where True selects the index
        - Tuple[int, int]: Range of variables as (start, end) where end is exclusive
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None and axes is None, creates a new figure.
    axes : numpy.ndarray of matplotlib.axes.Axes, optional
        2D array of axes to plot on. If None and fig is None, creates new axes.
        Must be provided if fig is provided and vice versa.
    settings : PopulationDVarPairsConfig
        Settings related to plotting the decision variables

    Returns
    -------
    tuple
        (matplotlib.figure.Figure, numpy.ndarray of matplotlib.axes.Axes)
        The figure and axes containing the pairs plot
    """
    # Make sure we have been given data to plot
    if not len(population):
        raise EmptyPopulationError()
    if not population.n:
        raise NoDecisionVarsError()

    # Process the reports selection
    var_indices = np.array(selection_to_indices(dvars, population.n))
    n_vars = len(var_indices)

    # Default, don't show bounds
    lower_bounds = None
    upper_bounds = None

    # Handle user specified problem
    if settings.problem is not None:
        if (settings.lower_bounds is not None) or (settings.upper_bounds is not None):
            raise ValueError("Only specify one of problem or the upper/lower bounds")
        problem = get_problem_from_obj_or_str(settings.problem)
        if problem.n != population.n:
            raise ValueError(
                f"Number of decision vars in problem must match number in population. Got {problem.n} in problem and {population.n} in population"
            )
        lower_bounds = problem.var_lower_bounds
        upper_bounds = problem.var_upper_bounds

    # Validate and convert bounds to numpy arrays if provided
    if settings.lower_bounds is not None:
        lower_bounds = np.asarray(settings.lower_bounds)
        if len(lower_bounds) != population.n:
            raise ValueError(
                f"Length of lower_bounds ({len(lower_bounds)}) must match number of variables ({population.n})"
            )

    if settings.upper_bounds is not None:
        upper_bounds = np.asarray(settings.upper_bounds)
        if len(upper_bounds) != population.n:
            raise ValueError(
                f"Length of upper_bounds ({len(upper_bounds)}) must match number of variables ({population.n})"
            )

    # Handle figure and axes creation/validation
    if fig is None and axes is None:
        # Create new figure and axes
        fig, axes = plt.subplots(
            n_vars,
            n_vars,
            figsize=(2 * n_vars, 2 * n_vars),
            gridspec_kw={"wspace": 0.05, "hspace": 0.05},
            layout="constrained",
            sharex="col",
        )

        # Convert axes to 2D array if only one variable is selected
        if n_vars == 1:
            axes = np.array([[axes]])

    elif (fig is None) != (axes is None):  # XOR operation
        raise ValueError("Either both fig and axes must be provided or neither must be provided")
    else:
        # Validate provided axes dimensions
        if axes.shape != (n_vars, n_vars):
            raise ValueError(f"Provided axes must have shape ({n_vars}, {n_vars}), got {axes.shape}")

    # Style all axes
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

            # Hide x-axis labels and ticks for all rows except the bottom row
            if i != n_vars - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

    # Get variable names or create default ones
    if population.names_x and settings.show_names:
        var_names = [population.names_x[i] for i in var_indices]
    else:
        var_names = [f"x{i+1}" for i in var_indices]

    # Define bound line properties
    bound_props = dict(color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Get the point settings for this plot
    ps = get_per_point_settings_population(population, settings.domination_filt, settings.feasibility_filt)

    # Plot on all axes
    base_color = settings.color
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            # Diagonal plots (histograms)
            if i == j:
                # Plot the histogram
                _, _, patches = ax.hist(
                    population.x[ps.plot_filt, var_indices[i]],
                    bins=settings.hist_bins,
                    density=True,
                    alpha=0.7,
                    color=base_color,
                )
                if base_color is None:
                    base_color = patches[0].get_facecolor()

                # Add vertical bound lines to histograms
                if lower_bounds is not None:
                    ax.axvline(lower_bounds[var_indices[i]], **bound_props)
                if upper_bounds is not None:
                    ax.axvline(upper_bounds[var_indices[i]], **bound_props)

            # Off-diagonal plots (scatter plots)
            else:
                # Plot the decision vars
                scatter = alpha_scatter(
                    ax,
                    population.x[ps.plot_filt, var_indices[j]],
                    population.x[ps.plot_filt, var_indices[i]],
                    alpha=ps.alpha[ps.plot_filt],
                    color=base_color,
                    s=15,
                    marker=ps.markers[ps.plot_filt],
                )
                if base_color is None:
                    base_color = scatter.get_facecolor()[0]  # Get the color that matplotlib assigned

                # Add bound lines to scatter plots
                if lower_bounds is not None:
                    ax.axvline(lower_bounds[var_indices[j]], **bound_props)  # x-axis bound
                    ax.axhline(lower_bounds[var_indices[i]], **bound_props)  # y-axis bound
                if upper_bounds is not None:
                    ax.axvline(upper_bounds[var_indices[j]], **bound_props)  # x-axis bound
                    ax.axhline(upper_bounds[var_indices[i]], **bound_props)  # y-axis bound
            if i == n_vars - 1:
                ax.set_xlabel(var_names[j])
            if j == 0:
                ax.set_ylabel(var_names[i])
    return fig, axes
