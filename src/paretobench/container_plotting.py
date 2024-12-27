import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import animation
from matplotlib.colors import to_rgb
from typing import Optional, Tuple, Literal
import numpy as np
from dataclasses import dataclass
from copy import copy

from .containers import Population, History
from .problem import ProblemWithFixedPF, ProblemWithPF, get_problem_from_obj_or_str
from .exceptions import EmptyPopulationError, NoDecisionVarsError, NoObjectivesError
from .utils import fast_dominated_argsort


@dataclass
class PointSettings:
    nd_inds: np.ndarray
    feas_inds: np.ndarray
    markers: np.ndarray
    plot_filt: np.ndarray
    alpha: np.ndarray


def get_per_point_settings_population(population: Population, plot_dominated, plot_feasible):
    """
    Calculate the per-point settings for scatter plots of the population (ie color, marker, which points are visible)
    based on shared settings across plot types.
    """
    # Break the objectives into those which are non-dominated and those which are not
    nd_inds = population.get_nondominated_indices()
    feas_inds = population.get_feasible_indices()
    markers = np.where(feas_inds, "o", "x")

    # Process filters for what is visible
    plot_filt = np.ones(len(population), dtype=bool)

    # Handle the domination filter
    if plot_dominated == "all":
        pass
    elif plot_dominated == "dominated":
        plot_filt = np.bitwise_and(plot_filt, ~nd_inds)
    elif plot_dominated == "non-dominated":
        plot_filt = np.bitwise_and(plot_filt, nd_inds)
    else:
        raise ValueError(f"Unrecognized option for plot_dominated: {plot_dominated}")

    # Handle the feasibility filter
    if plot_feasible == "all":
        pass
    elif plot_feasible == "feasible":
        plot_filt = np.bitwise_and(plot_filt, feas_inds)
    elif plot_feasible == "infeasible":
        plot_filt = np.bitwise_and(plot_filt, ~feas_inds)
    else:
        raise ValueError(f"Unrecognized option for plot_feasible: {plot_feasible}")

    # Get the domination ranks and set alpha based on that
    ranks = np.empty(len(population))
    for rank, idx in enumerate(fast_dominated_argsort(population.f, population.g)):
        ranks[idx] = rank
    if np.all(rank < 1):
        alpha = np.ones(len(population))
    else:
        alpha = 0.5 - ranks / ranks.max() * 0.3
        alpha[ranks == 0] = 1.0

    return PointSettings(nd_inds=nd_inds, feas_inds=feas_inds, plot_filt=plot_filt, alpha=alpha, markers=markers)


def alpha_scatter(ax, x, y, color=None, alpha=None, marker=None, **kwargs):
    if color is None:
        color = ax._get_lines.get_next_color()

    if alpha is None:
        alpha = 1.0

    r, g, b = to_rgb(color)
    if isinstance(alpha, float):
        color = (r, g, b, alpha)
    else:
        color = [(r, g, b, a) for a in alpha]

    if marker is None:
        return [ax.scatter(x, y, c=color, **kwargs)]

    if isinstance(marker, str):
        return [ax.scatter(x, y, c=color, marker=marker, **kwargs)]

    points = []
    unique_markers = set(marker)
    for m in unique_markers:
        mask = np.array(marker) == m
        filtered_color = np.array(color)[mask] if isinstance(color, list) else color
        points.append(ax.scatter(x[mask], y[mask], c=filtered_color, marker=m, **kwargs))
    return points


def compute_attainment_surface(points):
    """
    Compute the attainment surface for a set of non-dominated points in 2D.
    The surface consists of horizontal and vertical lines connecting the points,
    forming a staircase-like pattern.

    Parameters
    ----------
    points : np.ndarray
        2D array of non-dominated points, shape (n_points, 2)

    Returns
    -------
    np.ndarray
        Array of points defining the attainment surface, shape (n_segments, 2)
        Each consecutive pair of points defines a line segment of the surface
    """
    if points.shape[1] != 2:
        raise ValueError("Attainment surface can only be computed for 2D points")

    # Sort points by x coordinate (first objective)
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Initialize the surface points list with the first point
    surface = []
    surface.append(sorted_points[0])

    # Generate horizontal-then-vertical segments between each pair of points
    for i in range(len(sorted_points) - 1):
        current = sorted_points[i]
        next_point = sorted_points[i + 1]

        # Add horizontal line point
        surface.append([next_point[0], current[1]])
        # Add the next point
        surface.append(next_point)

    return np.array(surface)


@dataclass
class PlotObjectivesSettings:
    """
    Settings for plotting the objective functions from `Population`.

    plot_dominated : bool, optional
        Include the dominated individuals, by default True
    plot_feasible : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    problem : str, optional
        Name of the problem for Pareto front plotting, by default None
    n_pf : int, optional
        The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
    pf_objectives : array-like, optional
        User-specified Pareto front objectives. Should be a 2D array where each row represents a point
        on the Pareto front and each column represents an objective value.
    plot_attainment : bool, optional
        Whether to plot the attainment surface (2D only), by default False
    plot_dominated_area : bool, optional
        Plots the dominated region towards the larger values of each decision var
    ref_point : Union[str, Tuple[float, float]], optional
        Where to stop plotting the dominated region. Must be a point to the upper right (increasing value of objectives in 3D)
        of all plotted points. By default, will set to right of max of each objective plus padding.
    ref_point_padding : float
        Amount of padding to apply to the automatic reference point calculation.
    label : str, optional
        The label for these points, if shown in a legend
    legend_loc : str, optional
        Passed to `loc` argument of plt.legend
    """

    plot_dominated: Literal["all", "dominated", "non-dominated"] = "all"
    plot_feasible: Literal["all", "feasible", "infeasible"] = "all"
    problem: Optional[str] = None
    n_pf: int = 1000
    pf_objectives: Optional[np.ndarray] = None
    plot_attainment: bool = False
    plot_dominated_area: bool = False
    ref_point: Optional[Tuple[float, float]] = None
    ref_point_padding: float = 0.05
    label: Optional[str] = None
    legend_loc: Optional[str] = None


def plot_objectives(
    population: Population,
    fig=None,
    ax=None,
    settings: PlotObjectivesSettings = PlotObjectivesSettings(),
):
    """
    Plot the objectives in 2D and 3D. Optionally add in the Pareto front either from a known problem
    or from user-specified objectives. User must specify only one of 'problem' or 'pf_objectives'.

    Parameters
    ----------
    population : paretobench Population
        The population containing data to plot
    fig : matplotlib figure, optional
        Figure to plot on, by default None
    ax : matplotlib axis, optional
        Axis to plot on, by default None
    settings : PlotObjectivesSettings
        Settings for the plot

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

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Input validation for Pareto front specification
    pf_sources_specified = sum(x is not None for x in [settings.problem, settings.pf_objectives])
    if pf_sources_specified > 1:
        raise ValueError("Multiple Pareto front sources specified. Use only one of: 'problem' or 'pf_objectives'")

    # Get the Pareto front
    pf = None
    if settings.problem is not None:
        problem = get_problem_from_obj_or_str(settings.problem)
        if problem.m != population.f.shape[1]:
            raise ValueError(
                f"Number of objectives in problem must match number in population. Got {problem.m} in problem and {population.f.shape[1]} in population."
            )
        if isinstance(problem, ProblemWithPF):
            pf = problem.get_pareto_front(settings.n_pf)
        elif isinstance(problem, ProblemWithFixedPF):
            pf = problem.get_pareto_front()
        else:
            raise ValueError(f"Cannot get Pareto front from object of problem: {problem}")
    elif settings.pf_objectives is not None:
        pf = np.asarray(settings.pf_objectives)
        if pf.ndim != 2:
            raise ValueError("pf_objectives must be a 2D array")
        if pf.shape[1] != population.f.shape[1]:
            raise ValueError(
                f"Number of objectives in pf_objectives must match number in population. Got {pf.shape[1]} in pf_objectives and {population.f.shape[1]} in population"
            )

    # Get the point settings for this plot
    ps = get_per_point_settings_population(population, settings.plot_dominated, settings.plot_feasible)

    # For 2D problems
    add_legend = False
    base_color = None
    if population.f.shape[1] == 2:
        # Make axis if not supplied
        if ax is None:
            ax = fig.add_subplot(111)

        # Feasible individuals
        scatter = alpha_scatter(
            ax,
            population.f[ps.plot_filt, 0],
            population.f[ps.plot_filt, 1],
            alpha=ps.alpha[ps.plot_filt],
            marker=ps.markers[ps.plot_filt],
            color=base_color,
            s=15,
            label=settings.label,
        )
        if scatter:
            base_color = scatter[0].get_facecolor()[0]  # Get the color that matplotlib assigned

        # Plot attainment surface if requested (using the non-dominated, feasible objectives only)
        inds = np.bitwise_and(ps.nd_inds, ps.feas_inds)
        filt_f = population.f[inds, :]
        if settings.plot_attainment and len(filt_f) > 0:
            attainment = compute_attainment_surface(filt_f)
            ax.plot(attainment[:, 0], attainment[:, 1], color=base_color, alpha=0.5, label="Attainment Surface")
            add_legend = True

        if settings.plot_dominated_area and len(filt_f) > 0:
            if settings.ref_point is None:
                padding = settings.ref_point_padding
                x_lb, x_ub = np.min(population.f[ps.plot_filt, 0]), np.max(population.f[ps.plot_filt, 0])
                y_lb, y_ub = np.min(population.f[ps.plot_filt, 1]), np.max(population.f[ps.plot_filt, 1])
                ref_point = (x_ub + (x_ub - x_lb) * padding, y_ub + (y_ub - y_lb) * padding)
            else:
                ref_point = settings.ref_point

            attainment = compute_attainment_surface(filt_f)
            if (ref_point[0] < attainment[:, 0]).any() or (ref_point[1] < attainment[:, 1]).any():
                raise ValueError(
                    f"Reference point coordinates must exceed all points in non-dominated set "
                    f"(ref_point={ref_point}, max_pf=({np.max(attainment[:, 0])}, {np.max(attainment[:, 1])}))"
                )

            plt.fill_between(
                np.concatenate((attainment[:, 0], [ref_point[0]])),
                np.concatenate((attainment[:, 1], [attainment[-1, 1]])),
                ref_point[1] * np.ones(attainment.shape[0] + 1),
                color=base_color,
                alpha=0.5,
            )

        # Add in Pareto front
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], c="k", s=10, label="PF")
            add_legend = True

        # Handle the axis labels
        if population.names_f:
            ax.set_xlabel(population.names_f[0])
            ax.set_ylabel(population.names_f[1])
        else:
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$f_2$")

    # For 3D problems
    elif population.f.shape[1] == 3:
        raise NotImplementedError("3D plotting will be implemented in future")
        # if settings.plot_attainment or settings.plot_dominated_area:
        #     raise ValueError("Attainment surface and dominated area plotting is only supported for 2D problems")

        # # Get an axis if not supplied
        # if ax is None:
        #     ax = fig.add_subplot(111, projection="3d")

        # # Plot non-dominated solutions with high alpha
        # scatter = ax.scatter(nd_objs[:, 0], nd_objs[:, 1], nd_objs[:, 2], alpha=0.9, s=15)
        # base_color = scatter.get_facecolor()[0]  # Get the color that matplotlib assigned

        # # Plot dominated solutions with low alpha if requested
        # if settings.plot_dominated and len(dom_objs) > 0:
        #     ax.scatter(dom_objs[:, 0], dom_objs[:, 1], dom_objs[:, 2], color=base_color, alpha=0.25, s=15)

        # # Add in Pareto front
        # if pf is not None:
        #     ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c="k", s=10, label="PF")
        #     add_legend = True

        # # Handle the axis labels
        # if population.names_f:
        #     ax.set_xlabel(population.names_f[0])
        #     ax.set_ylabel(population.names_f[1])
        #     ax.set_zlabel(population.names_f[2])
        # else:
        #     ax.set_xlabel(r"$f_1$")
        #     ax.set_ylabel(r"$f_2$")
        #     ax.set_zlabel(r"$f_3$")

    # We can't plot in 4D :(
    else:
        raise ValueError(f"Cannot plot more than three objectives at the same time: n_objs={population.f.shape[1]}")

    if add_legend:
        plt.legend(loc=settings.legend_loc)

    return fig, ax


@dataclass
class PlotDecisionVarPairsSettings:
    """
    Settings related to plotting decision variables from `Population`.

    plot_dominated : bool, optional
        Include the dominated individuals, by default True
    plot_feasible : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    hist_bins : int, optional
        Number of bins for histograms on the diagonal, default is 20
    include_names : bool, optional
        Whether to include variable names on the axes if they exist, default is True
    problem : str/Problem, optional
        The problem for plotting decision variable bounds
    lower_bounds : array-like, optional
        Lower bounds for each decision variable
    upper_bounds : array-like, optional
        Upper bounds for each decision variable
    """

    plot_dominated: bool = True
    plot_feasible: Literal["all", "feasible", "infeasible"] = "all"
    hist_bins: Optional[int] = None
    include_names: bool = True
    problem: Optional[str] = None
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None


def plot_decision_var_pairs(
    population: Population, fig=None, axes=None, settings: PlotDecisionVarPairsSettings = PlotDecisionVarPairsSettings()
):
    """
    Creates a pairs plot (scatter matrix) showing correlations between decision variables
    and their distributions.

    Parameters
    ----------
    population : paretobench Population
        The population containing data to plot
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None and axes is None, creates a new figure.
    axes : numpy.ndarray of matplotlib.axes.Axes, optional
        2D array of axes to plot on. If None and fig is None, creates new axes.
        Must be provided if fig is provided and vice versa.
    settings : PlotDecisionVarPairsSettings
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

    n_vars = population.n

    # Default, don't show bounds
    lower_bounds = None
    upper_bounds = None

    # Handle user specified problem
    if settings.problem is not None:
        if (settings.lower_bounds is not None) or (settings.upper_bounds is not None):
            raise ValueError("Only specify one of problem or the upper/lower bounds")
        problem = get_problem_from_obj_or_str(settings.problem)
        if problem.n != n_vars:
            raise ValueError(
                f"Number of decision vars in problem must match number in population. Got {problem.n} in problem and {n_vars} in population"
            )
        lower_bounds = problem.var_lower_bounds
        upper_bounds = problem.var_upper_bounds

    # Validate and convert bounds to numpy arrays if provided
    if settings.lower_bounds is not None:
        lower_bounds = np.asarray(settings.lower_bounds)
        if len(lower_bounds) != n_vars:
            raise ValueError(f"Length of lower_bounds ({len(lower_bounds)}) must match number of variables ({n_vars})")

    if settings.upper_bounds is not None:
        upper_bounds = np.asarray(settings.upper_bounds)
        if len(upper_bounds) != n_vars:
            raise ValueError(f"Length of upper_bounds ({len(upper_bounds)}) must match number of variables ({n_vars})")

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
                # Hide x-axis labels and ticks for all rows except the bottom row
            if j != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

    # Get variable names or create default ones
    if population.names_x and settings.include_names:
        var_names = population.names_x
    else:
        var_names = [f"x{i+1}" for i in range(n_vars)]

    # Define bound line properties
    bound_props = dict(color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Break the objectives into those which are non-dominated and those which are not
    nd_inds = population.get_nondominated_indices()
    filt = np.ones(len(population), dtype=bool) if settings.plot_dominated else nd_inds  # Filter which are shown
    feas_inds = population.get_feasible_indices()

    # Get the domination ranks and set alpha based on that
    ranks = np.empty(len(population))
    for rank, idx in enumerate(fast_dominated_argsort(population.f, population.g)):
        ranks[idx] = rank
    if np.all(rank < 1):
        alpha = np.ones(len(population))
    else:
        alpha = 0.5 - ranks / ranks.max() * 0.3
        alpha[ranks == 0] = 1.0

    # Plot on all axes
    base_color = None
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            # Diagonal plots (histograms)
            if i == j:
                # Filter which individuals are shown
                if settings.plot_dominated:
                    filt = np.ones(len(population), dtype=bool)
                else:
                    filt = nd_inds
                if settings.plot_feasible == "feasible":
                    filt = np.bitwise_and(filt, feas_inds)
                elif settings.plot_feasible == "infeasible":
                    filt = np.bitwise_and(filt, ~feas_inds)

                # Plot the histogram
                _, _, patches = ax.hist(
                    population.x[filt, i], bins=settings.hist_bins, density=True, alpha=0.7, color=base_color
                )
                if base_color is None:
                    base_color = patches[0].get_facecolor()

                # Add vertical bound lines to histograms
                if lower_bounds is not None:
                    ax.axvline(lower_bounds[i], **bound_props)
                if upper_bounds is not None:
                    ax.axvline(upper_bounds[i], **bound_props)

            # Off-diagonal plots (scatter plots)
            else:
                # Feasible individuals
                if settings.plot_feasible in ["all", "feasible"]:
                    inds = np.bitwise_and(filt, feas_inds)
                    scatter = alpha_scatter(
                        ax, population.x[inds, j], population.x[inds, i], alpha=alpha[inds], color=base_color, s=15
                    )
                    if base_color is None:
                        base_color = scatter.get_facecolor()[0]  # Get the color that matplotlib assigned

                # Non-dominated infeasible individuals
                if settings.plot_feasible in ["all", "infeasible"]:
                    inds = np.bitwise_and(filt, ~feas_inds)
                    scatter = alpha_scatter(
                        ax,
                        population.x[inds, j],
                        population.x[inds, i],
                        color=base_color,
                        alpha=alpha[inds],
                        s=15,
                        marker="x",
                    )
                    if base_color is None:
                        base_color = scatter.get_facecolor()[0]  # Get the color that matplotlib assigned

                # Add bound lines to scatter plots
                if lower_bounds is not None:
                    ax.axvline(lower_bounds[j], **bound_props)  # x-axis bound
                    ax.axhline(lower_bounds[i], **bound_props)  # y-axis bound
                if upper_bounds is not None:
                    ax.axvline(upper_bounds[j], **bound_props)  # x-axis bound
                    ax.axhline(upper_bounds[i], **bound_props)  # y-axis bound
            if i == n_vars - 1:
                ax.set_xlabel(var_names[j])
            if j == 0:
                ax.set_ylabel(var_names[i])
    return fig, axes


def animate_objectives(
    history: History, interval: int = 200, objectives_plot_settings: PlotObjectivesSettings = PlotObjectivesSettings()
) -> animation.Animation:
    """
    Creates an animated visualization of how the Pareto front evolves across generations.

    Parameters
    ----------
    history : paretobench History
        The history object containing populations with data to plot
    interval : int, optional
        Delay between frames in milliseconds, by default 200
    problem : str, optional
        Name of the problem for true Pareto front plotting, by default None
    n_pf : int, optional
        Number of points for Pareto front plotting when applicable, by default 1000
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height), by default (8, 6)

    Returns
    -------
    animation.Animation
        The animation object that can be displayed in notebooks or saved to file

    Raises
    ------
    ValueError
        If the population has more than 3 objectives (cannot be visualized)
        If there are no reports in the history

    Notes
    -----
    This method creates an animation showing how the Pareto front evolves over time.
    For each frame:
    - Red points show the current population's non-dominated solutions
    - Gray points show dominated solutions
    - Black points show the true Pareto front (if problem is provided)

    The animation can be displayed in Jupyter notebooks using:
    >>> from IPython.display import HTML
    >>> HTML(anim.to_jshtml())

    Or saved to file using:
    >>> anim.save('filename.gif', writer='pillow')
    """
    if not history.reports:
        raise ValueError("No populations in history to animate")

    settings = copy(objectives_plot_settings)
    if settings.problem is None:
        settings.problem = history.problem
    if settings.legend_loc is None:
        settings.legend_loc = "upper right"

    # Get dimensions from first population
    n_objectives = history.reports[0].f.shape[1]
    if n_objectives > 3:
        raise ValueError(f"Cannot animate more than three objectives: n_objs={n_objectives}")

    # Set up the figure based on dimensionality
    fig = plt.figure()
    if n_objectives == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    # Ensure consistent axis limits across frames
    if n_objectives == 2:
        padding = 0.05
        all_f = np.vstack([pop.f for pop in history.reports])
        xlim = (np.min(all_f[:, 0]), np.max(all_f[:, 0]))
        ylim = (np.min(all_f[:, 1]), np.max(all_f[:, 1]))
        xlim = (xlim[0] - (xlim[1] - xlim[0]) * padding, xlim[1] + (xlim[1] - xlim[0]) * padding)
        ylim = (ylim[0] - (ylim[1] - ylim[0]) * padding, ylim[1] + (ylim[1] - ylim[0]) * padding)

    # Function to update frame for animation
    def update(frame_idx):
        ax.clear()
        population = history.reports[frame_idx]

        # Plot the current population
        plot_objectives(population, fig=fig, ax=ax, settings=settings)

        # Add generation counter
        generation = frame_idx + 1
        fevals = population.fevals
        ax.set_title(f"Generation {generation} (Fevals: {fevals})")

        # Scale the plot
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return (ax,)

    # Create and return the animation
    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(history.reports), interval=interval, blit=False)

    return anim


def animate_decision_vars(
    history: History,
    interval: int = 200,
    decision_var_plot_settings: PlotDecisionVarPairsSettings = PlotDecisionVarPairsSettings(),
) -> animation.Animation:
    """
    Creates an animated visualization of how the decision variables evolve across generations.

    Parameters
    ----------
    history : paretobench History
        The history object containing populations with data to plot
    interval : int, optional
        Delay between frames in milliseconds, by default 200
    decision_var_plot_settings : PlotDecisionVarPairsSettings
        Settings for the decision variable plots

    Returns
    -------
    animation.Animation
        The animation object that can be displayed in notebooks or saved to file
    """
    if not history.reports:
        raise ValueError("No populations in history to animate")

    settings = copy(decision_var_plot_settings)
    if settings.problem is None:
        settings.problem = history.problem

    # Create initial plot to get figure and axes
    fig, axes = plot_decision_var_pairs(history.reports[0], settings=settings)

    # Function to update frame for animation
    def update(frame_idx):
        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        population = history.reports[frame_idx]

        # Use the population's plotting method with existing fig and axes
        plot_decision_var_pairs(population, fig=fig, axes=axes, settings=settings)

        # Add generation counter
        generation = frame_idx + 1
        fevals = population.fevals
        fig.suptitle(f"Generation {generation} (Fevals: {fevals})")

        return tuple(axes.flat)

    # Create and return the animation
    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(history.reports), interval=interval, blit=False)

    return anim
