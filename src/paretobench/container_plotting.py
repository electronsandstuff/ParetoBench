import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib import animation
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from copy import copy

from .containers import Population, History
from .problem import ProblemWithFixedPF, ProblemWithPF, get_problem_from_obj_or_str
from .exceptions import EmptyPopulationError, NoConstraintsError, NoDecisionVarsError


# TODO: Add better error when no decision vars in population and no individuals
# TODO: 3D attainment surface (should 3D and 2D plotting be broken up since they can have different features?)
# TODO: Make sure we are compatible with plotting of multiple fronts
# TODO: standardize plot name "pareto front" vs "objective functions"
# TODO: move limits calculation to start of objectives animation instead of in body
# TODO: look into make pair plot more compact
# TODO: Use rule of thumb for number of histogram bins


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
    plot_dominated: bool = True
    problem: Optional[str] = None
    n_pf: int = 1000
    pf_objectives: Optional[np.ndarray] = None
    plot_attainment: bool = False
    plot_dominated_area: bool = False
    ref_point: Union[str, Tuple[float, float]] = "auto"


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
    plot_dominated : bool, optional
        Include the dominated individuals, by default True
    problem : str, optional
        Name of the problem for Pareto front plotting, by default None
    n_pf : int, optional
        The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
    pf_objectives : array-like, optional
        User-specified Pareto front objectives. Should be a 2D array where each row represents a point
        on the Pareto front and each column represents an objective value.
    plot_attainment : bool, optional
        Whether to plot the attainment surface (2D only), by default False

    Raises
    ------
    ValueError
        If both problem and pf_objectives are specified
        If neither problem nor pf_objectives are specified when trying to plot a Pareto front
        If pf_objectives dimensions don't match the population's objectives
        If attempting to plot more than three objectives
        If attempting to plot attainment surface for 3D problem
    """
    # Make sure we have been given data to plot
    if not len(population):
        raise EmptyPopulationError()
    if not population.m:
        raise NoObjectivesError()

    # Input validation for Pareto front specification
    pf_sources_specified = sum(x is not None for x in [settings.problem, settings.pf_objectives])
    if pf_sources_specified > 1:
        raise ValueError(
            "Multiple Pareto front sources specified. Use only one of: 'problem' or 'pf_objectives'"
        )

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Break the objectives into those which are non-dominated and those which are not
    nd_inds = population.get_nondominated_indices()
    nd_objs = population.f[nd_inds, :]
    dom_objs = population.f[~nd_inds, :]

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

    # For 2D problems
    if population.f.shape[1] == 2:
        # Make axis if not supplied
        if ax is None:
            ax = fig.add_subplot(111)

        # Plot dominated solutions with low alpha if requested
        if settings.plot_dominated and len(dom_objs) > 0:
            ax.scatter(dom_objs[:, 0], dom_objs[:, 1], color='C0', alpha=0.25, s=15)
        
        # Plot non-dominated solutions with high alpha
        if len(nd_objs) > 0:
            ax.scatter(nd_objs[:, 0], nd_objs[:, 1], color='C0', alpha=0.9, s=15)

        # Plot attainment surface if requested
        if settings.plot_attainment and len(nd_objs) > 0:
            surface_points = compute_attainment_surface(nd_objs)
            ax.plot(surface_points[:, 0], surface_points[:, 1], 
                'C0', alpha=0.5, label='Attainment Surface')
            plt.legend()

        if settings.plot_dominated_area:
            if settings.ref_point == 'auto':
                if settings.plot_dominated:
                    x_lb, x_ub = np.min(population.f[:, 0]), np.max(population.f[:, 0])
                    y_lb, y_ub = np.min(population.f[:, 1]), np.max(population.f[:, 1])
                    ref_point = (x_ub + (x_ub - x_lb)*0.05, y_ub + (y_ub - y_lb)*0.05)
                else:
                    x_lb, x_ub = np.min(nd_objs[:, 0]), np.max(nd_objs[:, 0])
                    y_lb, y_ub = np.min(nd_objs[:, 1]), np.max(nd_objs[:, 1])
                    ref_point = (x_ub + (x_ub - x_lb)*0.1, y_ub + (y_ub - y_lb)*0.1)
            else:
                ref_point = settings.ref_point

            attainment = compute_attainment_surface(nd_objs)
            if (ref_point[0] < attainment[:, 0]).any() or (ref_point[1] < attainment[:, 1]).any():
                raise ValueError(f"Reference point coordinates must exceed all points in non-dominated set "
                                f"(ref_point={ref_point}, max_pf=({np.max(attainment[:, 0])}, {np.max(attainment[:, 1])}))")

            plt.fill_between(
                np.concatenate((attainment[:, 0], [ref_point[0]])), 
                np.concatenate((attainment[:, 1], [attainment[-1, 1]])), 
                ref_point[1]*np.ones(attainment.shape[0]+1),
                alpha=0.5
            )

        # Add in Pareto front
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], c="k", s=10, label="PF")
            plt.legend()

        # Handle the axis labels
        if population.names_f:
            ax.set_xlabel(population.names_f[0])
            ax.set_ylabel(population.names_f[1])
        else:
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$f_2$")

    # For 3D problems
    elif population.f.shape[1] == 3:
        if settings.plot_attainment or settings.plot_dominated_area:
            raise ValueError("Attainment surface and dominated area plotting is only supported for 2D problems")
            
        # Get an axis if not supplied
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")

        # Plot dominated solutions with low alpha if requested
        if settings.plot_dominated and len(dom_objs) > 0:
            ax.scatter(dom_objs[:, 0], dom_objs[:, 1], dom_objs[:, 2], 
                    color='C0', alpha=0.25, s=15)
        
        # Plot non-dominated solutions with high alpha
        if len(nd_objs) > 0:
            ax.scatter(nd_objs[:, 0], nd_objs[:, 1], nd_objs[:, 2], color='C0', alpha=0.9, s=15)

        # Add in Pareto front
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c="k", s=10, label="PF")
            plt.legend()

        # Handle the axis labels
        if population.names_f:
            ax.set_xlabel(population.names_f[0])
            ax.set_ylabel(population.names_f[1])
            ax.set_zlabel(population.names_f[2])
        else:
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$f_2$")
            ax.set_zlabel(r"$f_3$")

    # We can't plot in 4D :(
    else:
        raise ValueError(f"Cannot plot more than three objectives at the same time: n_objs={population.f.shape[1]}")

    return fig, ax


@dataclass
class PlotDecisionVarPairsSettings:
    hist_bins: int = 20
    include_names: bool = True
    problem: Optional[str] = None
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None


def plot_decision_var_pairs(population: Population, fig=None, settings: PlotDecisionVarPairsSettings=PlotDecisionVarPairsSettings()):
    """
    Creates a pairs plot (scatter matrix) showing correlations between decision variables
    and their distributions.

    Parameters
    ----------
    population : paretobench Population
        The population containing data to plot
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, creates a new figure.
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

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the pairs plot

    Examples
    --------
    >>> pop = Population.from_random(n_objectives=2, n_decision_vars=4,
                                n_constraints=0, pop_size=100)
    >>> lb = [0, 0, 0, 0]
    >>> ub = [1, 1, 1, 1]
    >>> fig = pop.plot_pairs(lower_bounds=lb, upper_bounds=ub)
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
            raise ValueError(
                f"Length of lower_bounds ({len(lower_bounds)}) must match number of variables ({n_vars})"
            )

    if settings.upper_bounds is not None:
        upper_bounds = np.asarray(settings.upper_bounds)
        if len(upper_bounds) != n_vars:
            raise ValueError(
                f"Length of upper_bounds ({len(upper_bounds)}) must match number of variables ({n_vars})"
            )

    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(2 * n_vars, 2 * n_vars))

    # Create gridspec for the layout
    gs = gridspec.GridSpec(n_vars, n_vars)
    gs.update(wspace=0.3, hspace=0.3)

    # Get variable names or create default ones
    if population.names_x and settings.include_names:
        var_names = population.names_x
    else:
        var_names = [f"x{i+1}" for i in range(n_vars)]

    # Define bound line properties
    bound_props = dict(color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Create all subplots
    for i in range(n_vars):
        for j in range(n_vars):
            ax = fig.add_subplot(gs[i, j])

            # Diagonal plots (histograms)
            if i == j:
                ax.hist(population.x[:, i], bins=settings.hist_bins, density=True, alpha=0.7, color="gray")
                if i == n_vars - 1:
                    ax.set_xlabel(var_names[i])
                if j == 0:
                    ax.set_ylabel("Density")

                # Add vertical bound lines to histograms
                if lower_bounds is not None:
                    ax.axvline(lower_bounds[i], **bound_props)
                if upper_bounds is not None:
                    ax.axvline(upper_bounds[i], **bound_props)

            # Off-diagonal plots (scatter plots)
            else:
                ax.scatter(population.x[:, j], population.x[:, i], alpha=0.5, s=20)
                if i == n_vars - 1:
                    ax.set_xlabel(var_names[j])
                if j == 0:
                    ax.set_ylabel(var_names[i])

                # Add bound lines to scatter plots
                if lower_bounds is not None:
                    ax.axvline(lower_bounds[j], **bound_props)  # x-axis bound
                    ax.axhline(lower_bounds[i], **bound_props)  # y-axis bound
                if upper_bounds is not None:
                    ax.axvline(upper_bounds[j], **bound_props)  # x-axis bound
                    ax.axhline(upper_bounds[i], **bound_props)  # y-axis bound

            # Remove top and right spines for cleaner look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Use fewer ticks for cleaner appearance
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            # Only show ticks on bottom and left
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

    return fig


def animate_pareto_front(
    history: History,
    interval: int = 200,
    objectives_plot_settings: PlotObjectivesSettings = PlotObjectivesSettings()
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
        
    # Get dimensions from first population
    n_objectives = history.reports[0].f.shape[1]
    if n_objectives > 3:
        raise ValueError(f"Cannot animate more than three objectives: n_objs={n_objectives}")
    
    # Set up the figure based on dimensionality
    fig = plt.figure()
    if n_objectives == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    # Function to update frame for animation
    def update(frame_idx):
        ax.clear()
        population = history.reports[frame_idx]
        
        # Plot the current population
        plot_objectives(
            population,
            fig=fig,
            ax=ax,
            settings=settings
        )
        
        # Add generation counter
        generation = frame_idx + 1
        fevals = population.fevals
        ax.set_title(f'Generation {generation} (Fevals: {fevals})')
        
        # Ensure consistent axis limits across frames
        if n_objectives == 2:
            all_f = np.vstack([pop.f for pop in history.reports])
            ax.set_xlim(np.min(all_f[:, 0]) * 0.9, np.max(all_f[:, 0]) * 1.1)
            ax.set_ylim(np.min(all_f[:, 1]) * 0.9, np.max(all_f[:, 1]) * 1.1)
        
        return ax,
    
    # Create and return the animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(history.reports),
        interval=interval,
        blit=False
    )
    
    return anim


def animate_decision_vars(
    history: History,
    interval: int = 200,
    decision_var_plot_settings: PlotDecisionVarPairsSettings = PlotDecisionVarPairsSettings
) -> animation.Animation:
    """
    Creates an animated visualization of how the decision variables evolve across generations.
    
    Parameters
    ----------
    history : paretobench History
        The history object containing populations with data to plot
    interval : int, optional
        Delay between frames in milliseconds, by default 200
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
        
    Returns
    -------
    animation.Animation
        The animation object that can be displayed in notebooks or saved to file
        
    Raises
    ------
    ValueError
        If there are no reports in the history
        If both problem and bounds are specified
        If bounds dimensions don't match
        
    Notes
    -----
    This method creates an animation showing how the decision variables evolve over time.
    For each frame:
    - Scatter plots show relationships between pairs of variables
    - Histograms on the diagonal show distribution of each variable
    - Red dashed lines show variable bounds if provided
    
    The animation can be displayed in Jupyter notebooks using:
    >>> from IPython.display import HTML
    >>> HTML(anim.to_jshtml())
    """
    if not history.reports:
        raise ValueError("No populations in history to animate")

    settings = copy(decision_var_plot_settings) 
    if settings.problem is None:
        settings.problem = history.problem
    
    # Get dimensions from first population
    n_vars = history.reports[0].x.shape[1]
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(2 * n_vars, 2 * n_vars))
    
    # Function to update frame for animation
    def update(frame_idx):
        fig.clear()
        population = history.reports[frame_idx]
        
        # Use the population's plotting method
        plot_decision_var_pairs(
            population,
            fig=fig,
            settings=settings
        )
        
        # Add generation counter
        generation = frame_idx + 1
        fevals = population.fevals
        fig.suptitle(f'Generation {generation} (Fevals: {fevals})')
        
        return fig,
    
    # Create and return the animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(history.reports),
        interval=interval,
        blit=False
    )
    
    return anim
