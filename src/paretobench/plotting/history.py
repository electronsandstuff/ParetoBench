from copy import copy
from dataclasses import dataclass
from matplotlib import animation
from typing import Optional, Tuple, Literal
import matplotlib.pyplot as plt
import numpy as np

from ..containers import History
from ..exceptions import EmptyPopulationError, NoObjectivesError
from .attainment import get_reference_point
from .population import PlotObjectivesSettings, plot_objectives, plot_decision_var_pairs, PlotDecisionVarPairsSettings


def plot_objectives_hist(
    history: History,
    idx=-1,
    fig=None,
    ax=None,
    settings: PlotObjectivesSettings = PlotObjectivesSettings(),
    cumulative=False,
    plot_pf=False,
):
    """
    Plot the objectives from a history object
    """
    # We need positive index in some places after this
    if idx < 0:
        idx = len(history.reports) + idx
        if idx < 0:
            raise IndexError(
                f"Negative index extends beyond length of history object (len(history) = {len(history.reports)})"
            )

    if cumulative:
        population = history.reports[0]
        for i in range(1, idx):
            population = population + history.reports[i]
    else:
        population = history.reports[idx]

    if plot_pf:
        settings.problem = history.problem

    return plot_objectives(population, fig=fig, ax=ax, settings=settings)


def plot_decision_var_pairs_hist(
    history: History,
    idx=-1,
    fig=None,
    axes=None,
    settings: PlotDecisionVarPairsSettings = PlotDecisionVarPairsSettings(),
    cumulative=False,
):
    """
    Plot the decision vars from a history object
    """
    # We need positive index in some places after this
    if idx < 0:
        idx = len(history.reports) + idx
        if idx < 0:
            raise IndexError(
                f"Negative index extends beyond length of history object (len(history) = {len(history.reports)})"
            )

    if cumulative:
        population = history.reports[0]
        for i in range(1, idx):
            population = population + history.reports[i]
    else:
        population = history.reports[idx]

    return plot_decision_var_pairs(population, fig=fig, axes=axes, settings=settings)


def animate_objectives(
    history: History,
    interval: int = 200,
    objectives_plot_settings: PlotObjectivesSettings = PlotObjectivesSettings(),
    dynamic_scaling: bool = False,
    cumulative: bool = False,
) -> animation.Animation:
    """
    Creates an animated visualization of how the Pareto front evolves across generations.

    Parameters
    ----------
    history : paretobench History
        The history object containing populations with data to plot
    interval : int, optional
        Delay between frames in milliseconds, by default 200
    objectives_plot_settings : PlotObjectivesSettings, optional
        Settings for plotting objectives
    dynamic_scaling : bool, optional
        If True, axes limits will update based on each frame's data.
        If False, axes limits will be fixed based on all data, by default False
    cumulative : bool, optional
        If True, shows all points seen up to current frame.
        If False, shows only current frame's points, by default False

    Returns
    -------
    animation.Animation
        The animation object that can be displayed in notebooks or saved to file
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

    # Calculate global axis limits if not using dynamic scaling
    if not dynamic_scaling and n_objectives == 2:
        padding = 0.05
        all_f = np.vstack([pop.f for pop in history.reports])
        xlim = (np.min(all_f[:, 0]), np.max(all_f[:, 0]))
        ylim = (np.min(all_f[:, 1]), np.max(all_f[:, 1]))
        xlim = (xlim[0] - (xlim[1] - xlim[0]) * padding, xlim[1] + (xlim[1] - xlim[0]) * padding)
        ylim = (ylim[0] - (ylim[1] - ylim[0]) * padding, ylim[1] + (ylim[1] - ylim[0]) * padding)

    # Function to update frame for animation
    def update(frame_idx):
        ax.clear()

        # Plot the objectives using the new function
        plot_objectives_hist(
            history,
            idx=frame_idx,
            fig=fig,
            ax=ax,
            settings=settings,
            cumulative=cumulative,
            plot_pf=True if settings.problem is not None else False,
        )

        # Add generation counter
        generation = frame_idx + 1
        fevals = history.reports[frame_idx].fevals
        ax.set_title(f"Generation {generation} (Fevals: {fevals})")

        if n_objectives == 2:
            if not dynamic_scaling:
                # Use global limits
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
    dynamic_scaling: bool = False,
    cumulative: bool = False,
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
    dynamic_scaling : bool, optional
        If True, axes limits will update based on each frame's data.
        If False, axes limits will be fixed based on all data, by default False
    cumulative : bool, optional
        If True, shows all points seen up to current frame.
        If False, shows only current frame's points, by default False

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

    # Calculate global axis limits if not using dynamic scaling
    if not dynamic_scaling:
        padding = 0.05
        all_x = np.vstack([pop.x for pop in history.reports])
        n_vars = all_x.shape[1]

        # Calculate limits for each variable
        var_limits = []
        for i in range(n_vars):
            var_min, var_max = np.min(all_x[:, i]), np.max(all_x[:, i])
            limit = (var_min - (var_max - var_min) * padding, var_max + (var_max - var_min) * padding)
            var_limits.append(limit)

    # Function to update frame for animation
    def update(frame_idx):
        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        # Plot using the history object plotting function
        plot_decision_var_pairs_hist(
            history,
            idx=frame_idx,
            fig=fig,
            axes=axes,
            settings=settings,
            cumulative=cumulative,
        )

        # Add generation counter
        generation = frame_idx + 1
        fevals = history.reports[frame_idx].fevals
        fig.suptitle(f"Generation {generation} (Fevals: {fevals})")

        if not dynamic_scaling:
            # Apply global limits to all subplots
            for i, ax in enumerate(axes.flat):
                row = i // axes.shape[1]
                col = i % axes.shape[1]
                if col < row:  # Skip lower triangle
                    continue
                if row != col:  # Only for pair plots
                    ax.set_xlim(*var_limits[col])
                    ax.set_ylim(*var_limits[row])

        return tuple(axes.flat)

    # Create and return the animation
    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(history.reports), interval=interval, blit=False)

    return anim


@dataclass
class PlotHistorySettings:
    """
    Settings for plotting the objective functions from a history of populations.

    plot_dominated : bool, optional
        Include the dominated individuals, by default True
    plot_feasible : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    plot_pf : bool, optional
        Whether to plot the Pareto front, by default True
    pf_objectives : array-like, optional
        User-specified Pareto front objectives
    colormap : str, optional
        Name of the colormap to use for generation colors, by default 'viridis'
    plot_attainment: bool = False
        Whether to plot attainment surfaces for each generation
    plot_dominated_area: bool = False
        Whether to plot the dominated area for each generation
    ref_point: Optional[Tuple[float, float]] = None
        Reference point for attainment surface calculation
    ref_point_padding: float = 0.05
        Padding for automatic reference point calculation
    label: Optional[str] = "Generation"
    legend_loc: Optional[str] = None
    """

    plot_dominated: Literal["all", "dominated", "non-dominated"] = "all"
    plot_feasible: Literal["all", "feasible", "infeasible"] = "all"
    plot_pf: bool = False
    pf_objectives: Optional[np.ndarray] = None
    colormap: str = "viridis"
    plot_attainment: bool = False
    plot_dominated_area: bool = False
    ref_point: Optional[Tuple[float, float]] = None
    ref_point_padding: float = 0.1
    label: Optional[str] = "Generation"
    legend_loc: Optional[str] = None


def plot_history(
    history,
    fig=None,
    ax=None,
    settings: PlotHistorySettings = PlotHistorySettings(),
):
    """
    Plot the objectives from a history of populations, using color to represent generations.

    Parameters
    ----------
    history : History object
        The history containing populations to plot
    fig : matplotlib figure, optional
        Figure to plot on, by default None
    ax : matplotlib axis, optional
        Axis to plot on, by default None
    settings : PlotHistorySettings
        Settings for the plot

    Returns
    -------
    matplotlib figure and matplotlib axis
        The figure and axis containing the history plot
    """
    # Basic input validation
    if not history.reports:
        raise ValueError("History contains no reports")

    # Get dimensions from first population
    first_pop = history.reports[0]
    if not len(first_pop):
        raise EmptyPopulationError()
    if not first_pop.m:
        raise NoObjectivesError()

    if fig is None:
        fig = plt.figure()

    # Calculate global reference point if not provided
    global_ref_point = None
    if settings.ref_point is None:
        # Combine all populations using addition operator
        combined_population = sum(history.reports[1:], history.reports[0])
        global_ref_point = get_reference_point(combined_population, padding=settings.ref_point_padding)
    else:
        global_ref_point = settings.ref_point

    # Create colormap for generations
    n_generations = len(history.reports)
    cmap = plt.get_cmap(settings.colormap)
    norm = plt.Normalize(0, n_generations - 1)

    # Create base settings for plot_objectives
    obj_settings = PlotObjectivesSettings(
        plot_dominated=settings.plot_dominated,
        plot_feasible=settings.plot_feasible,
        plot_attainment=settings.plot_attainment,
        plot_dominated_area=settings.plot_dominated_area,
        pf_objectives=settings.pf_objectives,
        ref_point=global_ref_point,
        legend_loc=settings.legend_loc,
    )

    # Plot each generation with its own color
    for gen_idx, population in enumerate(history.reports):
        color = cmap(norm(gen_idx))

        # Update settings for this generation
        obj_settings.color = color
        obj_settings.dominated_area_zorder = -1 - gen_idx

        # Only plot PF on the last generation if requested
        if gen_idx == n_generations - 1 and settings.plot_pf and history.problem is not None:
            obj_settings.problem = history.problem
        else:
            obj_settings.problem = None

        # Plot this generation
        fig, ax = plot_objectives(population, fig=fig, ax=ax, settings=obj_settings)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the colorbar
    fig.colorbar(sm, ax=ax, label=settings.label)

    return fig, ax
