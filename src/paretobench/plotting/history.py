from copy import copy
from dataclasses import dataclass
from matplotlib import animation
from typing import Optional, Tuple, Literal, Union, List
import matplotlib.pyplot as plt
import numpy as np

from ..containers import History
from ..exceptions import EmptyPopulationError, NoObjectivesError
from .attainment import get_reference_point
from .population import (
    PopulationObjScatterConfig,
    population_obj_scatter,
    population_dvar_pairs,
    PopulationDVarPairsConfig,
)


@dataclass
class HistoryObjScatterConfig:
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
        Label for colorbar (only used when generation_mode is 'cmap')
    legend_loc: Optional[str] = None
    generation_mode: Literal['cmap', 'cumulative'] = 'cmap'
        How to handle multiple generations:
        - 'cmap': Plot each generation separately with colors from colormap
        - 'cumulative': Merge all selected generations into single population
    single_color: Optional[str] = None
        Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
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
    generation_mode: Literal["cmap", "cumulative"] = "cmap"
    single_color: Optional[str] = None


def history_obj_scatter(
    history,
    select: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    fig=None,
    ax=None,
    settings: HistoryObjScatterConfig = HistoryObjScatterConfig(),
):
    """
    Plot the objectives from a history of populations, using either a colormap for generations
    or merging all generations into a single population.

    Parameters
    ----------
    history : History object
        The history containing populations to plot
    select : int, slice, List[int], or Tuple[int, int], optional
        Specifies which generations to plot. Can be:
        - None: All generations (default)
        - int: Single generation index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd gen)
        - List[int]: Explicit list of generation indices
        - Tuple[int, int]: Range of generations as (start, end) where end is exclusive
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

    # Handle different types of selection
    if select is None:
        # Select all populations
        indices = list(range(len(history.reports)))
    elif isinstance(select, int):
        # Single index - convert negative to positive
        if select < 0:
            select = len(history.reports) + select
            if select < 0:
                raise IndexError(f"Index {select} out of range for history with length {len(history.reports)}")
        indices = [select]
    elif isinstance(select, slice):
        # Slice - get list of indices
        indices = list(range(*select.indices(len(history.reports))))
    elif isinstance(select, (list, np.ndarray)):
        # List of indices - convert negative to positive
        indices = []
        for i in select:
            idx = i if i >= 0 else len(history.reports) + i
            if idx < 0 or idx >= len(history.reports):
                raise IndexError(f"Index {i} out of range for history with length {len(history.reports)}")
            indices.append(idx)
    elif isinstance(select, tuple) and len(select) == 2:
        # Range tuple (start, end)
        start, end = select
        if start < 0 or end > len(history.reports):
            raise IndexError(f"Range {start}:{end} out of bounds for history with length {len(history.reports)}")
        indices = list(range(start, end))
    else:
        raise ValueError(f"Unsupported selection type: {type(select)}")

    if not indices:
        raise ValueError("No generations selected")

    # Get dimensions from first population
    first_pop = history.reports[0]
    if not len(first_pop):
        raise EmptyPopulationError()
    if not first_pop.m:
        raise NoObjectivesError()

    if fig is None:
        fig = plt.figure()

    # Create base settings for population_obj_scatter
    obj_settings = PopulationObjScatterConfig(
        plot_dominated=settings.plot_dominated,
        plot_feasible=settings.plot_feasible,
        plot_attainment=settings.plot_attainment,
        plot_dominated_area=settings.plot_dominated_area,
        pf_objectives=settings.pf_objectives,
        legend_loc=settings.legend_loc,
    )

    # Calculate global reference point if not provided
    if settings.ref_point is None:
        combined_population = history.reports[indices[0]]
        for idx in indices[1:]:
            combined_population = combined_population + history.reports[idx]
        obj_settings.ref_point = get_reference_point(combined_population, padding=settings.ref_point_padding)
    else:
        obj_settings.ref_point = settings.ref_point

    if settings.generation_mode == "cumulative":
        # Merge all selected populations
        combined_population = history.reports[indices[0]]
        for idx in indices[1:]:
            combined_population = combined_population + history.reports[idx]

        # Set optional color and plot combined population
        obj_settings.color = settings.single_color  # Will use default if None
        if settings.plot_pf and history.problem is not None:
            obj_settings.problem = history.problem

        fig, ax = population_obj_scatter(combined_population, fig=fig, ax=ax, settings=obj_settings)

    elif settings.generation_mode == "cmap":
        cmap = plt.get_cmap(settings.colormap)
        norm = plt.Normalize(min(indices), max(indices))

        # Plot each selected generation
        for plot_idx, gen_idx in enumerate(indices):
            population = history.reports[gen_idx]
            obj_settings.color = cmap(norm(gen_idx))
            obj_settings.dominated_area_zorder = -2 - plot_idx

            # Only plot PF on the last iteration if requested
            if plot_idx == len(indices) - 1 and settings.plot_pf and history.problem is not None:
                obj_settings.problem = history.problem
            else:
                obj_settings.problem = None

            # Plot this generation
            fig, ax = population_obj_scatter(population, fig=fig, ax=ax, settings=obj_settings)

        # Add colorbar if label is provided
        if settings.label:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Dummy array for the colorbar
            fig.colorbar(sm, ax=ax, label=settings.label)
    else:
        raise ValueError(f"Unrecognized generation mode: {settings.generation_mode}")

    return fig, ax


@dataclass
class HistoryDVarPairsConfig:
    """
    Settings for plotting the decision variables from a history of populations.

    plot_dominated : Literal['all', 'dominated', 'non-dominated'], optional
        Include all, only dominated, or only non-dominated solutions, by default 'all'
    plot_feasible : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    colormap : str, optional
        Name of the colormap to use for generation colors, by default 'viridis'
    label: Optional[str] = "Generation"
        Label for colorbar (only used when generation_mode is 'cmap')
    legend_loc: Optional[str] = None
    generation_mode: Literal['cmap', 'cumulative'] = 'cmap'
        How to handle multiple generations:
        - 'cmap': Plot each generation separately with colors from colormap
        - 'cumulative': Merge all selected generations into single population
    single_color: Optional[str] = None
        Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
    plot_bounds: bool = False
        Whether to plot bounds for the problem
    """

    plot_dominated: Literal["all", "dominated", "non-dominated"] = "all"
    plot_feasible: Literal["all", "feasible", "infeasible"] = "all"
    colormap: str = "viridis"
    label: Optional[str] = "Generation"
    generation_mode: Literal["cmap", "cumulative"] = "cmap"
    single_color: Optional[str] = None
    plot_bounds: bool = False


def history_dvar_pairs(
    history: History,
    select: Optional[Union[int, slice, List[int], tuple[int, int]]] = None,
    fig=None,
    axes=None,
    settings: HistoryDVarPairsConfig = HistoryDVarPairsConfig(),
):
    """
    Plot the decision variables from a history of populations, using either a colormap
    for generations or merging all generations into a single population.

    Parameters
    ----------
    history : History object
        The history containing populations to plot
    select : int, slice, List[int], or Tuple[int, int], optional
        Specifies which generations to plot. Can be:
        - None: All generations (default)
        - int: Single generation index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd gen)
        - List[int]: Explicit list of generation indices
        - Tuple[int, int]: Range of generations as (start, end) where end is exclusive
    fig : matplotlib figure, optional
        Figure to plot on, by default None
    axes : array of matplotlib axes, optional
        Axes to plot on, by default None
    settings : HistoryDVarPairsConfig
        Settings for the plot

    Returns
    -------
    matplotlib figure and array of matplotlib axes
        The figure and axes containing the history plot
    """
    # Basic input validation
    if not history.reports:
        raise ValueError("History contains no reports")

    # Handle different types of selection
    if select is None:
        # Select all populations
        indices = list(range(len(history.reports)))
    elif isinstance(select, int):
        # Single index - convert negative to positive
        if select < 0:
            select = len(history.reports) + select
            if select < 0:
                raise IndexError(f"Index {select} out of range for history with length {len(history.reports)}")
        indices = [select]
    elif isinstance(select, slice):
        # Slice - get list of indices
        indices = list(range(*select.indices(len(history.reports))))
    elif isinstance(select, (list, np.ndarray)):
        # List of indices - convert negative to positive
        indices = []
        for i in select:
            idx = i if i >= 0 else len(history.reports) + i
            if idx < 0 or idx >= len(history.reports):
                raise IndexError(f"Index {i} out of range for history with length {len(history.reports)}")
            indices.append(idx)
    elif isinstance(select, tuple) and len(select) == 2:
        # Range tuple (start, end)
        start, end = select
        if start < 0 or end > len(history.reports):
            raise IndexError(f"Range {start}:{end} out of bounds for history with length {len(history.reports)}")
        indices = list(range(start, end))
    else:
        raise ValueError(f"Unsupported selection type: {type(select)}")

    if not indices:
        raise ValueError("No generations selected")

    # Get dimensions from first population
    first_pop = history.reports[0]
    if not len(first_pop):
        raise EmptyPopulationError()

    # Create base settings for population_dvar_pairs
    plot_settings = PopulationDVarPairsConfig(
        plot_dominated=settings.plot_dominated,
        plot_feasible=settings.plot_feasible,
    )

    if settings.generation_mode == "cumulative":
        # Merge all selected populations
        combined_population = history.reports[indices[0]]
        for idx in indices[1:]:
            combined_population = combined_population + history.reports[idx]

        if settings.plot_bounds and history.problem is not None:
            plot_settings.problem = history.problem

        # Set optional color and plot combined population
        plot_settings.color = settings.single_color  # Will use default if None
        fig, axes = population_dvar_pairs(combined_population, fig=fig, axes=axes, settings=plot_settings)

    elif settings.generation_mode == "cmap":
        cmap = plt.get_cmap(settings.colormap)
        norm = plt.Normalize(min(indices), max(indices))

        # Plot each selected generation
        for plot_idx, gen_idx in enumerate(indices):
            population = history.reports[gen_idx]
            plot_settings.color = cmap(norm(gen_idx))

            # Only plot PF on the last iteration if requested
            if plot_idx == len(indices) - 1 and settings.plot_bounds and history.problem is not None:
                plot_settings.problem = history.problem
            else:
                plot_settings.problem = None

            # Plot this generation
            fig, axes = population_dvar_pairs(population, fig=fig, axes=axes, settings=plot_settings)

        # Add colorbar if label is provided
        if settings.label:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Dummy array for the colorbar
            # Use ax parameter to let constrained_layout handle the positioning
            fig.colorbar(sm, ax=axes.ravel().tolist(), label=settings.label)
    else:
        raise ValueError(f"Unrecognized generation mode: {settings.generation_mode}")

    return fig, axes


def history_obj_animation(
    history: History,
    interval: int = 200,
    objectives_plot_settings: HistoryObjScatterConfig = HistoryObjScatterConfig(),
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
    objectives_plot_settings : HistoryObjScatterConfig, optional
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

        # Configure settings for this frame
        frame_settings = copy(settings)
        frame_settings.generation_mode = "cumulative"

        # Plot using the new history plotting function
        history_obj_scatter(
            history,
            select=slice(0, frame_idx + 1) if cumulative else slice(frame_idx, frame_idx + 1),
            fig=fig,
            ax=ax,
            settings=frame_settings,
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


def history_dvar_animation(
    history: History,
    interval: int = 200,
    decision_var_plot_settings: HistoryDVarPairsConfig = HistoryDVarPairsConfig(),
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
    decision_var_plot_settings : HistoryDVarPairsConfig
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

    # Create initial plot to get figure and axes
    settings.generation_mode = "cumulative"
    fig, axes = history_dvar_pairs(history, select=0, settings=settings)

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

        # Configure settings for this frame
        frame_settings = copy(settings)
        frame_settings.generation_mode = "cumulative"

        # Plot using the new history plotting function
        history_dvar_pairs(
            history,
            select=slice(0, frame_idx + 1) if cumulative else slice(frame_idx, frame_idx + 1),
            fig=fig,
            axes=axes,
            settings=frame_settings,
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
