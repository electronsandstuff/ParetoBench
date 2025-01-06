from matplotlib import animation
from typing import Optional, Tuple, Literal, Union, List
import matplotlib.pyplot as plt
import numpy as np

from ..containers import History
from ..exceptions import EmptyPopulationError, NoObjectivesError
from .attainment import get_reference_point
from .population import (
    population_obj_scatter,
    population_dvar_pairs,
)
from .utils import selection_to_indices


def history_obj_scatter(
    history,
    reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    fig=None,
    ax=None,
    domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
    show_points: bool = True,
    n_pf: int = 1000,
    pf_objectives: Optional[np.ndarray] = None,
    show_attainment: bool = False,
    show_dominated_area: bool = False,
    ref_point: Optional[Tuple[float, float]] = None,
    ref_point_padding: float = 0.05,
    legend_loc: Optional[str] = None,
    show_names: bool = True,
    show_pf: bool = False,
    colormap: str = "viridis",
    cmap_label: Optional[str] = None,
    generation_mode: Literal["cmap", "cumulative"] = "cmap",
    single_color: Optional[str] = None,
    label_mode: Literal["index", "fevals"] = "index",
):
    """
    Plot the objectives from a history of populations, using either a colormap for generations
    or merging all generations into a single population.

    Parameters
    ----------
    history : History object
        The history containing populations to plot
    reports : int, slice, List[int], or Tuple[int, int], optional
        Specifies which generations to plot. Can be:
        - None: All generations (default)
        - int: Single generation index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd gen)
        - List[int]: Explicit list of generation indices
        - List[bool] or np.ndarray of bools: boolean mask where True selects the index
        - Tuple[int, int]: Range of generations as (start, end) where end is exclusive
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
    n_pf : int, optional
        The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
    pf_objectives : array-like, optional
        User-specified Pareto front objectives. Should be a 2D array where each row represents a point
        on the Pareto front and each column represents an objective value.
    show_attainment : bool, optional
        Whether to plot the attainment surface, by default False
    show_dominated_area : bool, optional
        Plots the dominated region towards the larger values of each decision var
    ref_point : Union[str, Tuple[float, float]], optional
        Where to stop plotting the dominated region / attainment surface. Must be a point to the upper right (increasing
        value of objectives in 3D) of all plotted points. By default, will set to right of max of each objective plus
        padding.
    ref_point_padding : float
        Amount of padding to apply to the automatic reference point calculation.
    legend_loc : str, optional
        Passed to `loc` argument of plt.legend
    show_names : bool, optional
        Whether to show the names of the objectives if provided by population
    show_pf : bool, optional
        Whether to plot the Pareto front, by default True
    colormap : str, optional
        Name of the colormap to use for generation colors, by default 'viridis'
    cmap_label: Optional[str] = "Generation"
        Label for colorbar (only used when generation_mode is 'cmap')
    generation_mode: Literal['cmap', 'cumulative'] = 'cmap'
        How to handle multiple generations:
        - 'cmap': Plot each generation separately with colors from colormap
        - 'cumulative': Merge all selected generations into single population
    single_color: Optional[str] = None
        Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
    label_mode: Literal['index', 'fevals'] = 'index'
        Whether to use report index or function evaluations (fevals) for labels

    Returns
    -------
    matplotlib figure and matplotlib axis
        The figure and axis containing the history plot
    """
    # Basic input validation
    if not history.reports:
        raise ValueError("History contains no reports")

    # Handle different types of selection
    indices = selection_to_indices(reports, len(history.reports))

    if not indices:
        raise ValueError("No reports selected")

    # Get dimensions from first population
    first_pop = history.reports[0]
    if not len(first_pop):
        raise EmptyPopulationError()
    if not first_pop.m:
        raise NoObjectivesError()

    if fig is None:
        fig = plt.figure()

    # Create base settings for population_obj_scatter
    obj_settings = dict(
        domination_filt=domination_filt,
        feasibility_filt=feasibility_filt,
        show_points=show_points,
        n_pf=n_pf,
        show_names=show_names,
        show_attainment=show_attainment,
        show_dominated_area=show_dominated_area,
        pf_objectives=pf_objectives,
        legend_loc=legend_loc,
    )

    # Calculate global reference point if not provided
    if ref_point is None:
        combined_population = history.reports[indices[0]]
        for idx in indices[1:]:
            combined_population = combined_population + history.reports[idx]
        obj_settings["ref_point"] = get_reference_point(combined_population, padding=ref_point_padding)
    else:
        obj_settings["ref_point"] = ref_point

    if generation_mode == "cumulative":
        # Merge all selected populations
        combined_population = history.reports[indices[0]]
        for idx in indices[1:]:
            combined_population = combined_population + history.reports[idx]

        # Set optional color and plot combined population
        obj_settings["color"] = single_color  # Will use default if None
        if show_pf and pf_objectives is not None:
            obj_settings["pf_objectives"] = pf_objectives
        elif show_pf and history.problem is not None:
            obj_settings["problem"] = history.problem

        fig, ax = population_obj_scatter(combined_population, fig=fig, ax=ax, **obj_settings)

    elif generation_mode == "cmap":
        if label_mode == "index":
            norm_values = indices
            label = cmap_label if cmap_label else "Generation"
        else:  # fevals mode
            norm_values = [history.reports[idx].fevals for idx in indices]
            label = cmap_label if cmap_label else "Function Evaluations"

        cmap = plt.get_cmap(colormap)
        norm = plt.Normalize(min(norm_values), max(norm_values))

        # Plot each selected generation
        for plot_idx, gen_idx in enumerate(indices):
            population = history.reports[gen_idx]
            norm_value = gen_idx if label_mode == "index" else population.fevals
            obj_settings["color"] = cmap(norm(norm_value))
            obj_settings["dominated_area_zorder"] = -2 - plot_idx

            # Only plot PF on the last iteration if requested
            if plot_idx == len(indices) - 1 and show_pf and pf_objectives is not None:
                obj_settings["pf_objectives"] = pf_objectives
            elif plot_idx == len(indices) - 1 and show_pf and history.problem is not None:
                obj_settings["problem"] = history.problem
            else:
                obj_settings["pf_objectives"] = None
                obj_settings["problem"] = None

            # Plot this generation
            fig, ax = population_obj_scatter(population, fig=fig, ax=ax, **obj_settings)

        # Add colorbar if label is provided
        if label:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Dummy array for the colorbar
            fig.colorbar(sm, ax=ax, label=label)
    else:
        raise ValueError(f"Unrecognized generation mode: {generation_mode}")

    return fig, ax


def history_dvar_pairs(
    history: History,
    reports: Optional[Union[int, slice, List[int], tuple[int, int]]] = None,
    dvars: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    fig=None,
    axes=None,
    domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
    hist_bins: Optional[int] = None,
    show_names: bool = True,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    colormap: str = "viridis",
    cmap_label: Optional[str] = None,
    generation_mode: Literal["cmap", "cumulative"] = "cmap",
    single_color: Optional[str] = None,
    plot_bounds: bool = False,
    label_mode: Literal["index", "fevals"] = "index",
):
    """
    Plot the decision variables from a history of populations, using either a colormap
    for generations or merging all generations into a single population.

    Parameters
    ----------
    history : History object
        The history containing populations to plot
    reports : int, slice, List[int], or Tuple[int, int], optional
        Specifies which generations to plot. Can be:
        - None: All generations (default)
        - int: Single generation index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd gen)
        - List[int]: Explicit list of generation indices
        - List[bool] or np.ndarray of bools: boolean mask where True selects the index
        - Tuple[int, int]: Range of generations as (start, end) where end is exclusive
    dvars : int, slice, List[int], or Tuple[int, int], optional
        Which decision vars to plot. See `population_dvar_pairs` docstring for more details.
    fig : matplotlib figure, optional
        Figure to plot on, by default None
    axes : array of matplotlib axes, optional
        Axes to plot on, by default None
    domination_filt : Literal["all", "dominated", "non-dominated"], optional
        Plot only the dominated/non-dominated solutions, or all. Defaults to all
    feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    hist_bins : int, optional
        Number of bins for histograms on the diagonal, default is to let matplotlib choose
    show_names : bool, optional
        Whether to include variable names on the axes if they exist, default is True
    lower_bounds : array-like, optional
        Lower bounds for each decision variable
    upper_bounds : array-like, optional
        Upper bounds for each decision variable
    colormap : str, optional
        Name of the colormap to use for generation colors, by default 'viridis'
    cmap_label: Optional[str] = "Generation"
        Label for colorbar (only used when generation_mode is 'cmap')
    generation_mode: Literal['cmap', 'cumulative'] = 'cmap'
        How to handle multiple generations:
        - 'cmap': Plot each generation separately with colors from colormap
        - 'cumulative': Merge all selected generations into single population
    single_color: Optional[str] = None
        Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
    plot_bounds: bool = False
        Whether to plot bounds for the problem
    label_mode: Literal['index', 'fevals'] = 'index'
        Whether to use report index or function evaluations (fevals) for labels

    Returns
    -------
    matplotlib figure and array of matplotlib axes
        The figure and axes containing the history plot
    """
    # Basic input validation
    if not history.reports:
        raise ValueError("History contains no reports")

    # Handle different types of selection
    indices = selection_to_indices(reports, len(history.reports))

    if not indices:
        raise ValueError("No generations selected")

    # Get dimensions from first population
    first_pop = history.reports[0]
    if not len(first_pop):
        raise EmptyPopulationError()

    # Check if the user gave us bounds to use
    user_specified_bounds = (lower_bounds is not None) or (upper_bounds is not None)

    # Create base settings for population_dvar_pairs
    plot_settings = dict(
        domination_filt=domination_filt,
        feasibility_filt=feasibility_filt,
        hist_bins=hist_bins,
        show_names=show_names,
    )

    if generation_mode == "cumulative":
        # Merge all selected populations
        combined_population = history.reports[indices[0]]
        for idx in indices[1:]:
            combined_population = combined_population + history.reports[idx]

        # Deal with passing along the bounds
        if plot_bounds and user_specified_bounds:
            plot_settings["lower_bounds"] = lower_bounds
            plot_settings["upper_bounds"] = upper_bounds
        elif plot_bounds and history.problem is not None:
            plot_settings["problem"] = history.problem

        # Set optional color and plot combined population
        plot_settings["color"] = single_color  # Will use default if None
        fig, axes = population_dvar_pairs(combined_population, dvars=dvars, fig=fig, axes=axes, **plot_settings)

    elif generation_mode == "cmap":
        if label_mode == "index":
            norm_values = indices
            label = cmap_label if cmap_label else "Generation"
        else:  # fevals mode
            norm_values = [history.reports[idx].fevals for idx in indices]
            label = cmap_label if cmap_label else "Function Evaluations"

        cmap = plt.get_cmap(colormap)
        norm = plt.Normalize(min(norm_values), max(norm_values))

        # Plot each selected generation
        for plot_idx, gen_idx in enumerate(indices):
            population = history.reports[gen_idx]
            norm_value = gen_idx if label_mode == "index" else population.fevals
            plot_settings["color"] = cmap(norm(norm_value))

            # Only plot bounds on the last iteration if requested
            if plot_idx == len(indices) - 1:
                if plot_bounds and user_specified_bounds:
                    plot_settings["lower_bounds"] = lower_bounds
                    plot_settings["upper_bounds"] = upper_bounds
                elif plot_bounds and history.problem is not None:
                    plot_settings["problem"] = history.problem
            else:
                plot_settings["problem"] = None
                plot_settings["lower_bounds"] = None
                plot_settings["upper_bounds"] = None

            # Plot this generation
            fig, axes = population_dvar_pairs(population, dvars=dvars, fig=fig, axes=axes, **plot_settings)

        # Add colorbar if label is provided
        if label:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Dummy array for the colorbar
            # Use ax parameter to let constrained_layout handle the positioning
            fig.colorbar(sm, ax=axes.ravel().tolist(), label=label)
    else:
        raise ValueError(f"Unrecognized generation mode: {generation_mode}")

    return fig, axes


def history_obj_animation(
    history: History,
    reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    interval: int = 200,
    domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
    show_points: bool = True,
    n_pf: int = 1000,
    pf_objectives: Optional[np.ndarray] = None,
    show_attainment: bool = False,
    show_dominated_area: bool = False,
    ref_point: Optional[Tuple[float, float]] = None,
    ref_point_padding: float = 0.05,
    legend_loc: Optional[str] = "upper right",
    show_names: bool = True,
    show_pf: bool = False,
    single_color: Optional[str] = None,
    dynamic_scaling: bool = False,
    cumulative: bool = False,
    scale_padding: float = 0.05,
) -> animation.Animation:
    """
    Creates an animated visualization of how the Pareto front evolves across generations.

    Parameters
    ----------
    history : paretobench History
        The history object containing populations with data to plot
    reports : int, slice, List[int], or Tuple[int, int], optional
        Specifies which generations to animate. Can be:
        - None: All generations (default)
        - int: Single generation index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd gen)
        - List[int]: Explicit list of generation indices
        - List[bool] or np.ndarray of bools: boolean mask where True selects the index
        - Tuple[int, int]: Range of generations as (start, end) where end is exclusive
    interval : int, optional
        Delay between frames in milliseconds, by default 200
    domination_filt : Literal["all", "dominated", "non-dominated"], optional
        Plot only the dominated/non-dominated solutions, or all. Defaults to all
    feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    show_points : bool
        Whether to actually show the points (useful for only showing attainment surface or dominated region)
    n_pf : int, optional
        The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
    pf_objectives : array-like, optional
        User-specified Pareto front objectives. Should be a 2D array where each row represents a point
        on the Pareto front and each column represents an objective value.
    show_attainment : bool, optional
        Whether to plot the attainment surface, by default False
    show_dominated_area : bool, optional
        Plots the dominated region towards the larger values of each decision var
    ref_point : Union[str, Tuple[float, float]], optional
        Where to stop plotting the dominated region / attainment surface. Must be a point to the upper right (increasing
        value of objectives in 3D) of all plotted points. By default, will set to right of max of each objective plus
        padding.
    ref_point_padding : float
        Amount of padding to apply to the automatic reference point calculation.
    legend_loc : str, optional
        Passed to `loc` argument of plt.legend
    show_names : bool, optional
        Whether to show the names of the objectives if provided by population
    show_pf : bool, optional
        Whether to plot the Pareto front, by default True
    single_color: Optional[str] = None
        Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
    dynamic_scaling : bool, optional
        If True, axes limits will update based on each frame's data.
        If False, axes limits will be fixed based on all data, by default False
    cumulative : bool, optional
        If True, shows all points seen up to current frame.
        If False, shows only current frame's points, by default False
    scale_padding : float, optional
        Padding used when calculating axis limits w/ dynamic limits

    Returns
    -------
    animation.Animation
        The animation object that can be displayed in notebooks or saved to file
    """
    if not history.reports:
        raise ValueError("No populations in history to animate")

    # Handle different types of selection
    indices = selection_to_indices(reports, len(history.reports))

    if not indices:
        raise ValueError("No reports selected")

    # Get dimensions from first population
    n_objectives = history.reports[indices[0]].m
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
        selected_reports = [history.reports[idx] for idx in indices]
        all_f = np.vstack([pop.f for pop in selected_reports])
        xlim = (np.min(all_f[:, 0]), np.max(all_f[:, 0]))
        ylim = (np.min(all_f[:, 1]), np.max(all_f[:, 1]))
        xlim = (xlim[0] - (xlim[1] - xlim[0]) * scale_padding, xlim[1] + (xlim[1] - xlim[0]) * scale_padding)
        ylim = (ylim[0] - (ylim[1] - ylim[0]) * scale_padding, ylim[1] + (ylim[1] - ylim[0]) * scale_padding)

    # Function to update frame for animation
    def update(frame_idx):
        ax.clear()

        # Get the actual generation index from our selected indices
        gen_idx = indices[frame_idx]

        # Calculate the slice for history_obj_scatter based on cumulative setting
        if cumulative:
            # Find all indices up to current frame_idx
            plot_reports = indices[: frame_idx + 1]
        else:
            plot_reports = [gen_idx]

        # Plot using the history plotting function
        history_obj_scatter(
            history,
            reports=plot_reports,
            fig=fig,
            ax=ax,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            show_points=show_points,
            n_pf=n_pf,
            pf_objectives=pf_objectives,
            show_attainment=show_attainment,
            show_dominated_area=show_dominated_area,
            ref_point=ref_point,
            ref_point_padding=ref_point_padding,
            legend_loc=legend_loc,
            show_names=show_names,
            show_pf=show_pf,
            single_color=single_color,
            generation_mode="cumulative",
        )

        # Add generation counter
        generation = gen_idx + 1
        fevals = history.reports[gen_idx].fevals
        ax.set_title(f"Generation {generation} (Fevals: {fevals})")

        if n_objectives == 2:
            if not dynamic_scaling:
                # Use global limits
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

        return (ax,)

    # Create and return the animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(indices),  # Now use length of selected indices
        interval=interval,
        blit=False,
    )

    return anim


def history_dvar_animation(
    history: History,
    reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    dvars: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
    interval: int = 200,
    domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
    feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
    hist_bins: Optional[int] = None,
    show_names: bool = True,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    single_color: Optional[str] = None,
    plot_bounds: bool = False,
    dynamic_scaling: bool = False,
    cumulative: bool = False,
    scale_padding=0.05,
) -> animation.Animation:
    """
    Creates an animated visualization of how the decision variables evolve across generations.

    Parameters
    ----------
    history : paretobench History
        The history object containing populations with data to plot
    reports : int, slice, List[int], or Tuple[int, int], optional
        Specifies which generations to animate. Can be:
        - None: All generations (default)
        - int: Single generation index (negative counts from end)
        - slice: Range with optional step (e.g., slice(0, 10, 2) for every 2nd gen)
        - List[int]: Explicit list of generation indices
        - Tuple[int, int]: Range of generations as (start, end) where end is exclusive
    dvars : int, slice, List[int], or Tuple[int, int], optional
        Which decision vars to plot. See `population_dvar_pairs` docstring for more details.
    interval : int, optional
        Delay between frames in milliseconds, by default 200
    domination_filt : Literal["all", "dominated", "non-dominated"], optional
        Plot only the dominated/non-dominated solutions, or all. Defaults to all
    feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
        Plot only the feasible/infeasible solutions, or all. Defaults to all
    hist_bins : int, optional
        Number of bins for histograms on the diagonal, default is to let matplotlib choose
    show_names : bool, optional
        Whether to include variable names on the axes if they exist, default is True
    lower_bounds : array-like, optional
        Lower bounds for each decision variable
    upper_bounds : array-like, optional
        Upper bounds for each decision variable
    single_color: Optional[str] = None
        Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
    plot_bounds: bool = False
        Whether to plot bounds for the problem
    dynamic_scaling : bool, optional
        If True, axes limits will update based on each frame's data.
        If False, axes limits will be fixed based on all data, by default False
    cumulative : bool, optional
        If True, shows all points seen up to current frame.
        If False, shows only current frame's points, by default False
    scale_padding : float, optional
        Padding used when calculating axis limits w/ dynamic limits

    Returns
    -------
    animation.Animation
        The animation object that can be displayed in notebooks or saved to file
    """
    if not history.reports:
        raise ValueError("No populations in history to animate")

    # Handle different types of selection
    indices = selection_to_indices(reports, len(history.reports))

    if not indices:
        raise ValueError("No reports selected")

    # Create initial plot to get figure and axes using the first selected generation
    fig, axes = history_dvar_pairs(
        history,
        reports=indices[0],
        dvars=dvars,
        domination_filt=domination_filt,
        feasibility_filt=feasibility_filt,
        hist_bins=hist_bins,
        show_names=show_names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        single_color=single_color,
        plot_bounds=plot_bounds,
        generation_mode="cumulative",
    )

    # Calculate global axis limits if not using dynamic scaling
    if not dynamic_scaling:
        # Only use selected generations for calculating limits
        selected_reports = [history.reports[idx] for idx in indices]
        all_x = np.vstack([pop.x for pop in selected_reports])
        n_vars = all_x.shape[1]

        # Calculate limits for each variable
        var_limits = []
        for i in range(n_vars):
            var_min, var_max = np.min(all_x[:, i]), np.max(all_x[:, i])
            limit = (var_min - (var_max - var_min) * scale_padding, var_max + (var_max - var_min) * scale_padding)
            var_limits.append(limit)

    # Function to update frame for animation
    def update(frame_idx):
        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        # Get the actual generation index from our selected indices
        gen_idx = indices[frame_idx]

        # Calculate the appropriate reports for history_dvar_pairs based on cumulative setting
        if cumulative:
            # Find all indices up to current frame_idx
            plot_reports = indices[: frame_idx + 1]
        else:
            plot_reports = [gen_idx]

        # Plot using the history plotting function
        history_dvar_pairs(
            history,
            reports=plot_reports,
            dvars=dvars,
            fig=fig,
            axes=axes,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            hist_bins=hist_bins,
            show_names=show_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            single_color=single_color,
            plot_bounds=plot_bounds,
            generation_mode="cumulative",
        )

        # Add generation counter
        generation = gen_idx + 1
        fevals = history.reports[gen_idx].fevals
        fig.suptitle(f"Generation {generation} (Fevals: {fevals})")

        if not dynamic_scaling:
            # Apply global limits to all subplots including histograms
            for i, ax in enumerate(axes.flat):
                row = i // axes.shape[1]
                col = i % axes.shape[1]
                if row == col:
                    # For histograms on diagonal
                    ax.set_xlim(*var_limits[row])
                else:
                    # For scatter plots (both upper and lower triangle)
                    ax.set_xlim(*var_limits[col])
                    ax.set_ylim(*var_limits[row])

        return tuple(axes.flat)

    # Create and return the animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(indices),  # Now use length of selected indices
        interval=interval,
        blit=False,
    )

    return anim
