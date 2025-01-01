from dataclasses import dataclass
from matplotlib.colors import to_rgb
from typing import Literal
import numpy as np

from ..containers import Population
from ..utils import fast_dominated_argsort


@dataclass
class PointSettings:
    nd_inds: np.ndarray
    feas_inds: np.ndarray
    markers: np.ndarray
    plot_filt: np.ndarray
    alpha: np.ndarray


def get_per_point_settings_population(
    population: Population,
    plot_dominated: Literal["all", "dominated", "non-dominated"],
    plot_feasible: Literal["all", "feasible", "infeasible"],
):
    """
    Calculate the per-point settings for scatter plots of the population (ie color, marker, which points are visible)
    based on shared settings across plot types.

    Parameters
    ----------
    population : Population
        Population we are plottings.
    plot_dominated : {'all', 'dominated', 'non-dominated'}
        Which points to plot based on domination status.
    plot_feasible : {'all', 'feasible', 'infeasible'}
        Which points to plot based on feasibility status.

    Returns
    -------
    PointSettings
        Settings object containing:
        nd_inds : ndarray of bool
            Non-dominated indices.
        feas_inds : ndarray of bool
            Feasible indices.
        plot_filt : ndarray of bool
            Which points should be plotted.
        alpha : ndarray of float
            Alpha value per point based on domination rank.
        markers : ndarray of str
            Marker type per point ('o' for feasible, 'x' for infeasible).
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

    # Get the domination ranks (of only the visible solutions so we don't end up with a plot of all invisible points)
    ranks = np.zeros(len(population))
    filtered_indices = np.where(plot_filt)[0]
    for rank, idx in enumerate(fast_dominated_argsort(population.f[plot_filt, :], population.g[plot_filt, :])):
        ranks[filtered_indices[idx]] = rank

    # Compute alpha from the ranks
    if np.all(rank < 1):
        alpha = np.ones(len(population))
    else:
        alpha = 0.5 - ranks / ranks.max() * 0.3
        alpha[ranks == 0] = 1.0

    return PointSettings(nd_inds=nd_inds, feas_inds=feas_inds, plot_filt=plot_filt, alpha=alpha, markers=markers)


def alpha_scatter(ax, x, y, z=None, color=None, alpha=None, marker=None, **kwargs):
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
        if z is None:
            points.append(ax.scatter(x[mask], y[mask], c=filtered_color, marker=m, **kwargs))
        else:
            points.append(ax.scatter(x[mask], y[mask], z[mask], c=filtered_color, marker=m, **kwargs))
    return points
