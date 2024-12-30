from copy import copy
from dataclasses import dataclass
from matplotlib import animation
from matplotlib.colors import LightSource
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple, Literal
import matplotlib.pyplot as plt
import numpy as np


from .containers import Population, History
from .problem import ProblemWithFixedPF, ProblemWithPF, get_problem_from_obj_or_str
from .exceptions import EmptyPopulationError, NoDecisionVarsError, NoObjectivesError
from .utils import fast_dominated_argsort, get_nondominated_inds


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


def compute_attainment_surface_2d(points: np.ndarray, ref_point=None, padding=0.1):
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
    if len(points) == 0:
        return np.empty((0, 2))

    # Handle missing ref-point
    if ref_point is None:
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        ref_point = max_vals + (max_vals - min_vals) * padding

    # Get only nondominated points
    points = points[get_nondominated_inds(points), :]

    if (ref_point[0] < points[:, 0]).any() or (ref_point[1] < points[:, 1]).any():
        raise ValueError(
            f"Reference point coordinates must exceed all points in non-dominated set "
            f"(ref_point={ref_point}, max_pf=({np.max(points[:, 0])}, {np.max(points[:, 1])}))"
        )

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
    surface = np.array(surface)
    return np.concatenate(
        (
            [[surface[0, 0], ref_point[1]]],
            surface,
            [[ref_point[0], surface[-1, 1]]],
        ),
        axis=0,
    )


def save_mesh_to_stl(vertices: np.ndarray, triangles: np.ndarray, filename: str):
    """
    Save a triangular mesh to STL file format.

    Args:
        vertices: (n,3) array of vertex coordinates
        triangles: (m,3) array of triangle indices into vertices
        filename: output filename (should end in .stl)
    """
    # Ensure proper file extension
    if not filename.endswith(".stl"):
        filename += ".stl"

    with open(filename, "w") as f:
        f.write("solid attainment_surface\n")

        # For each triangle
        for triangle in triangles:
            # Get vertex coordinates for this triangle
            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]

            # Compute normal using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            # Normalize
            length = np.sqrt(np.sum(normal**2))
            if length > 0:
                normal = normal / length

            # Write facet
            f.write(f" facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("  outer loop\n")
            f.write(f"   vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"   vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"   vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("  endloop\n")
            f.write(" endfacet\n")


def get_vertex_index(point, vertex_dict, vertices):
    """Helper function to get or create vertex index for a point."""
    point_tuple = tuple(point)
    if point_tuple not in vertex_dict:
        vertex_dict[point_tuple] = len(vertices)
        vertices.append(point)
    return vertex_dict[point_tuple]


def find_rectangles(valid_cells, coords1, coords2):
    """
    Find maximal rectangles in a binary grid.
    Returns list of (min_coord1, min_coord2, max_coord1, max_coord2) for each rectangle.
    """
    if not valid_cells.any():
        return []

    rectangles = []
    remaining = valid_cells.copy()

    while remaining.any():
        # Find first remaining true cell
        row_idx, col_idx = np.nonzero(remaining)
        start_row, start_col = row_idx[0], col_idx[0]

        # Try to expand rectangle right and down
        max_col = start_col
        while max_col + 1 < remaining.shape[1] and remaining[start_row, max_col + 1]:
            max_col += 1

        max_row = start_row
        while max_row + 1 < remaining.shape[0]:
            can_expand = True
            for c in range(start_col, max_col + 1):
                if not remaining[max_row + 1, c]:
                    can_expand = False
                    break
            if not can_expand:
                break
            max_row += 1

        # Add rectangle
        rectangles.append((coords1[start_col], coords2[start_row], coords1[max_col + 1], coords2[max_row + 1]))

        # Mark used cells as processed
        remaining[start_row : max_row + 1, start_col : max_col + 1] = False

    return rectangles


def mesh_plane(sorted_points, fixed_dim, dim1, dim2, reference, vertex_dict, vertices):
    """
    Generate mesh for a plane where fixed_dim is the sorting dimension
    and dim1, dim2 are the dimensions to create the grid in.
    """
    triangles_plane = []

    # Process each point by increasing fixed_dim coordinate
    for i, current_point in enumerate(sorted_points):
        # Get points with smaller fixed_dim coordinate
        previous_points = sorted_points[:i]
        current_fixed = current_point[fixed_dim]

        if len(previous_points) == 0:
            # If no previous points, just add one rectangle up to reference point
            vertices_3d = []
            for p in [
                [current_point[dim1], current_point[dim2]],
                [reference[dim1], current_point[dim2]],
                [current_point[dim1], reference[dim2]],
                [reference[dim1], reference[dim2]],
            ]:
                coord = np.zeros(3)
                coord[fixed_dim] = current_fixed
                coord[dim1] = p[0]
                coord[dim2] = p[1]
                vertices_3d.append(get_vertex_index(coord, vertex_dict, vertices))

            triangles_plane.extend(
                [[vertices_3d[0], vertices_3d[1], vertices_3d[3]], [vertices_3d[0], vertices_3d[3], vertices_3d[2]]]
            )
            continue

        # Get nondominated points from previous layers
        previous_2d = np.column_stack((previous_points[:, dim1], previous_points[:, dim2]))
        nd_points = previous_2d[get_nondominated_inds(previous_2d), :]

        # Get unique coordinates including current point and reference
        coords1 = np.unique(np.concatenate([nd_points[:, 0], [current_point[dim1], reference[dim1]]]))
        coords2 = np.unique(np.concatenate([nd_points[:, 1], [current_point[dim2], reference[dim2]]]))

        # Create grid of valid cells
        valid_cells = np.zeros((len(coords2) - 1, len(coords1) - 1), dtype=bool)

        # For each grid cell
        for row in range(valid_cells.shape[0]):
            for col in range(valid_cells.shape[1]):
                cell_min = np.array([coords1[col], coords2[row]])

                # Check if cell is dominated by current point
                if cell_min[0] >= current_point[dim1] and cell_min[1] >= current_point[dim2]:
                    # Check if cell is not dominated by any previous nondominated point
                    is_dominated_by_previous = False
                    for nd_point in nd_points:
                        if cell_min[0] >= nd_point[0] and cell_min[1] >= nd_point[1]:
                            is_dominated_by_previous = True
                            break

                    valid_cells[row, col] = not is_dominated_by_previous

        # Find maximal rectangles in valid cells
        rectangles = find_rectangles(valid_cells, coords1, coords2)

        # Create triangles for each rectangle
        for rect_min1, rect_min2, rect_max1, rect_max2 in rectangles:
            vertices_3d = []
            for p in [[rect_min1, rect_min2], [rect_max1, rect_min2], [rect_min1, rect_max2], [rect_max1, rect_max2]]:
                coord = np.zeros(3)
                coord[fixed_dim] = current_fixed
                coord[dim1] = p[0]
                coord[dim2] = p[1]
                vertices_3d.append(get_vertex_index(coord, vertex_dict, vertices))

            triangles_plane.extend(
                [[vertices_3d[0], vertices_3d[1], vertices_3d[3]], [vertices_3d[0], vertices_3d[3], vertices_3d[2]]]
            )

    return triangles_plane


def compute_attainment_surface_3d(points: np.ndarray, ref_point=None, padding=0.1):
    """
    Generate triangular mesh for union of cuboids.
    Args:
        points: (n,3) array of points, each defining a cuboid corner
        reference: (3,) array defining the other corner of each cuboid
    Returns:
        vertices: (m,3) array of unique vertices in the mesh
        triangles: (k,3) array of indices into vertices defining triangles
    """
    if points.shape[1] != 3:
        raise ValueError("This function only works for 3D points")
    if len(points) == 0:
        return np.empty((0, 3)), np.empty((0,))

    # If no reference point provided, compute one
    if ref_point is None:
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        ref_point = max_vals + (max_vals - min_vals) * padding
    ref_point = np.asarray(ref_point)
    if not np.all(ref_point >= np.max(points, axis=0)):
        raise ValueError("Reference point must dominate all points")

    # Get the nondominated points
    points = points[get_nondominated_inds(points), :]

    vertices = []
    triangles = []
    vertex_dict = {}

    # Sort points in each dimension
    sorted_by_z = points[np.argsort(points[:, 2])]  # For XY plane
    sorted_by_x = points[np.argsort(points[:, 0])]  # For YZ plane
    sorted_by_y = points[np.argsort(points[:, 1])]  # For XZ plane

    # Process XY plane (sorted by Z)
    triangles.extend(mesh_plane(sorted_by_z, 2, 0, 1, ref_point, vertex_dict, vertices))

    # Process YZ plane (sorted by X)
    triangles.extend(mesh_plane(sorted_by_x, 0, 1, 2, ref_point, vertex_dict, vertices))

    # Process XZ plane (sorted by Y)
    triangles.extend(mesh_plane(sorted_by_y, 1, 0, 2, ref_point, vertex_dict, vertices))

    return np.array(vertices), np.array(triangles)


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

    if fig is None:
        fig = plt.figure()

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

        # Plot the data
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

        # Plot attainment surface if requested (using feasible solutions only)
        attainment = compute_attainment_surface_2d(
            population.f[ps.feas_inds, :], ref_point=settings.ref_point, padding=settings.ref_point_padding
        )
        if settings.plot_attainment:
            ax.plot(attainment[:, 0], attainment[:, 1], color=base_color, alpha=0.5, label="Attainment Surface")
            add_legend = True
        if settings.plot_dominated_area:
            plt.fill_between(
                attainment[:, 0],
                attainment[:, 1],
                attainment[0, 1] * np.ones(attainment.shape[0]),
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
        # Get an axis if not supplied
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")

        # Add in Pareto front
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c="k", s=10, label="PF", alpha=0.75)
            add_legend = True

        # Plot the data
        scatter = alpha_scatter(
            ax,
            population.f[ps.plot_filt, 0],
            population.f[ps.plot_filt, 1],
            population.f[ps.plot_filt, 2],
            alpha=ps.alpha[ps.plot_filt],
            marker=ps.markers[ps.plot_filt],
            color=base_color,
            s=15,
            label=settings.label,
        )
        if scatter:
            base_color = scatter[0].get_facecolor()[0]  # Get the color that matplotlib assigned

        if settings.plot_dominated_area:
            raise NotImplementedError("Cannot display dominated volume in 3D :(")

        if settings.plot_attainment:
            vertices, faces = compute_attainment_surface_3d(
                population.f[ps.feas_inds, :], ref_point=settings.ref_point, padding=settings.ref_point_padding
            )
            poly3d = Poly3DCollection(
                [vertices[face] for face in faces],
                shade=True,
                facecolors=base_color,
                edgecolors=base_color,
                lightsource=LightSource(azdeg=174, altdeg=-15),
            )
            ax.add_collection3d(poly3d)

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

    plot_dominated: Literal["all", "dominated", "non-dominated"] = "all"
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

    # Get the point settings for this plot
    ps = get_per_point_settings_population(population, settings.plot_dominated, settings.plot_feasible)

    # Plot on all axes
    base_color = None
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            # Diagonal plots (histograms)
            if i == j:
                # Plot the histogram
                _, _, patches = ax.hist(
                    population.x[ps.plot_filt, i], bins=settings.hist_bins, density=True, alpha=0.7, color=base_color
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
                # Plot the decision vars
                scatter = alpha_scatter(
                    ax,
                    population.x[ps.plot_filt, j],
                    population.x[ps.plot_filt, i],
                    alpha=ps.alpha[ps.plot_filt],
                    color=base_color,
                    s=15,
                    marker=ps.markers[ps.plot_filt],
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
