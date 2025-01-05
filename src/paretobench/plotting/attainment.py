import numpy as np

from ..containers import Population
from ..utils import get_nondominated_inds


def get_reference_point(population: Population, padding=0.1):
    """Calculate reference point for the attainment surface from all points in the population.

    Parameters
    ----------
    population : Population
        Population to compute reference point for.
    padding : float, optional
        Padding factor to extend reference point beyond maximum values.
        Default is 0.1 (10% extension).

    Returns
    -------
    ndarray
        Reference point coordinates calculated as max values plus padding.
    """
    min_vals = np.min(population.f, axis=0)
    max_vals = np.max(population.f, axis=0)
    ref_point = max_vals + (max_vals - min_vals) * padding
    return ref_point


def compute_attainment_surface_2d(population: Population, ref_point=None, padding=0.1):
    """
    Compute the attainment surface for a set of points in 2D. The surface
    consists of horizontal and vertical lines connecting the points, forming a
    staircase-like pattern.

    Parameters
    ----------
    population : Population
        The population of individuals to compute the attainment surface for.
    ref_point : np.ndarray
        2D reference point for extending surface until
    padding : float
        How much padding to apply if we compute ref_point ourselves

    Returns
    -------
    np.ndarray
        Array of points defining the attainment surface.
    """
    if population.m != 2:
        raise ValueError("Attainment surface can only be computed for 2D points")

    # This only works on feasible individuals
    population = population[population.get_feasible_indices()]

    if len(population) == 0:
        return np.empty((0, 2))

    # Handle missing ref-point
    if ref_point is None:
        ref_point = get_reference_point(population, padding=padding)
    ref_point = np.asarray(ref_point)

    # Get only nondominated points
    population = population.get_nondominated_set()

    if not np.all(ref_point >= np.max(population.f, axis=0)):
        raise ValueError(
            f"Reference point coordinates must exceed all points in non-dominated set "
            f"(ref_point={ref_point}, max_pf=({np.max(population.f[:, 0])}, {np.max(population.f[:, 1])}))"
        )

    # Sort points by x coordinate (first objective)
    sorted_points = population.f[np.argsort(population.f[:, 0])]

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

    return vertices, triangles_plane


def compute_attainment_surface_3d(population: Population, ref_point=None, padding=0.1):
    """
    Generate triangular mesh for union of cuboids attainment surface.

    Parameters
    ----------
    pop : Population
        The population of individuals to compute the attainment surface for.
    reference : ndarray, shape (3,)
        Array defining the other corner of each cuboid starting from nondominated individuals in the population.

    Returns
    -------
    vertices : ndarray, shape (m, 3)
        Array of unique vertices in the mesh.
    triangles : ndarray, shape (k, 3)
        Array of indices into vertices defining triangles.
    """
    if population.m != 3:
        raise ValueError("This function only works for 3D points")

    # This only works on feasible individuals
    population = population[population.get_feasible_indices()]

    if len(population) == 0:
        return np.empty((0, 3)), np.empty((0,))

    # If no reference point provided, compute one
    if ref_point is None:
        ref_point = get_reference_point(population, padding=padding)
    ref_point = np.asarray(ref_point)

    if not np.all(ref_point >= np.max(population.f, axis=0)):
        raise ValueError("Reference point must dominate all points")

    # Get the nondominated points
    population = population.get_nondominated_set()

    vertices = []
    triangles = []
    vertex_dict = {}

    # Sort points in each dimension
    sorted_by_z = population.f[np.argsort(population.f[:, 2])]  # For XY plane
    sorted_by_x = population.f[np.argsort(population.f[:, 0])]  # For YZ plane
    sorted_by_y = population.f[np.argsort(population.f[:, 1])]  # For XZ plane

    # Process XY plane (sorted by Z)
    triangles.extend(mesh_plane(sorted_by_z, 2, 0, 1, ref_point, vertex_dict, vertices)[1])

    # Process YZ plane (sorted by X)
    triangles.extend(mesh_plane(sorted_by_x, 0, 1, 2, ref_point, vertex_dict, vertices)[1])

    # Process XZ plane (sorted by Y)
    triangles.extend(mesh_plane(sorted_by_y, 1, 0, 2, ref_point, vertex_dict, vertices)[1])

    return np.array(vertices), np.array(triangles)
