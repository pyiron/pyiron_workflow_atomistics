from ase import Atoms

import numpy as np
from ase import Atoms
from copy import deepcopy
from ase.build import add_vacuum  # or your custom add_vacuum implementation
def find_viable_cleavage_planes_around_plane(
    atoms: Atoms,
    axis: int,
    plane_coord: float,
    coord_tol: float,
    layer_tolerance: float = 1e-3,
    fractional: bool = False
) -> list:
    """
    Identify viable cleavage‐plane positions along a given axis around a specified plane
    coordinate, rather than a site. This function scans all atomic coordinates (either Cartesian
    or fractional) to find “layer” positions, computes midpoints between adjacent layers, and
    then filters those midpoints that lie within ±coord_tol of the provided plane_coord (within
    a small tolerance for merging nearly‐identical layer positions).

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to analyze.
    axis : int
        Which axis to cleave along: 0 = x, 1 = y, 2 = z.
    plane_coord : float
        The target plane coordinate along `axis`. If `fractional=False`, this is in Å; if
        `fractional=True`, this is a fractional coordinate (0 ≤ plane_coord < 1).
    coord_tol : float
        The half‐width tolerance around `plane_coord`. Units match `plane_coord` (Å if
        `fractional=False`, fractional units if `fractional=True`).
    tolerance : float, optional (default=1e-3)
        Tolerance for merging nearly‐identical layer positions. If two coordinates differ
        by less than `tolerance` (or `tolerance / cell_length` in fractional mode), they
        are considered the same layer.
    fractional : bool, optional (default=False)
        If True, use fractional (scaled) coordinates along `axis` to identify layers.
        If False, use Cartesian coordinates (in Å).

    Returns
    -------
    viable_planes : list of float
        A list of viable cleavage‐plane coordinates (in the same units as `plane_coord`):
        each coordinate is a midpoint between two adjacent “layers” that falls within
        ±`coord_tol` of `plane_coord`.

    Prints
    ------
    - The provided plane coordinate and tolerance.
    - The list of unique layer positions.
    - The list of all candidate cleavage‐plane midpoints.
    """

    import numpy as np

    # 1) Get cell lengths along each axis
    cell = atoms.get_cell()
    cell_lengths = np.linalg.norm(cell, axis=1)  # length of each cell vector

    # 2) Gather coordinates along the axis (Cartesian or fractional)
    if fractional:
        all_coords = atoms.get_scaled_positions(wrap=False)[:, axis] % 1.0
        # Tolerance for merging layers in fraction space
        tol = layer_tolerance / cell_lengths[axis]
        # Interpret plane_coord and coord_tol in fraction‐space
        target = plane_coord % 1.0
        tol_plane = coord_tol / cell_lengths[axis]
        print(f"Target fractional plane: {target} ± {tol_plane}")
    else:
        all_coords = atoms.get_positions(wrap=False)[:, axis]
        tol = layer_tolerance
        target = plane_coord
        tol_plane = coord_tol
        print(f"Target Cartesian plane: {target} Å ± {tol_plane} Å")

    # 3) Identify unique “layers” by merging coords within `tol`
    sorted_coords = np.sort(all_coords)
    unique_layers = []
    for v in sorted_coords:
        if not any(abs(v - u) < tol for u in unique_layers):
            unique_layers.append(v)
    print(f"Unique layer positions: {unique_layers}")

    # 4) Compute midpoints between each pair of adjacent layers
    candidate_planes = [
        (unique_layers[i] + unique_layers[i + 1]) / 2.0
        for i in range(len(unique_layers) - 1)
    ]
    print(f"Candidate cleavage midpoints: {candidate_planes}")

    # 5) Filter midpoints that lie within ±tol_plane of `target`
    if fractional:
        # Wrap‐aware difference in fractional space:
        def frac_dist(a, b):
            return abs(((a - b + 0.5) % 1.0) - 0.5)

        viable_planes = [
            cp for cp in candidate_planes
            if frac_dist(cp, target) <= tol_plane
        ]
    else:
        viable_planes = [
            cp for cp in candidate_planes
            if (target - tol_plane) <= cp <= (target + tol_plane)
        ]

    return viable_planes

def find_viable_cleavage_planes_around_site(atoms: Atoms,
                                axis: int,
                                site_index: int,
                                site_dist_threshold: float,
                                layer_tolerance: float = 1e-3,
                                fractional: bool = False) -> list:
    """
    Identify viable cleavage‐plane positions along a given axis around a specified site.

    Scans all atomic coordinates (either Cartesian or fractional)
    to find “layer” positions. It computes midpoints between adjacent layers,
    and filters those midpoints that lie within ±site_dist_threshold of the
    chosen site’s coordinate (within a small tolerance).

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to analyze.
    axis : int
        Which Cartesian axis to cleave along: 0 = x, 1 = y, 2 = z.
    site_index : int
        The index of the atom whose layer defines the neighborhood of interest.
    site_dist_threshold : float
        Maximum allowed distance (in Å if `fractional=False`, or in fraction of cell length
        if `fractional=True`) between the site’s coordinate and a candidate cleavage plane.
    tolerance : float, optional (default=1e-3)
        Tolerance for merging nearly‐identical layer positions. If two coordinates
        differ by less than `tolerance`, they are considered the same layer.
        When `fractional=True`, this tolerance is divided by the cell length along `axis`.
    fractional : bool, optional (default=False)
        If True, use fractional (scaled) coordinates along `axis` to identify layers.
        If False, use Cartesian coordinates (in Å).
    add_vacuum_block_length : float or None, optional (default=None)
        If not None, first make a copy of `atoms`, insert a vacuum slab of this thickness
        (in Å) along `axis`, and then perform the layer analysis on the vacuum‐padded structure.
        Set to None to disable adding vacuum.

    Returns
    -------
    cp_viable : list of float
        A list of viable cleavage‐plane coordinates (in the same units used for `coords`):
        each coordinate is a midpoint between two adjacent “layers” that falls within
        ±`site_dist_threshold` of the site’s own coordinate.

    Prints
    ------
    - If vacuum is added, prints the vacuum thickness added.
    - The site’s coordinate along `axis` (fractional or Å).
    - The list of unique layer positions.
    - The list of all candidate cleavage‐plane midpoints.
    """

    # 2) Fetch cell and determine length along the chosen axis
    cell = atoms.get_cell()
    cell_lengths = cell.diagonal()  # length along x, y, z

    # 3) Gather coordinates along the axis (Cartesian or fractional)
    if fractional:
        all_coords = atoms.get_scaled_positions(wrap=False)[:, axis]
        site_coord = all_coords[site_index]
        tol = layer_tolerance / cell_lengths[axis]
        threshold_frac = site_dist_threshold / cell_lengths[axis]
        min_lim = site_coord - threshold_frac
        max_lim = site_coord + threshold_frac
    else:
        all_coords = atoms.get_positions(wrap=False)[:, axis]
        site_coord = all_coords[site_index]
        tol = layer_tolerance
        min_lim = site_coord - site_dist_threshold
        max_lim = site_coord + site_dist_threshold
    print(min_lim, max_lim)
    print(f"Site coordinate along axis {axis}: {site_coord} ({'fractional' if fractional else 'Å'})")

    # 4) Identify unique “layers” by merging coordinates within `tol`
    sorted_coords = np.sort(all_coords)
    unique_layers = []
    for v in sorted_coords:
        if not any(abs(v - u) < tol for u in unique_layers):
            unique_layers.append(v)
    print(f"Unique layer positions: {unique_layers}")

    # 5) Compute midpoints between each pair of adjacent layers
    candidate_planes = [
        (unique_layers[i] + unique_layers[i + 1]) / 2.0
        for i in range(len(unique_layers) - 1)
    ]
    print(f"Candidate cleavage positions: {candidate_planes}")

    # 6) Filter midpoints to those within ± site_dist_threshold of the site
    viable_planes = [
        cp for cp in candidate_planes if (min_lim <= cp <= max_lim)
    ]
    print(f"Viable cleavage positions: {viable_planes}")

    return viable_planes

def get_cleavage_planes_around_site(atoms: Atoms,
                                axis: int,
                                site_index: int,
                                site_dist_threshold: float,
                                tolerance: float = 1e-3,
                                fractional: bool = False) -> list:
    """
    Identify viable cleavage‐plane positions along a given axis around a specified site.

    This function optionally adds a vacuum block along the cleavage axis,
    then scans all atomic coordinates (either Cartesian or fractional)
    to find “layer” positions. It computes midpoints between adjacent layers,
    and filters those midpoints that lie within ±site_dist_threshold of the
    chosen site’s coordinate (within a small tolerance).

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to analyze.
    axis : int
        Which Cartesian axis to cleave along: 0 = x, 1 = y, 2 = z.
    site_index : int
        The index of the atom whose layer defines the neighborhood of interest.
    site_dist_threshold : float
        Maximum allowed distance (in Å if `fractional=False`, or in fraction of cell length
        if `fractional=True`) between the site’s coordinate and a candidate cleavage plane.
    tolerance : float, optional (default=1e-3)
        Tolerance for merging nearly‐identical layer positions. If two coordinates
        differ by less than `tolerance`, they are considered the same layer.
        When `fractional=True`, this tolerance is divided by the cell length along `axis`.
    fractional : bool, optional (default=False)
        If True, use fractional (scaled) coordinates along `axis` to identify layers.
        If False, use Cartesian coordinates (in Å).
    add_vacuum_block_length : float or None, optional (default=None)
        If not None, first make a copy of `atoms`, insert a vacuum slab of this thickness
        (in Å) along `axis`, and then perform the layer analysis on the vacuum‐padded structure.
        Set to None to disable adding vacuum.

    Returns
    -------
    cp_viable : list of float
        A list of viable cleavage‐plane coordinates (in the same units used for `coords`):
        each coordinate is a midpoint between two adjacent “layers” that falls within
        ±`site_dist_threshold` of the site’s own coordinate.

    Prints
    ------
    - If vacuum is added, prints the vacuum thickness added.
    - The site’s coordinate along `axis` (fractional or Å).
    - The list of unique layer positions.
    - The list of all candidate cleavage‐plane midpoints.
    """

    # 2) Fetch cell and determine length along the chosen axis
    cell = atoms.get_cell()
    cell_lengths = cell.diagonal()  # length along x, y, z

    # 3) Gather coordinates along the axis (Cartesian or fractional)
    if fractional:
        all_coords = atoms.get_scaled_positions(wrap=False)[:, axis]
        site_coord = all_coords[site_index]
        tol = tolerance / cell_lengths[axis]
        threshold_frac = site_dist_threshold / cell_lengths[axis]
        min_lim = site_coord - threshold_frac
        max_lim = site_coord + threshold_frac
    else:
        all_coords = atoms.get_positions(wrap=False)[:, axis]
        site_coord = all_coords[site_index]
        tol = tolerance
        min_lim = site_coord - site_dist_threshold
        max_lim = site_coord + site_dist_threshold

    print(f"Site coordinate along axis {axis}: {site_coord} ({'fractional' if fractional else 'Å'})")

    # 4) Identify unique “layers” by merging coordinates within `tol`
    sorted_coords = np.sort(all_coords)
    unique_layers = []
    for v in sorted_coords:
        if not any(abs(v - u) < tol for u in unique_layers):
            unique_layers.append(v)
    print(f"Unique layer positions: {unique_layers}")

    # 5) Compute midpoints between each pair of adjacent layers
    candidate_planes = [
        (unique_layers[i] + unique_layers[i + 1]) / 2.0
        for i in range(len(unique_layers) - 1)
    ]
    print(f"Candidate cleavage positions: {candidate_planes}")

    # 6) Filter midpoints to those within ± site_dist_threshold of the site
    viable_planes = [
        cp for cp in candidate_planes if (min_lim <= cp <= max_lim)
    ]

    return viable_planes

def cleave_axis_aligned(
    atoms: Atoms,
    axis: str | int,
    plane_coord: float,
    separation: float,
    use_fractional: bool = False
) -> Atoms:
    """
    Cleave an ASE Atoms object by an axis-aligned plane and move the two halves apart.
    You can specify the plane either in Cartesian (Å) or fractional (0–1) coordinates.

    Parameters
    ----------
    atoms : ase.Atoms
        The original system to be cleaved.
    axis : {'x','y','z'} or {0,1,2}
        The Cartesian axis along which to cleave:
          - 'x' or 0 → plane normal = [1, 0, 0]
          - 'y' or 1 → plane normal = [0, 1, 0]
          - 'z' or 2 → plane normal = [0, 0, 1]
    plane_coord : float
        The coordinate along the chosen axis where the cleavage plane lies.
        If use_fractional=False, this is a Cartesian coordinate in Å (e.g. z = 3.2 Å).
        If use_fractional=True, this is a fractional coordinate (0 ≤ plane_coord < 1).
    separation : float
        The total distance (in Å) by which the two halves should be separated
        along the chosen axis. Atoms on the “+” side move by +separation/2,
        and atoms on the “–” side move by –separation/2 along that axis.
    use_fractional : bool, optional (default=False)
        If False, compare each atom’s Cartesian coordinate to plane_coord (in Å).
        If True, compare each atom’s fractional (scaled) coordinate to plane_coord
        (in 0–1) to decide which side of the plane it lies on.

    Returns
    -------
    new_atoms : ase.Atoms
        A deep-copied Atoms object where:
          - any atom with coord(axis) ≥ plane_coord (cart or frac) has been shifted
            by +separation/2 along that axis (in Å),
          - any atom with coord(axis) <  plane_coord (cart or frac) has been shifted
            by –separation/2 along that axis (in Å).

    Notes
    -----
    1. Comparing in fractional mode does NOT modify cell vectors: it only
       uses scaled positions to classify atoms. The actual displacement is
       always in Cartesian (Å) by ±separation/2.
    2. Atoms exactly at coord == plane_coord (within floating-point tolerance)
       are treated as “≥” → on the positive side. If you want them on the negative
       side, change `>=` to `>` in the mask.
    3. This function does NOT modify the cell geometry or PBC flags. If you
       need periodicity, you must enlarge the cell or turn off PBC manually.
    """

    # 1) Copy so original Atoms isn’t modified
    new_atoms = deepcopy(atoms)

    # 2) Determine integer axis index (0=x,1=y,2=z)
    if isinstance(axis, str):
        ax_char = axis.lower()
        if ax_char == 'x':
            ax = 0
        elif ax_char == 'y':
            ax = 1
        elif ax_char == 'z':
            ax = 2
        else:
            raise ValueError("If `axis` is a string, it must be 'x', 'y', or 'z'.")
    elif isinstance(axis, (int, np.integer)):
        if axis in (0, 1, 2):
            ax = int(axis)
        else:
            raise ValueError("If `axis` is an integer, it must be 0 (x), 1 (y), or 2 (z).")
    else:
        raise ValueError("`axis` must be either 'x','y','z' or the integer 0,1,2.")

    # 3) Fetch positions or fractional coords along that axis
    if use_fractional:
        coords = new_atoms.get_scaled_positions(wrap=False)[:, ax] % 1.0
        target = plane_coord % 1.0
        # Build masks: positive side = coord >= target (wrap‐aware)
        # We treat standard comparison (no wrap here), since 0 ≤ target < 1 and coords in [0,1)
        mask_positive = coords >= target
        mask_negative = coords < target
    else:
        coords = new_atoms.get_positions(wrap=False)[:, ax]
        target = plane_coord
        mask_positive = coords >= target
        mask_negative = coords < target

    # 4) Determine shift distances (in Å)
    delta_pos =  0.5 * separation
    delta_neg = -0.5 * separation

    # 5) Build displacement array (N×3) all zeros except the chosen axis
    positions = new_atoms.get_positions(wrap=False)
    displacements = np.zeros_like(positions)
    displacements[mask_positive, ax] = delta_pos
    displacements[mask_negative, ax] = delta_neg

    # 6) Apply translations
    new_positions = positions + displacements
    new_atoms.set_positions(new_positions)

    return new_atoms

import matplotlib.pyplot as plt
import numpy as np

def plot_structure_with_cleavage(
    atoms,
    cleavage_planes,
    projection=(0, 2),
    reps=(1, 1),
    figsize=(8, 6),
    atom_color='C0',
    plane_color='r',
    plane_linestyle='--',
    atom_size=30,
    save_path=None,
    dpi=300,
    show_fractional_axes: bool = True
):
    """
    Plot a 2D projection of `atoms` with cleavage planes overlaid as lines,
    and optionally add secondary axes showing fractional coordinates.

    Parameters
    ----------
    atoms : ASE Atoms
        The atomic structure to visualize.
    cleavage_planes : list of float
        List of cleavage plane coordinates along the axis specified in `projection[1]`.
        For projection=(0,2), these are z-coordinates in Å.
    projection : tuple(int, int), optional
        Two axes to project onto, e.g. (0,2) for x-z projection.
    reps : tuple(int, int), optional
        Number of periodic repeats along the projection axes (for tiling).
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    atom_color : color, optional
        Color for the atom scatter points.
    plane_color : color, optional
        Color for the cleavage plane lines.
    plane_linestyle : str, optional
        Line style for the cleavage plane lines (e.g. '--', '-.', etc.).
    atom_size : int, optional
        Marker size for atoms.
    save_path : str or None, optional
        If provided, path to save the figure (PNG, etc.).
    dpi : int, optional
        Resolution in dots per inch if the figure is saved.
    show_fractional_axes : bool, optional
        If True, add secondary x- and y-axes showing fractional coordinates.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    # Unpack projection
    p0, p1 = projection
    cell = atoms.get_cell()
    # Compute tiling shifts (in Cartesian)
    shifts = [i * cell[p0] + j * cell[p1] for i in range(reps[0]) for j in range(reps[1])]

    # Extract atomic positions
    pos = atoms.get_positions()
    xs = pos[:, p0]
    ys = pos[:, p1]

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Plot atoms for each tile
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]
        ax.scatter(
            xs + sx, ys + sy,
            s=atom_size, color=atom_color,
            label='Atoms' if shift is shifts[0] else None
        )

    # Determine overall x-range for the horizontal lines
    all_x_positions = np.concatenate([xs + shift[p0] for shift in shifts])
    x_min, x_max = all_x_positions.min(), all_x_positions.max()

    # 2) Plot cleavage planes: for each plane coordinate, draw a horizontal line at that y (p1)
    for plane in cleavage_planes:
        for shift in shifts:
            line_y = plane + shift[p1]
            ax.hlines(
                y=line_y,
                xmin=x_min, xmax=x_max,
                colors=plane_color,
                linestyles=plane_linestyle,
                label='Cleavage plane' if (plane == cleavage_planes[0] and shift is shifts[0]) else None
            )

    # 3) Labels and aesthetics
    ax.set_xlabel(f'Axis {p0} (Å)')
    ax.set_ylabel(f'Axis {p1} (Å)')
    ax.set_title(f'2D Projection with Cleavage Planes (proj {p0}-{p1})')
    ax.set_aspect('equal')
    ax.autoscale()

    # 4) Add secondary fractional axes if requested
    if show_fractional_axes:
        # Secondary Y-axis (right): fractional along p1
        cell_len_p1 = np.linalg.norm(cell[p1])
        def to_frac_y(y): return y / cell_len_p1
        def to_cart_y(f): return f * cell_len_p1
        secax_y = ax.secondary_yaxis('right', functions=(to_frac_y, to_cart_y))
        secax_y.set_ylabel(f'Axis {p1} (fractional)')

        # Secondary X-axis (top): fractional along p0
        cell_len_p0 = np.linalg.norm(cell[p0])
        def to_frac_x(x): return x / cell_len_p0
        def to_cart_x(f): return f * cell_len_p0
        secax_x = ax.secondary_xaxis('top', functions=(to_frac_x, to_cart_x))
        secax_x.set_xlabel(f'Axis {p0} (fractional)')

        # Optionally set ticks on the fractional axes (uncomment to use):
        # frac_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        # secax_y.set_yticks(frac_ticks)
        # secax_y.set_ylim(to_frac_y(ax.get_ylim()[0]), to_frac_y(ax.get_ylim()[1]))
        # secax_x.set_xticks(frac_ticks)
        # secax_x.set_xlim(to_frac_x(ax.get_xlim()[0]), to_frac_x(ax.get_xlim()[1]))

    # 5) Legend outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    # 6) Adjust layout to accommodate legend and save if requested
    if save_path:
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # leave room on right
        fig.savefig(save_path, dpi=dpi)
    else:
        plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.show()
    return fig, ax