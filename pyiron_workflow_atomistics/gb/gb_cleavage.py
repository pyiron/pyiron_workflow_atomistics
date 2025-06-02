import numpy as np
from ase import Atoms
import matplotlib.pyplot as plt
import pyiron_workflow as pwf
from .analysis import get_sites_on_plane
# Wrap‐aware difference in fractional space:
def frac_dist(a, b):
    return abs(((a - b + 0.5) % 1.0) - 0.5)
@pwf.as_function_node
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
    layer_tolerance : float, optional (default=1e-3)
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

@pwf.as_function_node
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
    layer_tolerance : float, optional (default=1e-3)
        Tolerance for merging nearly‐identical layer positions. If two coordinates
        differ by less than `tolerance`, they are considered the same layer.
        When `fractional=True`, this tolerance is divided by the cell length along `axis`.
    fractional : bool, optional (default=False)
        If True, use fractional (scaled) coordinates along `axis` to identify layers.
        If False, use Cartesian coordinates (in Å).

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

@pwf.as_function_node
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

@pwf.as_function_node
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
    new_atoms = atoms.copy()

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

@pwf.as_function_node
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
    show_fractional_axes: bool = True,
    ylims=None
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
    ylims : tuple(float, float) or None, optional
        If provided, sets the y-axis limits on the primary axis as (ymin, ymax).

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    import numpy as np
    import matplotlib.pyplot as plt

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

    # 4) Apply user-specified y-limits if provided
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.autoscale()

    # 5) Add secondary fractional axes if requested
    if show_fractional_axes:
        # Secondary Y-axis (right): fractional along p1
        cell_len_p1 = np.linalg.norm(cell[p1])
        secax_y = ax.secondary_yaxis(
            'right',
            functions=(lambda y: y / cell_len_p1,    # cart → frac
                       lambda f: f * cell_len_p1)    # frac → cart
        )
        secax_y.set_ylabel(f'Axis {p1} (fractional)')

        # Secondary X-axis (top): fractional along p0
        cell_len_p0 = np.linalg.norm(cell[p0])
        secax_x = ax.secondary_xaxis(
            'top',
            functions=(lambda x: x / cell_len_p0,     # cart → frac
                       lambda f: f * cell_len_p0)      # frac → cart
        )
        secax_x.set_xlabel(f'Axis {p0} (fractional)')

        # If you want to align fractional tick ranges with the primary axis limits,
        # you can uncomment and use the following lines:
        #
        # # For Y-axis:
        # if ylims is not None:
        #     frac_ymin = ylims[0] / cell_len_p1
        #     frac_ymax = ylims[1] / cell_len_p1
        #     secax_y.set_ylim(frac_ymin, frac_ymax)
        #
        # # For X-axis:
        # xlims = ax.get_xlim()
        # frac_xmin = xlims[0] / cell_len_p0
        # frac_xmax = xlims[1] / cell_len_p0
        # secax_x.set_xlim(frac_xmin, frac_xmax)

    # 6) Legend outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    # 7) Adjust layout to accommodate legend and save if requested
    # if save_path:
    #     plt.tight_layout(rect=[0, 0, 0.75, 1])  # leave room on right
    #     fig.savefig(save_path, dpi=dpi)
    # else:
    #     plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.show()
    return fig, ax


@pwf.as_function_node
def cleave_gb_structure(
    base_atoms,
    axis_to_cleave="z",
    target_coord=None,
    tol=0.3,
    cleave_region_halflength=5.0,
    layer_tolerance=0.3,
    separation=8.0,
    use_fractional=False
):
    """
    Find and cleave a grain‐boundary‐type slab (base_atoms) into multiple
    “cleaved” pieces along all viable GB planes found near a given target_coord.

    Parameters
    ----------
    base_atoms : ase.Atoms
        The input slab/Supercell (e.g. Fe+GB+vacuum) that you want to cleave.
    axis_to_cleave : {"x", "y", "z", 0, 1, 2}, default "z"
        The crystallographic direction along which to cleave (e.g. "z" or 2).
        This is passed directly to cleave_axis_aligned.
    target_coord : array‐like of length 3, required
        The Cartesian (x,y,z) coordinate of the GB plane (e.g. from your
        gb_plane_extractor.outputs.gb_plane_analysis_dict["gb_cart"]).
        Used to locate which atomic “site” sits on/near the GB plane. 
    tol : float, default 0.3
        Tolerance (in Å or fractional units, depending on use_fractional)
        for selecting atoms on that plane when calling get_sites_on_plane.
    cleave_region_halflength : float, default 5.0
        In Å, how far away from the “best” mid‐plane site to search for
        additional nearby planes. Passed to find_viable_cleavage_planes_around_site.
    layer_tolerance : float, default 0.3
        Tolerance (in Å) for how “stacked” atomic layers can be when picking
        cleavage planes. Passed to find_viable_cleavage_planes_around_site.
    separation : float, default 8.0
        Final gap (in Å) between the two half‐slabs after cleaving. Passed to cleave_axis_aligned.
    use_fractional : bool, default False
        Whether target_coord and tol are in fractional (True) or Cartesian (False)
        units when calling get_sites_on_plane.

    Returns
    -------
    cleaved_structures : list of ase.Atoms
        A list containing one `Atoms` object per viable cleavage plane.
        Each entry is the full “cleaved” supercell with the specified separation.

    Example
    -------
    >>> from ase.build import bulk
    >>> from pyiron_workflow_atomistics.gb.analysis import get_sites_on_plane
    >>> # … (suppose you have a workflow `wf` that built `wf.gb_with_vacuum.outputs.new_atoms.value`) …
    >>> base = wf.gb_with_vacuum.outputs.new_atoms.value
    >>> gb_cart = wf.gb_plane_extractor.outputs.gb_plane_analysis_dict.value["gb_cart"]
    >>> cleaved_list = cleave_gb_structure(
    ...     base_atoms=base,
    ...     axis_to_cleave="z",
    ...     target_coord=gb_cart,
    ...     tol=0.3,
    ...     site_dist_threshold=5.0,
    ...     layer_tolerance=0.3,
    ...     separation=8.0,
    ...     use_fractional=False
    ... )
    >>> len(cleaved_list)
    3  # say there were 3 viable planes found
    >>> # You can then inspect or write any of them:
    >>> cleaved_list[0].write("gb_cleaved_0.vasp", format="vasp")
    """
    # 1) Map axis_to_cleave → the “axis letter” that get_sites_on_plane expects.
    #
    #    - If user passed integer (0/1/2), convert to 'a'/'b'/'c'.
    #    - If user passed string 'x','y','z' (case‐insensitive), map to 'a','b','c'.
    #    - Otherwise, assume they passed already 'a','b','c'.
    axis_map_int_to_letter = {0: "a", 1: "b", 2: "c"}
    axis_map_xyz_to_abc = {"x": "a", "y": "b", "z": "c"}
    if isinstance(axis_to_cleave, int):
        plane_axis_letter = axis_map_int_to_letter[axis_to_cleave]
        cleave_axis_arg = axis_to_cleave
    else:
        # string case
        a2c = axis_to_cleave.lower()
        if a2c in axis_map_xyz_to_abc:
            plane_axis_letter = axis_map_xyz_to_abc[a2c]
            cleave_axis_arg = ["x", "y", "z"].index(a2c)
        elif a2c in ("a", "b", "c"):
            plane_axis_letter = a2c
            cleave_axis_arg = {"a": 0, "b": 1, "c": 2}[a2c]
        else:
            raise ValueError(
                f"axis_to_cleave='{axis_to_cleave}' not recognized. Must be 0/1/2 or "
                f"'x','y','z','a','b','c'."
            )

    # 2) Identify which atom index sits “on/near” the GB plane.
    #    We call get_sites_on_plane with the chosen axis‐letter, target_coord, and tol.
    #    That returns an array of site indices; pick the first (or only) one.
    mid_site_indices = get_sites_on_plane.node_function(
        atoms=base_atoms,
        axis=plane_axis_letter,
        target_coord=target_coord,
        tol=tol,
        use_fractional=use_fractional,
    )
    if len(mid_site_indices) == 0:
        raise RuntimeError(
            f"No atoms found within tol={tol} Å (or fractional) of {plane_axis_letter}={target_coord}."
        )
    mid_site_idx = int(mid_site_indices[0])

    # 3) Find all viable cleavage planes around that site index.
    #    We pass the integer‐axis (0/1/2), the site index, plus thresholds.
    cleavage_plane_coords = find_viable_cleavage_planes_around_site.node_function(
        atoms=base_atoms,
        axis=cleave_axis_arg,
        site_index=mid_site_idx,
        site_dist_threshold=cleave_region_halflength,
        layer_tolerance=layer_tolerance,
        fractional=use_fractional,
    )

    # 4) For each plane coord, call cleave_axis_aligned to get a “cleaved” slab.
    cleaved_structures = []
    for plane_c in cleavage_plane_coords:
        slab_pair = cleave_axis_aligned.node_function(
            atoms=base_atoms,
            axis=cleave_axis_arg,
            plane_coord=plane_c,
            separation=separation,
            use_fractional=use_fractional
        )
        cleaved_structures.append(slab_pair)

    return cleaved_structures, cleavage_plane_coords

from pyiron_workflow_atomistics.featurisers import voronoiSiteFeaturiser
from pyiron_workflow_atomistics.gb.analysis import find_GB_plane, plot_GB_plane
from pyiron_workflow_atomistics.calculator import fillin_default_calckwargs, calculate_structure_node
import os
from pyiron_workflow.api import for_node
@pwf.as_macro_node(
    "gb_plane_extractor",
    "gb_plane_plot",
    "cleave_setup",
    "cleavage_foldernames",
    "cleavage_structure_plot",
    "calculate_cleaved",
    "calculate_rigid"
)
def gb_cleavage_macro(
    wf,
    structure,
    calc,
    # Parent directory for outputs
    parent_dir: str = "gb_cleavage",
    #
    # --- find_GB_plane flags ---
    featuriser=voronoiSiteFeaturiser,
    gb_axis: str = "c",
    approx_frac: float = 0.5,
    tolerance: float = 5.0,
    bulk_offset: float = 10.0,
    slab_thickness: float = 2.0,
    featuriser_kwargs: dict = {},
    n_bulk: int = 10,
    threshold_frac: float = 0.1,
    #
    # --- plot_GB_plane flags ---
    plot_projection: tuple = (0, 2),
    plot_reps: tuple = (5, 1),
    plot_figsize: tuple = (10, 6),
    bulk_color: str = "C0",
    window_cmap: str = "viridis",
    plane_linestyles: tuple = ("--", "-"),
    plot_axis: int = 2,
    plot_dpi: int = 300,
    gb_plane_save_path: str = None,
    #
    # --- cleave_gb_structure flags ---
    axis_to_cleave: str = "z",
    cleave_tol: float = 0.3,
    cleave_region_halflength: float = 5.0,
    layer_tolerance: float = 0.3,
    separation: float = 8.0,
    use_fractional: bool = False,
    #
    # --- plot_structure_with_cleavage flags ---
    struct_projection: tuple = (0, 2),
    struct_reps: tuple = (5, 1),
    struct_figsize: tuple = (8, 6),
    atom_color: str = "C0",
    plane_color: str = "r",
    plane_linestyle: str = "--",
    atom_size: float = 30,
    struct_save_path: str = None,
    struct_dpi: int = 300,
    show_fractional_axes: bool = True,
    ylims: list = [0, 61],
    #
    # --- calculation kwargs ---
    calc_kwargs: dict = None,
    calc_kwargs_defaults: dict = {
        "output_dir":           "gb_cleavage/calculations",
        "fmax":                 0.01,
        "max_steps":            1000,
        "properties":           ("energy", "forces", "stresses", "volume"),
        "write_to_disk":        False,
        "initial_struct_path":  "initial_structure.xyz",
        "initial_results_path": "initial_results.json",
        "traj_struct_path":     "trajectory.xyz",
        "traj_results_path":    "trajectory_results.json",
        "final_struct_path":    "final_structure.xyz",
        "final_results_path":   "final_results.json",
    },
    static_max_steps: int = 0
):
    """
    Macro node that:
    1. Finds the grain‐boundary (GB) plane in the relaxed structure.
    2. Plots the GB‐plane identification.
    3. Generates and plots the cleaved structures along that plane.
    4. Sets up and runs “full” and “rigid” calculations on each cleaved slab.

    Parameters:
    -----------
    wf : Workflow
        The workflow to which nodes are added.
    relax_original_structure : Node
        Node whose output `.outputs.atoms` is the relaxed GB structure.
    calc : Calc
        Calculation setup/node to apply to each cleaved structure.

    parent_dir : str
        Base folder for storing plots & calculation subfolders.

    --- find_GB_plane flags ---
    featuriser : Featuriser class
    gb_axis : {"a","b","c"}
    approx_frac : float
    tolerance : float
    bulk_offset : float
    slab_thickness : float
    featuriser_kwargs : dict
    n_bulk : int
    threshold_frac : float

    --- plot_GB_plane flags ---
    plot_projection : tuple[int, int]
    plot_reps : tuple[int, int]
    plot_figsize : tuple[float, float]
    bulk_color : str
    window_cmap : str
    plane_linestyles : tuple[str, str]
    plot_axis : int
    plot_dpi : int
    gb_plane_save_path : str or None

    --- cleave_gb_structure flags ---
    axis_to_cleave : {"a","b","c"}
    cleave_tol : float
    cleave_region_halflength : float
    layer_tolerance : float
    separation : float
    use_fractional : bool

    --- plot_structure_with_cleavage flags ---
    struct_projection : tuple[int, int]
    struct_reps : tuple[int, int]
    struct_figsize : tuple[float, float]
    atom_color : str
    plane_color : str
    plane_linestyle : str
    atom_size : float
    struct_save_path : str or None
    struct_dpi : int
    show_fractional_axes : bool
    ylims : list[float]

    --- calculation kwargs ---
    calc_kwargs : dict
        Overrides for calculation parameters. Merged with calc_kwargs_defaults.
    calc_kwargs_defaults : dict
        Default calculation parameters for “full” runs.
    static_max_steps : int
        Number of steps for a “rigid/static” run (defaults to 0).
    """
    # 1. Ensure the parent directory exists
    os.makedirs(parent_dir, exist_ok=True)

    # 2. Find the GB plane
    wf.gb_plane_extractor = find_GB_plane(
        atoms=structure,
        featuriser=featuriser,
        axis=gb_axis,
        approx_frac=approx_frac,
        tolerance=tolerance,
        bulk_offset=bulk_offset,
        slab_thickness=slab_thickness,
        featuriser_kwargs=featuriser_kwargs,
        n_bulk=n_bulk,
        threshold_frac=threshold_frac,
    )

    # 3. Plot the GB‐plane identification
    wf.gb_plane_plot = plot_GB_plane(
        atoms=structure,
        res=wf.gb_plane_extractor.outputs.gb_plane_analysis_dict,
        projection=plot_projection,
        reps=plot_reps,
        figsize=plot_figsize,
        bulk_color=bulk_color,
        window_cmap=window_cmap,
        plane_linestyles=plane_linestyles,
        axis=plot_axis,
        dpi=plot_dpi,
        save_path=gb_plane_save_path or f"{parent_dir}/pureGB_plane_identifier.jpg",
    )

    # 4. Cleave the structure at the GB plane
    wf.cleave_setup = cleave_gb_structure(
        base_atoms=structure,
        axis_to_cleave=axis_to_cleave,
        target_coord=wf.gb_plane_extractor.outputs.gb_plane_analysis_dict["gb_cart"],
        tol=cleave_tol,
        cleave_region_halflength=cleave_region_halflength,
        layer_tolerance=layer_tolerance,
        separation=separation,
        use_fractional=use_fractional,
    )

    # 5. Generate folder‐names for each cleaved slab
    wf.cleavage_foldernames = get_cleavage_calc_names(
        parent_dir=f"{parent_dir}/S3_RA110_S112",
        cleavage_planes=wf.cleave_setup.outputs.cleavage_plane_coords,
    )

    # 6. Plot the cleaved structures
    wf.cleavage_structure_plot = plot_structure_with_cleavage(
        atoms=structure,
        cleavage_planes=wf.cleave_setup.outputs.cleavage_plane_coords,
        projection=struct_projection,
        reps=struct_reps,
        figsize=struct_figsize,
        atom_color=atom_color,
        plane_color=plane_color,
        plane_linestyle=plane_linestyle,
        atom_size=atom_size,
        save_path=struct_save_path,
        dpi=struct_dpi,
        show_fractional_axes=show_fractional_axes,
        ylims=ylims,
    )

    # 7. Fill in default “full” calculation kwargs (merge overrides)
    wf.full_calc_kwargs = fillin_default_calckwargs(
        calc_kwargs=calc_kwargs,
        default_values=calc_kwargs_defaults
    )

    # 8. Run “fully‐relaxed” calculations on each cleaved slab
    wf.calculate_cleaved = for_node(
        calculate_structure_node,
        zip_on=("structure", "output_dir"),
        structure=wf.cleave_setup.outputs.cleaved_structures,
        output_dir=wf.cleavage_foldernames,
        calc=calc,
        fmax=wf.full_calc_kwargs.outputs.full_calc_kwargs2["fmax"],
        max_steps=wf.full_calc_kwargs.outputs.full_calc_kwargs2["max_steps"],
        properties=wf.full_calc_kwargs.outputs.full_calc_kwargs2["properties"],
        write_to_disk=wf.full_calc_kwargs.outputs.full_calc_kwargs2["write_to_disk"],
        initial_struct_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["initial_struct_path"],
        initial_results_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["initial_results_path"],
        traj_struct_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["traj_struct_path"],
        traj_results_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["traj_results_path"],
        final_struct_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["final_struct_path"],
        final_results_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["final_results_path"],
    )

    # 9. Override “max_steps” for rigid/static runs
    rigid_kwargs = {
        **wf.full_calc_kwargs.outputs.full_calc_kwargs2,
        "max_steps": static_max_steps
    }
    wf.static_calc_kwargs = fillin_default_calckwargs(
        calc_kwargs=rigid_kwargs,
        default_values=None
    )

    # 10. Run “rigid/static” (zero‐step) calculations on each cleaved slab
    wf.calculate_rigid = for_node(
        calculate_structure_node,
        zip_on=("structure", "output_dir"),
        structure=wf.cleave_setup.outputs.cleaved_structures,
        output_dir=wf.cleavage_foldernames,
        calc=calc,
        fmax=wf.static_calc_kwargs.outputs.full_calc_kwargs2["fmax"],
        max_steps=wf.static_calc_kwargs.outputs.full_calc_kwargs2["max_steps"],
        properties=wf.static_calc_kwargs.outputs.full_calc_kwargs2["properties"],
        write_to_disk=wf.static_calc_kwargs.outputs.full_calc_kwargs2["write_to_disk"],
        initial_struct_path=wf.static_calc_kwargs.outputs.full_calc_kwargs2["initial_struct_path"],
        initial_results_path=wf.static_calc_kwargs.outputs.full_calc_kwargs2["initial_results_path"],
        traj_struct_path=wf.static_calc_kwargs.outputs.full_calc_kwargs2["traj_struct_path"],
        traj_results_path=wf.static_calc_kwargs.outputs.full_calc_kwargs2["traj_results_path"],
        final_struct_path=wf.static_calc_kwargs.outputs.full_calc_kwargs2["final_struct_path"],
        final_results_path=wf.static_calc_kwargs.outputs.full_calc_kwargs2["final_results_path"],
    )

    return (
        wf.gb_plane_extractor,
        wf.gb_plane_plot,
        wf.cleave_setup,
        wf.cleavage_foldernames,
        wf.cleavage_structure_plot,
        wf.calculate_cleaved,
        wf.calculate_rigid,
    )
    
@pwf.as_function_node
def get_cleavage_calc_names(parent_dir, cleavage_planes):
    folder_name_list = []
    for plane in cleavage_planes:
        calc_foldername = f"{os.path.basename(parent_dir)}_cp_{np.round(plane,3)}"
        folder_name_list.append(os.path.join(parent_dir, calc_foldername))
    return folder_name_list