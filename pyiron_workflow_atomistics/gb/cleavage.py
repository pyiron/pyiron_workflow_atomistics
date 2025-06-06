import numpy as np
from ase import Atoms
import matplotlib.pyplot as plt
import pyiron_workflow as pwf
from .analysis import get_sites_on_plane

from pyiron_workflow_atomistics.featurisers import voronoiSiteFeaturiser
from pyiron_workflow_atomistics.gb.analysis import find_GB_plane, plot_GB_plane
from pyiron_workflow_atomistics.calculator import fillin_default_calckwargs, calculate_structure_node
import os
from pyiron_workflow.api import for_node
from pyiron_workflow_atomistics.pyiron_workflow_atomistics.gb.optimiser import fillin_default_calckwargs
import pandas as pd
import pandas as pd
import numpy as np
from ase import Atoms

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
        Which axis to cleave along: 0 = a, 1 = b, 2 = c.
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
    if save_path:
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # leave room on right
        fig.savefig(save_path, dpi=dpi)
    else:
        plt.tight_layout(rect=[0, 0, 0.75, 1])
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
    
@pwf.as_function_node
def get_cleavage_calc_names(parent_dir,
                            cleavage_planes):
    folder_name_list = []
    for plane in cleavage_planes:
        calc_foldername = f"{os.path.basename(parent_dir)}_cp_{np.round(plane,3)}"
        folder_name_list.append(os.path.join(parent_dir, calc_foldername))
    return folder_name_list


@pwf.as_function_node("df")
def get_results_df(df,
                   cleavage_coords,
                   cleaved_structures,
                   uncleaved_energy,
                   cleavage_axis: str = "c"):
    """
    Assemble a results DataFrame from cleavage scan outputs and compute cleavage energy.

    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame from structure calculations. Must have 'results' and 'atoms' columns.
    cleavage_coords : list[float]
        List of cleavage plane coordinates.
    cleaved_structures : list[ase.Atoms]
        List of the initial cleaved ASE structures.
    uncleaved_energy : float
        Energy of the original un-cleaved (reference) structure.
    cleavage_axis : str
        Axis along which the cleavage occurs ("x", "y", or "z").

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['cleavage_coord', 'initial_structure', 'final_structure', 'energy', 'cleavage_energy']
        where cleavage_energy is in J/m².
    """
    relaxed_structures = df.atoms.tolist()
    energies = [res["energy"] for res in df.results.tolist()]
    
    axis_index = {"a": 0, "b": 1, "c": 2}[cleavage_axis.lower()]
    cleavage_energies = []

    for E, struct in zip(energies, relaxed_structures):
        cell = struct.get_cell()
        # Get the 2 vectors that span the cleavage plane perpendicular to the cleavage axis
        a1, a2 = np.delete(cell, axis_index, axis=0)
        area = np.linalg.norm(np.cross(a1, a2))  # in Å²
        #print(area, struct.cell[-2][-2] *struct.cell[0][0])
        # Cleavage energy in J/m²
        E_cleave = (E - uncleaved_energy) / (area) * 16.0218  # eV/Å² → J/m² # Only 1 GB so no 2 factor on bottom
        cleavage_energies.append(E_cleave)

    return pd.DataFrame({
        "cleavage_coord": cleavage_coords,
        "initial_structure": cleaved_structures,
        "final_structure": relaxed_structures,
        "energy": energies,
        "cleavage_energy": cleavage_energies
    })


@pwf.as_function_node
def toggle_rigid_calc(rigid, calc_kwargs):
    if rigid:
        max_steps = 0
    else:
        max_steps = calc_kwargs["max_steps"]
    return max_steps
from pyiron_workflow_atomistics.gb.dataclass_storage import CleaveGBStructureInput, PlotCleaveInput
@pwf.as_macro_node("cleaved_structure_list",
                    "cleaved_plane_coords_list",
                    "cleavage_plane_plot_fig",
                   "cleavage_plane_plot_ax",
                   "cleavage_calcs_df")
def calc_cleavage_GB(wf,
                     structure: Atoms,
                     energy,
                     calc,
                     input_cleave_gb_structure: CleaveGBStructureInput,
                     input_plot_cleave: PlotCleaveInput,
                     input_calc_structure: dict,
                     parent_dir: str = "gb_cleavage",
                     rigid=True):
    wf.cleave_setup = cleave_gb_structure(
        base_atoms=structure,
        axis_to_cleave=input_cleave_gb_structure.axis_to_cleave,
        target_coord=input_cleave_gb_structure.cleavage_target_coord,
        tol=input_cleave_gb_structure.tol,
        cleave_region_halflength=input_cleave_gb_structure.cleave_region_halflength,
        layer_tolerance=input_cleave_gb_structure.layer_tolerance,
        separation=input_cleave_gb_structure.separation,
        use_fractional=input_cleave_gb_structure.use_fractional,
    )
    wf.cleavage_structure_plot = plot_structure_with_cleavage(
        atoms=structure,
        cleavage_planes=wf.cleave_setup.outputs.cleavage_plane_coords,
        projection=input_plot_cleave.projection,
        reps=input_plot_cleave.reps,
        figsize=input_plot_cleave.figsize,
        atom_color=input_plot_cleave.atom_color,
        plane_color=input_plot_cleave.plane_color,
        plane_linestyle=input_plot_cleave.plane_linestyle,
        atom_size=input_plot_cleave.atom_size,
        save_path=input_plot_cleave.save_path,
        dpi=input_plot_cleave.dpi,
        show_fractional_axes=input_plot_cleave.show_fractional_axes,
        ylims=input_plot_cleave.ylims,
    )

    wf.full_calc_kwargs = fillin_default_calckwargs(calc_kwargs = input_calc_structure,
                                                    default_values = None)
    wf.cleave_structure_foldernames = get_cleavage_calc_names(parent_dir = parent_dir,
                                                               cleavage_planes = wf.cleave_setup.outputs.cleavage_plane_coords)
    wf.toggle_rigid_calc = toggle_rigid_calc(rigid=rigid,
                                             calc_kwargs = wf.full_calc_kwargs.outputs.full_calc_kwargs2)
    wf.calculate_cleaved = pwf.api.for_node(
        calculate_structure_node,
        zip_on=("structure", "output_dir"),
        structure=wf.cleave_setup.outputs.cleaved_structures,
        output_dir=wf.cleave_structure_foldernames,
        calc=calc,
        fmax=wf.full_calc_kwargs.outputs.full_calc_kwargs2["fmax"],
        max_steps=wf.toggle_rigid_calc.outputs.max_steps,
        properties=wf.full_calc_kwargs.outputs.full_calc_kwargs2["properties"],
        write_to_disk=wf.full_calc_kwargs.outputs.full_calc_kwargs2["write_to_disk"],
        initial_struct_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["initial_struct_path"],
        initial_results_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["initial_results_path"],
        traj_struct_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["traj_struct_path"],
        traj_results_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["traj_results_path"],
        final_struct_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["final_struct_path"],
        final_results_path=wf.full_calc_kwargs.outputs.full_calc_kwargs2["final_results_path"],
    )
    wf.collate_results = get_results_df(df = wf.calculate_cleaved.outputs.df,
                                        cleavage_coords = wf.cleave_setup.outputs.cleavage_plane_coords,
                                        cleaved_structures = wf.cleave_setup.outputs.cleaved_structures,
                                        uncleaved_energy = energy,
                                        cleavage_axis = input_cleave_gb_structure.axis_to_cleave)

    return (wf.cleave_setup.outputs.cleaved_structures,
            wf.cleave_setup.outputs.cleavage_plane_coords,
            wf.cleavage_structure_plot.outputs.fig,
            wf.cleavage_structure_plot.outputs.ax,
            wf.collate_results.outputs.df)