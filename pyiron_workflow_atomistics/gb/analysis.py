import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from ase import Atoms
from ase.atoms import Atom
import pyiron_workflow as pwf
import matplotlib.pyplot as plt
from pyiron_workflow_atomistics.gb.utils import axis_to_index
import os
from pyiron_snippets.logger import logger


@pwf.as_function_node("atom")
def get_middle_atom(atoms: Atoms, axis: int | str = 2) -> Atom:
    """
    Return the index of the atom whose coordinate along the given axis
    is closest to the mid-plane of the cell.

    Parameters
    ----------
    atoms : ase.Atoms
        The supercell.
    axis : int or {'x','y','z'}, default=2
        Which axis to slice along. 0='x', 1='y', 2='z'.

    Returns
    -------
    idx : int
        Index of the atom closest to the center along that axis.
    """
    # allow strings 'x','y','z'
    if isinstance(axis, str):
        axis = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    # get fractional positions along axis (handles PBC nicely)
    scaled = atoms.get_scaled_positions()[:, axis]
    # the mid-plane in fractional coords is always 0.5
    target = 0.5
    # find atom nearest to that plane
    idx = int(np.argmin(np.abs(scaled - target)))
    return atoms[idx]


@pwf.as_function_node("gb_plane_analysis_dict")
def find_GB_plane(
    atoms: Atoms,
    featuriser: callable,
    axis: str = "c",
    approx_frac: float | None = None,
    tolerance: float = 5.0,
    bulk_offset: float = 10.0,
    slab_thickness: float = 2.0,
    featuriser_kwargs: dict | None = None,
    n_bulk: int = 10,
    threshold_frac: float = 0.5,
    extend_region_length: int | float = 0.0,
) -> dict:
    """
    Locate the GB plane by finding where disorder (feature-space distance
    from bulk template) begins and ends, then returning the midpoint.
    Always returns region_start_frac and region_end_frac, even if multiple
    equal peaks are found. Optionally extend the selection by a Cartesian distance.

    Parameters
    ----------
    atoms : ASE Atoms
    featuriser : callable
    axis : str or int
        GB normal axis ('a','b','c' or 0,1,2).
    approx_frac : float, optional
        Rough fractional GB location (default = mean(frac coords)).
    tolerance : float
        Half-thickness (Å) around approx_frac for GB window.
    bulk_offset : float
        Distance (Å) from approx_frac to centre bulk sampling slabs.
    slab_thickness : float
        Half-thickness (Å) of each bulk sampling slab.
    featuriser_kwargs : dict, optional
    n_bulk : int
        Max number of bulk atoms to sample for template.
    threshold_frac : float
        Fraction of peak disorder at which region boundaries are set.
    extend_frac : float
        Cartesian distance (Å) to extend region boundaries by along the axis.

    Returns
    -------
    dict with keys:
      gb_frac : float
        Fractional mid-plane coordinate.
      gb_cart : float
        Cartesian mid-plane coordinate (Å).
      sel_indices : List[int]
        Atoms featurised in GB window.
      bulk_indices : List[int]
        Bulk-sampled atom indices used for template.
      sel_fracs : np.ndarray
        Fractional positions of sel_indices.
      scores : np.ndarray
        Disorder scores for sel_indices.
      region_start_frac : float
        Fraction where disorder first drops below the threshold (before first peak).
      region_end_frac : float
        Fraction where disorder first drops below the threshold (after last peak).
      extended_sel_indices : List[int]
        All atoms within [region_start_frac – extend_frac_frac, region_end_frac + extend_frac_frac].
    """
    import numpy as np
    import pandas as pd

    if featuriser_kwargs is None:
        featuriser_kwargs = {}

    # 1) axis index, fractional coords, cell length
    idx = axis_to_index(axis)
    fracs = atoms.get_scaled_positions()[:, idx] % 1.0
    cell_len = np.linalg.norm(atoms.get_cell()[idx])

    # 2) approximate GB location if not provided
    if approx_frac is None:
        approx_frac = fracs.mean()

    # 3) masks for GB window and bulk slabs (in fractional units)
    tol_frac = tolerance / cell_len
    off_frac = bulk_offset / cell_len
    slab_frac = slab_thickness / cell_len

    sel_mask = np.abs(fracs - approx_frac) <= tol_frac
    sel_indices = np.where(sel_mask)[0]

    bulk1 = np.abs(fracs - (approx_frac - off_frac)) <= slab_frac
    bulk2 = np.abs(fracs - (approx_frac + off_frac)) <= slab_frac
    bulk_all = np.where(bulk1 | bulk2)[0]

    # 4) sample bulk indices (up to n_bulk)
    if len(bulk_all) <= n_bulk:
        bulk_indices = bulk_all
    else:
        bulk_indices = np.random.choice(bulk_all, n_bulk, replace=False)

    # 5) build bulk template (mean of feature‐vectors)
    feats_bulk = [
        pd.Series(featuriser(atoms, i, **featuriser_kwargs)) for i in bulk_indices
    ]
    df_bulk = pd.DataFrame(feats_bulk).fillna(0.0)
    bulk_template = df_bulk.mean(axis=0).values

    # 6) featurise the GB window atoms
    feats_sel = [
        pd.Series(featuriser(atoms, i, **featuriser_kwargs)) for i in sel_indices
    ]
    df_sel = pd.DataFrame(feats_sel).fillna(0.0)
    X_sel = df_sel.values

    # 7) compute disorder scores and get fractional positions
    scores = np.linalg.norm(X_sel - bulk_template[None, :], axis=1)
    sel_fracs = fracs[sel_indices]

    # 8) find region boundaries and midpoint
    order = np.argsort(sel_fracs)
    fs = sel_fracs[order]
    ss = scores[order]
    peak_val = ss.max()
    i_peaks = np.where(np.isclose(ss, peak_val, atol=0.1))[0]

    # threshold for region edges
    thr = threshold_frac * peak_val

    # If multiple equal peaks exist:
    if len(i_peaks) >= 2:
        logger.info("double peak detected")

        # Define first and last peak positions
        i_peak_min = i_peaks.min()
        i_peak_max = i_peaks.max()
        logger.info(f"i_peak_min: {i_peak_min}, i_peak_max: {i_peak_max}")
        p_min = fs[i_peak_min]
        p_max = fs[i_peak_max]

        # Find left boundary: last index < thr before i_peak_min
        left_idxs = np.where(ss[:i_peak_min] < thr)[0]
        if left_idxs.size:
            start_frac = fs[left_idxs[-1]]
        else:
            # If no drop below thr before first peak, use p_min itself
            start_frac = p_min

        # Find right boundary: first index < thr after i_peak_max
        right_rel_idxs = np.where(ss[i_peak_max:] < thr)[0]
        if right_rel_idxs.size:
            end_frac = fs[i_peak_max + right_rel_idxs[0]]
        else:
            # If no drop below thr after last peak, use p_max
            end_frac = p_max

        # Midpoint between the extreme peak positions (wrap‐aware)
        # If p_max–p_min > 0.5, adjust for wrap
        if (p_max - p_min) > 0.5:
            p_min += 1.0
        mid_frac = (0.5 * (p_min + p_max)) % 1.0

    else:
        logger.info("single peak detected")
        # Only one clear peak → treat it as a single index
        i_peak = i_peaks[0]

        # left boundary: last < thr before i_peak
        left_idxs = np.where(ss[:i_peak] < thr)[0]
        if left_idxs.size:
            start_frac = fs[left_idxs[-1]]
        else:
            # If none found, set to the peak itself
            start_frac = fs[i_peak]

        # right boundary: first < thr after i_peak
        right_rel_idxs = np.where(ss[i_peak:] < thr)[0]
        if right_rel_idxs.size:
            end_frac = fs[i_peak + right_rel_idxs[0]]
        else:
            # If none found, set to the peak itself
            end_frac = fs[i_peak]

        mid_frac = 0.5 * (start_frac + end_frac) % 1.0

    mid_cart = mid_frac * cell_len
    frac_diffs = np.abs(sel_fracs - mid_frac)
    i_mid = np.argmin(frac_diffs)
    mid_index = sel_indices[i_mid]

    # 9) optionally extend selection by extend_frac (Å → fraction)
    if extend_region_length > 0 and start_frac is not None and end_frac is not None:
        # Convert Cartesian extend_frac into fractional units
        extend_frac = extend_region_length / cell_len

        lower = start_frac - extend_frac
        upper = end_frac + extend_frac

        # Handle wrap‐around in fraction space
        if lower < 0 or upper > 1.0:
            low_mod = lower % 1.0
            high_mod = upper % 1.0
            if low_mod < high_mod:
                ext_mask = (fracs >= low_mod) & (fracs <= high_mod)
            else:
                ext_mask = (fracs >= low_mod) | (fracs <= high_mod)
        else:
            ext_mask = (fracs >= lower) & (fracs <= upper)

        extended_sel_indices = np.where(ext_mask)[0].tolist()
    else:
        extended_sel_indices = sel_indices.tolist()

    return {
        "gb_frac": mid_frac,
        "gb_cart": mid_cart,
        "mid_index": mid_index,
        "sel_indices": sel_indices.tolist(),
        "bulk_indices": bulk_indices.tolist(),
        "sel_fracs": sel_fracs,
        "scores": scores,
        "region_start_frac": start_frac,
        "region_end_frac": end_frac,
        "extended_sel_indices": extended_sel_indices,
    }


@pwf.as_function_node
def plot_GB_plane(
    atoms: Atoms,
    res: dict,
    projection=(0, 2),
    reps=(1, 1),
    figsize=(8, 6),
    bulk_color="C0",
    window_cmap="viridis",
    extended_only_color="r",
    plane_linestyles=("--", "-"),
    axis=2,
    save_filename=None,
    working_directory=None,
    dpi=300,
):
    """
    Plot GB disorder-region analysis results:
      - bulk_indices: sampled bulk atoms (background point cloud)
      - sel_indices: GB-window atoms colored by disorder score
      - extended-only indices: atoms in extended_sel_indices but not in sel_indices (plotted as red crosses)
    Overlays the start/end of the disorder region and the mid-plane.

    Parameters
    ----------
    atoms : ASE Atoms
        Full ASE structure for obtaining cell vectors and positions.
    res : dict
        Output of find_GB_plane (latest version), containing keys:
          - 'bulk_indices': List[int]
          - 'sel_indices': List[int]
          - 'extended_sel_indices': List[int]
          - 'scores': np.ndarray of disorder scores
          - 'sel_fracs': np.ndarray of fractional positions
          - 'region_start_frac', 'region_end_frac': floats
          - 'gb_cart': float (mid-plane in Å)
    projection : tuple(int, int)
        Pair of Cartesian axes to project onto (e.g. (0,2)).
    reps : tuple(int, int)
        Number of repeats along projection axes for tiling.
    figsize : tuple
        Figure size (width, height) in inches.
    bulk_color : color
        Color for sampled bulk atoms.
    window_cmap : str or Colormap
        Colormap for coloring GB-window atoms by score.
    extended_only_color : color
        Color for atoms exclusively in the extended region (plotted as crosses).
    plane_linestyles : tuple(str, str)
        Line styles for (region boundaries, mid-plane).
    axis : int
        Index of the GB-normal axis used in mid-plane calculation.
    save_path : str or None
        File path to save the figure. If None, the figure is not saved.
    dpi : int
        Resolution in dots per inch when saving.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Unpack projection and cell
    p0, p1 = projection
    cell = atoms.get_cell()
    shifts = [
        i * cell[p0] + j * cell[p1] for i in range(reps[0]) for j in range(reps[1])
    ]

    # Extract positions
    pos = atoms.get_positions()
    bulk_indices = res.get("bulk_indices", [])
    sel_indices = res["sel_indices"]
    ext_indices = res.get("extended_sel_indices", [])

    bulk_pos = pos[bulk_indices] if bulk_indices else np.zeros((0, 3))
    window_pos = pos[sel_indices] if sel_indices else np.zeros((0, 3))

    # Determine "exclusive extended" indices = extended_sel_indices minus sel_indices
    sel_set = set(sel_indices)
    ext_only_indices = [i for i in ext_indices if i not in sel_set]
    ext_only_pos = pos[ext_only_indices] if ext_only_indices else np.zeros((0, 3))

    scores = res["scores"]

    # Region boundaries and mid-plane
    start_frac = res.get("region_start_frac")
    end_frac = res.get("region_end_frac")
    gb_cart = res["gb_cart"]
    cell_len = np.linalg.norm(cell[axis])

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Plot bulk samples and GB-window atoms, tiled
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]

        # Bulk samples
        if bulk_pos.size:
            bx = bulk_pos[:, p0] + sx
            by = bulk_pos[:, p1] + sy
            ax.scatter(
                bx,
                by,
                s=20,
                color=bulk_color,
                alpha=0.5,
                label="Bulk samples" if shift is shifts[0] else None,
            )

        # GB-window atoms (colored by disorder score)
        if window_pos.size:
            wx = window_pos[:, p0] + sx
            wy = window_pos[:, p1] + sy
            sc = ax.scatter(
                wx,
                wy,
                c=scores,
                cmap=window_cmap,
                s=50,
                label="GB search window" if shift is shifts[0] else None,
            )

    # 2) Plot atoms exclusively in the extended region (as crosses)
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]
        if ext_only_pos.size:
            ex = ext_only_pos[:, p0] + sx
            ey = ext_only_pos[:, p1] + sy
            ax.scatter(
                ex,
                ey,
                marker="x",
                s=80,
                color=extended_only_color,
                label="Extended-only" if shift is shifts[0] else None,
            )

    # 3) Overlay region boundaries and mid-plane lines
    bstyle, mstyle = plane_linestyles
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]

        # Region boundaries, if defined
        if start_frac is not None and end_frac is not None:
            start_cart = start_frac * cell_len
            end_cart = end_frac * cell_len
            if axis == p0:
                ax.axvline(
                    start_cart + sx,
                    linestyle=bstyle,
                    color="grey",
                    label="Region boundaries" if shift is shifts[0] else None,
                )
                ax.axvline(end_cart + sx, linestyle=bstyle, color="grey")
            elif axis == p1:
                ax.axhline(
                    start_cart + sy,
                    linestyle=bstyle,
                    color="grey",
                    label="Region boundaries" if shift is shifts[0] else None,
                )
                ax.axhline(end_cart + sy, linestyle=bstyle, color="grey")

        # Mid-plane
        if axis == p0:
            ax.axvline(
                gb_cart + sx,
                linestyle=mstyle,
                color="k",
                label="algo-GB-plane" if shift is shifts[0] else None,
            )
        elif axis == p1:
            ax.axhline(
                gb_cart + sy,
                linestyle=mstyle,
                color="k",
                label="algo-GB-plane" if shift is shifts[0] else None,
            )

    # Labels and aesthetics
    ax.set_xlabel(f"Axis {p0} (Å)")
    ax.set_ylabel(f"Axis {p1} (Å)")
    ax.set_title(f"GB disorder region (proj {p0}–{p1})")
    ax.set_aspect("equal")
    ax.autoscale()

    # Colorbar for the GB-window scatter (disorder scores)
    if window_pos.size:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Disorder score")

    # Legend (ensuring one entry per label)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    # Save if requested
    if save_filename:
        os.makedirs(working_directory, exist_ok=True)
        fig.savefig(os.path.join(working_directory, save_filename), dpi=dpi)

    plt.show()
    return fig, ax


@pwf.as_function_node
def get_sites_on_plane(
    atoms: Atoms,
    axis: str,
    target_coord: float,
    tol: float = 1e-2,
    use_fractional: bool = False,
) -> list[int]:
    """
    Return the indices of all atoms whose coordinate along a given cell axis
    lies within a specified tolerance of a target value, either in Cartesian
    (Å) or fractional (0–1) units.

    Parameters
    ----------
    atoms : ASE Atoms
        The atomic structure to query.
    axis : str
        The cell‐axis to use. Accepts 'a','b','c' to indicate the
        first, second, or third cell vector.
    target_coord : float
        The target coordinate. If use_fractional is False, interprets this as
        a Cartesian coordinate in Å. If use_fractional is True, interprets
        this as a fractional coordinate (0 ≤ target_coord < 1).
    tol : float, optional
        If use_fractional is False, the half‐width tolerance (in Å) around
        target_coord. Default is 1e-2 Å.
        If use_fractional is True, the half‐width tolerance (in fractional
        units) around target_coord. Default is 1e-2 (i.e. ±0.01 in fraction).
    use_fractional : bool, optional
        If False (default), compare Cartesian positions. If True, compare
        fractional positions.

    Returns
    -------
    List[int]
        A list of atom indices whose coordinate along the specified axis
        lies within [target_coord − tol, target_coord + tol], handling
        wrap‐around correctly in fractional mode.

    Notes
    -----
    - In Cartesian mode, no periodic wrap‐around is applied. If your cell
      is non‐orthogonal or you want to include atoms in neighboring images,
      first tile the cell (e.g., `atoms = atoms.repeat((nx,ny,nz))`).
    - In fractional mode, wrap‐around is handled so that, for example, a
      target_frac near 0 or 1 includes atoms just below 1 or just above 0
      when within tol.
    """
    import numpy as np

    # 1) Determine axis index
    idx = axis_to_index(axis)

    if use_fractional:
        # 2a) Get fractional coords in [0,1)
        fracs = atoms.get_scaled_positions()[:, idx] % 1.0
        # 3a) Compute wrap‐aware distance: |((f - target + 0.5) % 1.0) - 0.5|
        diffs = np.abs((fracs - target_coord + 0.5) % 1.0 - 0.5)
        matched_indices = np.where(diffs <= tol)[0].tolist()
    else:
        # 2b) Get Cartesian coords (Å)
        cart_coords = atoms.get_positions()[:, idx]
        # 3b) Select indices where |coord - target_coord| <= tol
        matched_indices = np.where(np.abs(cart_coords - target_coord) <= tol)[
            0
        ].tolist()

    return matched_indices


import matplotlib.pyplot as plt
import numpy as np


def plot_structure_2d(
    atoms,
    projection=(0, 2),
    reps=(1, 1),
    figsize=(6, 6),
    atom_color="C0",
    atom_size=30,
    save_path=None,
    dpi=300,
):
    """
    Plot a 2D projection of `atoms` (scatter) with optional periodic tiling.

    Parameters
    ----------
    atoms : ASE Atoms
        The atomic structure to visualize.
    projection : tuple(int, int), optional
        Pair of Cartesian axes to project onto (e.g. (0, 2) for x–z).
    reps : tuple(int, int), optional
        Number of repeats along each projection axis (for periodic tiling).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    atom_color : color, optional
        Color for the atom scatter points.
    atom_size : int, optional
        Marker size for atoms.
    save_path : str or None, optional
        If provided, path to save the figure (PNG, etc.).
    dpi : int, optional
        Resolution in dots per inch if the figure is saved.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    p0, p1 = projection
    cell = atoms.get_cell()
    # Compute tiling shifts along the two projected cell vectors
    shifts = [
        i * cell[p0] + j * cell[p1] for i in range(reps[0]) for j in range(reps[1])
    ]

    # Extract atomic positions
    pos = atoms.get_positions()
    xs = pos[:, p0]
    ys = pos[:, p1]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each periodic tile
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]
        ax.scatter(
            xs + sx,
            ys + sy,
            s=atom_size,
            color=atom_color,
            label="Atoms" if shift is shifts[0] else None,
        )

    ax.set_xlabel(f"Axis {p0} (Å)")
    ax.set_ylabel(f"Axis {p1} (Å)")
    ax.set_title(f"2D Projection (proj {p0}-{p1})")
    ax.set_aspect("equal")
    ax.autoscale()

    # Show legend once
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi)
    else:
        plt.tight_layout()

    plt.show()
    return fig, ax
