import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from ase import Atoms
import pyiron_workflow as pwf
import matplotlib.pyplot as plt
@pwf.as_function_node("gb_plane_analysis_dict")
def find_GB_plane(
    atoms: Atoms,
    featuriser: callable,
    axis: str = "c",
    approx_frac: float = 0.5,
    tolerance: float = 5.0,
    bulk_offset: float = 10.0,
    slab_thickness: float = 2.0,
    featuriser_kwargs: dict | None = None,
    n_bulk: int = 10,
    threshold_frac: float = 0.5
) -> dict:
    """
    Locate the GB plane by finding where disorder (feature-space distance
    from bulk template) begins and ends, then returning the midpoint.
    Only featurises atoms in a narrow GB window and samples up to n_bulk
    atoms in two small bulk slabs.

    Parameters
    ----------
    atoms : ASE Atoms
      Full atomic structure.
    featuriser : function
      Callable featuriser(atoms, site, **featuriser_kwargs) -> dict of floats.
    axis : str or int
      GB normal axis ('a','b','c' or 0,1,2).
    approx_frac : float, optional
      Rough fractional GB location. Defaults to mean(frac coords).
    tolerance : float
      Half-thickness (Å) around approx_frac for GB window.
    bulk_offset : float
      Distance (Å) from approx_frac to centre bulk sampling slabs.
    slab_thickness : float
      Half-thickness (Å) of each bulk sampling slab.
    featuriser_kwargs : dict, optional
      Keyword args passed to featuriser.
    n_bulk : int
      Max number of bulk atoms to sample for template.
    threshold_frac : float
      Fraction of max disorder at which region boundaries are set.

    Returns
    -------
    dict with:
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
      region_start_frac : float or None
        Fraction where disorder first exceeds threshold.
      region_end_frac : float or None
        Fraction where disorder falls below threshold again.
    """
    if featuriser_kwargs is None:
        featuriser_kwargs = {}

    # 1) axis index, fractional coords, cell length
    idx = {"a":0, "b":1, "c":2}[axis] if isinstance(axis, str) else axis
    fracs = atoms.get_scaled_positions()[:, idx] % 1.0
    cell_len = np.linalg.norm(atoms.get_cell()[idx])

    # 2) approximate GB location
    if approx_frac is None:
        approx_frac = fracs.mean()

    # 3) masks for GB window and bulk slabs
    tol_frac = tolerance / cell_len
    off_frac = bulk_offset / cell_len
    slab_frac = slab_thickness / cell_len

    sel_mask = np.abs(fracs - approx_frac) <= tol_frac
    sel_indices = np.where(sel_mask)[0]

    bulk1 = np.abs(fracs - (approx_frac - off_frac)) <= slab_frac
    bulk2 = np.abs(fracs - (approx_frac + off_frac)) <= slab_frac
    bulk_all = np.where(bulk1 | bulk2)[0]

    # 4) sample bulk indices
    if len(bulk_all) <= n_bulk:
        bulk_indices = bulk_all
    else:
        bulk_indices = np.random.choice(bulk_all, n_bulk, replace=False)

    # 5) build bulk template
    feats_bulk = [pd.Series(featuriser(atoms, i, **featuriser_kwargs))
                  for i in bulk_indices]
    df_bulk = pd.DataFrame(feats_bulk).fillna(0.0)
    bulk_template = df_bulk.mean(axis=0).values

    # 6) featurise GB window
    feats_sel = [pd.Series(featuriser(atoms, i, **featuriser_kwargs))
                 for i in sel_indices]
    df_sel = pd.DataFrame(feats_sel).fillna(0.0)
    X_sel = df_sel.values

    # 7) disorder scores
    scores = np.linalg.norm(X_sel - bulk_template[None, :], axis=1)
    sel_fracs = fracs[sel_indices]

    # 8) find region boundaries and mid
    order = np.argsort(sel_fracs)
    fs = sel_fracs[order]
    ss = scores[order]
    i_peak = np.argmax(ss)
    peak = ss[i_peak]
    thr = threshold_frac * peak

    # left boundary: last < thr before peak
    left_idxs = np.where(ss[:i_peak] < thr)[0]
    # right boundary: first < thr after peak
    right_idxs = np.where(ss[i_peak:] < thr)[0]

    if left_idxs.size and right_idxs.size:
        start_frac = fs[left_idxs[-1]]
        end_frac = fs[i_peak + right_idxs[0]]
        mid_frac = 0.5 * (start_frac + end_frac)
    else:
        # fallback to peak location
        start_frac = None
        end_frac = None
        mid_frac = fs[i_peak]

    mid_cart = mid_frac * cell_len

    return {
        "gb_frac": mid_frac,
        "gb_cart": mid_cart,
        "sel_indices": sel_indices.tolist(),
        "bulk_indices": bulk_indices.tolist(),
        "sel_fracs": sel_fracs,
        "scores": scores,
        "region_start_frac": start_frac,
        "region_end_frac": end_frac
    }

@pwf.as_function_node
def plot_GB_plane(
    atoms: Atoms,
    res: dict,
    projection=(0, 2),
    reps=(1, 1),
    figsize=(8, 6),
    bulk_color='C0',
    window_cmap='viridis',
    plane_linestyles=('--', '-'),
    axis=2,
    save_path=None,
    dpi=300
):
    """
    Plot GB disorder-region analysis results:
      - bulk_indices: sampled bulk atoms (background point cloud)
      - sel_indices: GB-window atoms colored by disorder score
    Overlays the start/end of the disorder region and the mid-plane.

    Parameters
    ----------
    atoms : ASE Atoms
        Full ASE structure for obtaining cell vectors and positions.
    res : dict
        Output of find_gb_midplane_region, containing keys:
          - 'bulk_indices': List[int]
          - 'sel_indices': List[int]
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
    plane_linestyles : tuple(str, str)
        Line styles for (region boundaries, mid-plane).
    axis : int
        Index of the GB-normal axis used in mid-plane calculation.
    save_path : str or None
        File path to save the figure. If None, the figure is not saved.
    dpi : int
        Resolution in dots per inch when saving.
    """
    # Unpack projection and cell
    p0, p1 = projection
    cell = atoms.get_cell()
    shifts = [i * cell[p0] + j * cell[p1] for i in range(reps[0]) for j in range(reps[1])]

    # Extract positions
    pos = atoms.get_positions()
    bulk_pos = pos[res['bulk_indices']]
    window_pos = pos[res['sel_indices']]
    scores = res['scores']

    # Region boundaries and mid-plane
    start_frac = res.get('region_start_frac')
    end_frac = res.get('region_end_frac')
    gb_cart = res['gb_cart']
    cell_len = np.linalg.norm(cell[axis])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot samples and window
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]

        # Bulk samples
        bx = bulk_pos[:, p0] + sx
        by = bulk_pos[:, p1] + sy
        ax.scatter(
            bx, by,
            s=20, color=bulk_color, alpha=0.5,
            label='Bulk samples' if shift is shifts[0] else None
        )

        # GB-window atoms
        wx = window_pos[:, p0] + sx
        wy = window_pos[:, p1] + sy
        sc = ax.scatter(
            wx, wy,
            c=scores, cmap=window_cmap, s=50,
            label='GB window' if shift is shifts[0] else None
        )

    # Overlay region boundaries and mid-plane
    bstyle, mstyle = plane_linestyles
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]
        if start_frac is not None and end_frac is not None:
            start_cart = start_frac * cell_len
            end_cart = end_frac * cell_len
            if axis == p0:
                ax.axvline(
                    start_cart + sx,
                    linestyle=bstyle, color='grey',
                    label='Region boundaries' if shift is shifts[0] else None
                )
                ax.axvline(end_cart + sx, linestyle=bstyle, color='grey')
            elif axis == p1:
                ax.axhline(
                    start_cart + sy,
                    linestyle=bstyle, color='grey',
                    label='Region boundaries' if shift is shifts[0] else None
                )
                ax.axhline(end_cart + sy, linestyle=bstyle, color='grey')

        # Mid-plane
        if axis == p0:
            ax.axvline(
                gb_cart + sx,
                linestyle=mstyle, color='k',
                label='Mid-plane' if shift is shifts[0] else None
            )
        elif axis == p1:
            ax.axhline(
                gb_cart + sy,
                linestyle=mstyle, color='k',
                label='Mid-plane' if shift is shifts[0] else None
            )

    # Labels and aesthetics
    ax.set_xlabel(f'Axis {p0} (Å)')
    ax.set_ylabel(f'Axis {p1} (Å)')
    ax.set_title(f'GB disorder region (proj {p0}–{p1})')
    ax.set_aspect('equal')
    ax.autoscale()

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Disorder score')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi)

    plt.show()
    return fig, ax