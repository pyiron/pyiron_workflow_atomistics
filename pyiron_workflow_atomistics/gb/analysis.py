import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from ase import Atoms
import pyiron_workflow as pwf
import matplotlib.pyplot as plt
@pwf.as_function_node("gb_plane_analysis_dict")
def find_GB_plane(
    atoms: Atoms,
    featuriser,
    axis: str = "c",
    cutoff: float = 1.0,
    extra_region: float = 1.0,
    clustering_eps: float = 0.5,
    clustering_min_samples: int = 5,
    featuriser_k: int = 6,
) -> dict:
    """
    ...
    - extended_indices : List[int]
        All atoms whose coordinate along `axis` lies between
        (min_outlier_coord - extra_region) and (max_outlier_coord + extra_region).
    """
    # 1) axis → index
    idx = {"a":0,"b":1,"c":2}[axis] if isinstance(axis, str) else axis

    # 2) fractional positions and mean‐plane selection
    fracs = atoms.get_scaled_positions()[:, idx]
    plane_mean = fracs.mean()
    cell_len = np.linalg.norm(atoms.get_cell()[idx])
    delta_frac = cutoff / cell_len
    selected = np.where(np.abs(fracs - plane_mean) <= delta_frac)[0].tolist()

    # 3) cartesian coords of selected
    pos = atoms.get_positions()
    selected_coords = pos[selected].tolist()

    # 4) featurise & DBSCAN‐cluster
    feats = []
    for i in selected:
        d = featuriser(atoms, i, k=featuriser_k)
        d["site"] = i
        feats.append(d)
    feature_df = pd.DataFrame(feats).set_index("site")

    X = feature_df.select_dtypes(float).values
    labels = DBSCAN(eps=clustering_eps,
                    min_samples=clustering_min_samples).fit_predict(X)
    cluster_labels = dict(zip(selected, labels.tolist()))
    outliers = [i for i, lab in cluster_labels.items() if lab == -1]

    # 5) determine c‐coords of outliers and build extended window
    all_coords = pos[:, idx]
    if outliers:
        out_coords = all_coords[outliers]
        min_o, max_o = out_coords.min(), out_coords.max()
    else:
        # fallback to plane‐mean single slice
        min_o = max_o = plane_mean * cell_len

    lower = min_o - extra_region
    upper = max_o + extra_region
    extended = np.where((all_coords >= lower) & (all_coords <= upper))[0].tolist()

    # 6) compute mid‐planes
    #    outlier window midpoint
    mid_outlier_cart = 0.5 * (min_o + max_o)
    mid_outlier_frac = mid_outlier_cart / cell_len

    #    extended window midpoint
    mid_extended_cart = 0.5 * (lower + upper)
    mid_extended_frac = mid_extended_cart / cell_len

    return {
        "selected_indices":        selected,
        "selected_coords":         selected_coords,
        "feature_df":              feature_df,
        "cluster_labels":          cluster_labels,
        "outlier_indices":         outliers,
        "plane_mean_frac":         plane_mean,
        "plane_min_frac":          fracs.min(),
        "plane_max_frac":          fracs.max(),
        "extended_indices":        extended,
        "mid_outlier_plane_cart":  mid_outlier_cart,
        "mid_outlier_plane_frac":  mid_outlier_frac,
        "mid_extended_plane_cart": mid_extended_cart,
        "mid_extended_plane_frac": mid_extended_frac,
    }

@pwf.as_function_node
def plot_selected_clusters(atoms: Atoms,
                           res: dict,
                           projection=[0, 2],
                           reps=(1, 1),
                           figsize=(8, 6),
                           extended_color='r',
                           non_outlier_color='C0',
                           outlier_color='k',
                           plane_linestyles=('--', '-'),
                           axis=2,
                           save_path=None,
                           dpi=300):
    """
    Plot GB-plane-selected atoms tiled periodically, with three tiers:
      1) extended_indices (background dots)
      2) selected non-outliers (one colour)
      3) selected outliers (black crosses)
    Also overlays the mid-outlier and mid-extended planes.
    Optionally save the figure to disk.

    Parameters
    ----------
    atoms : ase.Atoms
      Full ASE structure (for cell vectors).
    res : dict
      Output of find_GB_plane, must have:
        - "extended_indices": list[int]
        - "selected_indices": list[int]
        - "selected_coords":  list of [x,y,z]
        - "cluster_labels":   dict site→label
        - "mid_outlier_plane_cart": float
        - "mid_extended_plane_cart": float
    projection : [int,int]
      Which Cartesian axes to plot (0=x,1=y,2=z).
    reps : (int,int)
      Tiling repeats along proj[0] and proj[1].
    figsize : tuple
      Figure size in inches (width, height).
    extended_color : color
      Colour for the extended-region sites (background).
    non_outlier_color : color
      Colour for selected inliers.
    outlier_color : color
      Colour for selected outliers.
    plane_linestyles : tuple[str,str]
      Line styles for (extended_plane, outlier_plane).
    axis : int
      Index of the axis along which the GB plane was computed.
    save_path : str or None
      Path to save the figure. If None, the figure is not saved.
    dpi : int
      Resolution in dots per inch for saving.
    """
    # unpack
    sel_idx      = res["selected_indices"]
    sel_coords   = np.array(res["selected_coords"])
    labels_map   = res["cluster_labels"]
    ext_idx      = res.get("extended_indices", [])
    ext_coords   = atoms.get_positions()[ext_idx] if ext_idx else np.zeros((0,3))

    # mid-planes
    mid_ext_cart = res["mid_extended_plane_cart"]
    mid_out_cart = res["mid_outlier_plane_cart"]

    # projection & cell
    p0, p1 = projection
    cell = atoms.get_cell()
    shifts = [i * cell[p0] + j * cell[p1] for i in range(reps[0]) for j in range(reps[1])]

    fig, ax = plt.subplots(figsize=figsize)

    for shift in shifts:
        coords_e = ext_coords + shift if ext_coords.size else np.zeros((0,3))
        coords_s = sel_coords + shift
        xs_e, ys_e = (coords_e[:, p0], coords_e[:, p1]) if coords_e.size else ([], [])
        xs_s, ys_s = coords_s[:, p0], coords_s[:, p1]
        labs = np.array([labels_map[i] for i in sel_idx])
        mask_good = (labs != -1)
        mask_bad  = (labs == -1)

        if coords_e.size:
            ax.scatter(xs_e, ys_e,
                       s=15, color=extended_color, alpha=0.5,
                       label='Extended region' if shift is shifts[0] else "")
        ax.scatter(xs_s[mask_good], ys_s[mask_good],
                   s=50, color=non_outlier_color,
                   label='Selected (inliers)' if shift is shifts[0] else "")
        ax.scatter(xs_s[mask_bad], ys_s[mask_bad],
                   marker='x', s=80, color=outlier_color,
                   label='Selected (outliers)' if shift is shifts[0] else "")

    # draw mid planes for tiles
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]
        if axis == p0:
            x_ext = mid_ext_cart + sx
            x_out = mid_out_cart + sx
            ax.axvline(x=x_ext, linestyle=plane_linestyles[0], color=extended_color,
                       label='Mid extended plane' if shift is shifts[0] else "")
            ax.axvline(x=x_out, linestyle=plane_linestyles[1], color=outlier_color,
                       label='Mid outlier plane' if shift is shifts[0] else "")
        elif axis == p1:
            y_ext = mid_ext_cart + sy
            y_out = mid_out_cart + sy
            ax.axhline(y=y_ext, linestyle=plane_linestyles[0], color=extended_color,
                       label='Mid extended plane' if shift is shifts[0] else "")
            ax.axhline(y=y_out, linestyle=plane_linestyles[1], color=outlier_color,
                       label='Mid outlier plane' if shift is shifts[0] else "")

    ax.set_xlabel(f'Axis {p0} (Å)')
    ax.set_ylabel(f'Axis {p1} (Å)')
    ax.set_title(f'Tiled {reps[0]}×{reps[1]} (proj {p0}–{p1})')
    ax.set_aspect('equal')
    ax.autoscale()

    handles, labels = ax.get_legend_handles_labels()
    by_lbl = dict(zip(labels, handles))
    ax.legend(by_lbl.values(), by_lbl.keys(), loc='upper left')

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi)

    plt.show()
    return fig, ax

