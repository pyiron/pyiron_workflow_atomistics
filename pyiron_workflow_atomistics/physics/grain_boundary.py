"""Grain-boundary workflows: pure_gb_study, cleavage_study, segregation_study."""

import os

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
from pyiron_workflow import Workflow
from pyiron_workflow.api import for_node

from pyiron_workflow_atomistics._internal.engine_output import (
    extract_outputs_from_EngineOutputs,
)
from pyiron_workflow_atomistics.engine import Engine, calculate, subdir_path, subengine
from pyiron_workflow_atomistics.physics._grain_boundary_helpers.geometry import (
    axis_to_index,
)


@pwf.as_function_node
def get_extended_struct_list(structure, extensions=np.linspace(-0.2, 0.8, 11)):

    base_structure = structure.copy()
    extended_structure_list = []
    for ext in extensions:
        structure = base_structure.copy()
        a, b, c = structure.get_cell_lengths_and_angles()[:3]
        structure.set_cell([a, b, c + ext], scale_atoms=True)
        extended_structure_list.append(structure)
    return extended_structure_list, extensions


@pwf.as_function_node
def get_min_energy_structure_from_forloop_df(df):
    extracted_dict = extract_outputs_from_EngineOutputs(
        engine_outputs=df.engine_output,
        keys=["final_energy", "final_structure", "final_volume"],
    )
    if not extracted_dict["final_energy"]:
        raise ValueError("No converged runs.")
    energies = extracted_dict["final_energy"]
    i = int(np.argmin(energies))
    min_energy_structure = extracted_dict["final_structure"][i]
    min_energy = energies[i]
    return min_energy_structure, min_energy


@pwf.as_function_node("modified_structure")
def get_modified_cell_structure(structure, cell):
    modified_structure = structure.copy()
    modified_structure.set_cell(cell, scale_atoms=True)
    return modified_structure


@pwf.as_function_node()
def fit_polynomial_extremum(x_vals, y_vals, degree=2, num_points=None, extremum="min"):
    x = np.array(x_vals, float)
    y = np.array(y_vals, float)
    if degree < 2:
        raise ValueError("Degree must be >= 2")
    if num_points and num_points < len(y):
        idxs = np.argsort(y) if extremum == "min" else np.argsort(-y)
        idxs = idxs[:num_points]
        x, y = x[idxs], y[idxs]
    coeffs = np.polyfit(x, y, degree)
    roots = np.roots(np.polyder(coeffs))
    real_roots = roots[np.isreal(roots)].real
    second_derivative = np.polyder(coeffs, 2)
    candidates = [
        r
        for r in real_roots
        if (extremum == "min" and np.polyval(second_derivative, r) > 0)
        or (extremum == "max" and np.polyval(second_derivative, r) < 0)
    ]
    if not candidates:
        raise RuntimeError(f"No {extremum} found")
    vals = [(r, np.polyval(coeffs, r)) for r in candidates]
    ext_val = (
        min(vals, key=lambda t: t[1])
        if extremum == "min"
        else max(vals, key=lambda t: t[1])
    )
    return ext_val, coeffs


@pwf.as_function_node("interpolated_structure", "interpolated_energy")
def get_interp_min_energy_structure_from_forloop_df(
    df, axis="c", check_orthorhombic=False, tol=1e-6, degree=2, num_points=None
):
    extracted_dict = extract_outputs_from_EngineOutputs(
        engine_outputs=df.engine_output,
        keys=["final_energy", "final_structure", "final_volume"],
    )
    energies = extracted_dict["final_energy"]
    structs = extracted_dict["final_structure"]
    lengths = [
        np.linalg.norm(np.array(struct.cell)[axis_to_index(axis)]) for struct in structs
    ]
    (length_min, interpolated_energy), _ = fit_polynomial_extremum.node_function(
        lengths, energies, degree, num_points, extremum="min"
    )
    ref_idx = int(np.argmin(energies))
    ref_struct = structs[ref_idx]
    cell = np.array(ref_struct.cell)
    idx = dict(a=0, b=1, c=2)[axis]
    unit_vec = cell[idx] / np.linalg.norm(cell[idx])
    cell[idx] = unit_vec * length_min
    interpolated_structure = get_modified_cell_structure.node_function(ref_struct, cell)
    return interpolated_structure, interpolated_energy


@pwf.as_function_node("GB_energy")
def get_GB_energy(atoms, total_energy, e0_per_atom, gb_normal_axis="c"):
    idx = axis_to_index(gb_normal_axis)
    cell = np.array(atoms.get_cell())
    normals = [i for i in range(3) if i != idx]
    area = np.linalg.norm(np.cross(cell[normals[0]], cell[normals[1]]))
    deltaE = total_energy - (len(atoms) * e0_per_atom)
    gamma_GB = deltaE / (2 * area) * 16.021766208  # eV to J/m^2
    return gamma_GB


@pwf.as_function_node("excess_volume")
def get_GB_exc_volume(atoms, bulk_vol_per_atom, gb_normal_axis="c"):
    idx = axis_to_index(gb_normal_axis)
    cell = np.array(atoms.get_cell())
    normals = [i for i in range(3) if i != idx]
    area = np.linalg.norm(np.cross(cell[normals[0]], cell[normals[1]]))
    delta_vol = atoms.get_volume() - len(atoms) * bulk_vol_per_atom
    excess_volume = delta_vol / area / 2
    return excess_volume


@pwf.as_function_node("extended_dirnames")
def get_extended_names(extensions):
    extended_names = []
    for extension in extensions:
        extended_names.append(f"ext_{extension:.3f}")
    return extended_names


@pwf.as_function_node("engines")
def _make_engines_with_subdirs(engine: Engine, subdirnames: list) -> list:
    """Return a list of engines, one per subdir name."""
    return [engine.with_working_directory(name) for name in subdirnames]


@Workflow.wrap.as_macro_node(
    "extended_GB_results",
    "min_energy_GB_struct",
    "min_energy_GB_energy",
    "min_interp_energy_GB_struct",
    "min_interp_energy_GB_energy",
    "exc_volume",
    "gb_energy",
)
def gb_length_optimiser(
    wf,
    gb_structure,
    equil_bulk_volume,
    equil_bulk_energy,
    extensions,
    engine: Engine,
    gb_normal_axis: str = "c",
):
    # 1. Generate extended structures
    wf.extended_GBs = get_extended_struct_list(gb_structure, extensions=extensions)
    wf.extended_GBs_subdirnames = get_extended_names(extensions=extensions)
    wf.engines_per_calc = _make_engines_with_subdirs(
        engine=engine,
        subdirnames=wf.extended_GBs_subdirnames,
    )
    # 2. Compute energies/volumes for extended structures
    wf.extended_GBs_calcs = for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.extended_GBs.outputs.extended_structure_list,
        engine=wf.engines_per_calc,
    )

    # 4. Fit and extract minimum-energy structure
    wf.GB_min_energy_struct = get_min_energy_structure_from_forloop_df(
        wf.extended_GBs_calcs.outputs.df
    )
    # 5. Interpolate the min-energy GB from the datapoints
    wf.GB_min_energy_struct_interp = get_interp_min_energy_structure_from_forloop_df(
        wf.extended_GBs_calcs.outputs.df
    )
    # 6. Compute GB excess volume per area
    wf.exc_volume = get_GB_exc_volume(
        wf.GB_min_energy_struct_interp.outputs.interpolated_structure,
        equil_bulk_volume,
        gb_normal_axis=gb_normal_axis,
    )
    # 7. Compute GB energy per area
    wf.gb_energy = get_GB_energy(
        atoms=wf.GB_min_energy_struct_interp.outputs.interpolated_structure,
        total_energy=wf.GB_min_energy_struct_interp.outputs.interpolated_energy,
        e0_per_atom=equil_bulk_energy,
        gb_normal_axis=gb_normal_axis,
    )
    return (
        wf.extended_GBs_calcs,
        wf.GB_min_energy_struct.outputs.min_energy_structure,
        wf.GB_min_energy_struct.outputs.min_energy,
        wf.GB_min_energy_struct_interp.outputs.interpolated_structure,
        wf.GB_min_energy_struct_interp.outputs.interpolated_energy,
        wf.exc_volume,
        wf.gb_energy,
    )


@pwf.as_function_node
def get_concat_df(df_list):
    concat_df = pd.concat(df_list)
    return concat_df


from copy import deepcopy


@pwf.as_function_node("generic_output")
def generate_deepcopy(input_obj):
    return deepcopy(input_obj)


@pwf.as_function_node("length")
def get_length(extensions):
    return len(extensions)


@Workflow.wrap.as_macro_node(
    "stage1_opt_struct",
    "stage1_opt_excvol",
    "stage1_opt_GBEnergy",
    "stage2_opt_struct",
    "stage2_opt_excvol",
    "stage2_opt_GBEnergy",
    "stage1_plot",
    "stage2_plot",
    "results_df",
    "combined_plot",
    "gb_structure_final",
    "gb_structure_final_energy",
)
def full_gb_length_optimization(
    wf,
    gb_structure,
    equil_bulk_energy,
    equil_bulk_volume,
    extensions_stage1,
    extensions_stage2,
    engine: Engine,
    interpolate_min_n_points=5,
    gb_normal_axis="c",
):
    wf.stage1_engine = subengine(engine=engine, subdir="stage1")
    wf.stage2_engine = subengine(engine=engine, subdir="stage2")
    wf.stage1_path = subdir_path(engine=engine, subdir="stage1")
    wf.stage2_path = subdir_path(engine=engine, subdir="stage2")

    # 1. First length-scan + optimise
    wf.stage1_opt = gb_length_optimiser(
        gb_structure=gb_structure,
        engine=wf.stage1_engine,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions=extensions_stage1,
        gb_normal_axis=gb_normal_axis,
    )

    # 2. Second (refined) scan + optimise
    wf.stage2_opt = gb_length_optimiser(
        gb_structure=wf.stage1_opt.outputs.min_interp_energy_GB_struct,
        engine=wf.stage2_engine,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions=extensions_stage2,
        gb_normal_axis=gb_normal_axis,
    )
    wf.stage2_opt_struct_copy = generate_deepcopy(
        wf.stage2_opt.outputs.min_interp_energy_GB_struct
    )
    wf.stage1_plot_len = get_length(extensions_stage1)
    # 3. Plot each stage
    wf.stage1_plot = get_gb_length_optimiser_plot(
        df=wf.stage1_opt.outputs.extended_GB_results,
        n_points=wf.stage1_plot_len,
        working_directory=wf.stage1_path,
        save_filename="gb_optimiser_stage1.jpg",
    )
    wf.stage2_plot_len = get_length(extensions_stage2)
    wf.stage2_plot = get_gb_length_optimiser_plot(
        df=wf.stage2_opt.outputs.extended_GB_results,
        n_points=wf.stage2_plot_len,
        working_directory=wf.stage2_path,
        save_filename="gb_optimiser_stage2.jpg",
    )

    # 4. Concatenate results and re-plot combined
    wf.concat_results = pwf.api.inputs_to_list(
        2,
        wf.stage1_opt.outputs.extended_GB_results,
        wf.stage2_opt.outputs.extended_GB_results,
    )
    wf.concat_df = get_concat_df(wf.concat_results)

    wf.combined_plot = get_gb_length_optimiser_plot(
        df=wf.concat_df,
        n_points=interpolate_min_n_points,
        working_directory=engine.working_directory,
        save_filename="gb_optimiser_combined.jpg",
    )

    # 5. Return the key outputs
    return (
        wf.stage1_opt.outputs.min_interp_energy_GB_struct,
        wf.stage1_opt.outputs.exc_volume,
        wf.stage1_opt.outputs.gb_energy,
        wf.stage2_opt.outputs.min_interp_energy_GB_struct,
        wf.stage2_opt.outputs.exc_volume,
        wf.stage2_opt.outputs.gb_energy,
        wf.stage1_plot,
        wf.stage2_plot,
        wf.concat_df,
        wf.combined_plot,
        wf.stage2_opt_struct_copy,
        wf.stage2_opt.outputs.min_interp_energy_GB_energy,
    )


import matplotlib.pyplot as plt


@pwf.as_function_node
def get_gb_length_optimiser_plot(
    df,
    plot_label="run",
    degree=2,
    n_points=None,
    save_filename=None,
    dpi=300,
    figsize=(6, 4),
    working_directory=None,
):
    """
    Plot GB c-length vs energy for one optimisation run, fit a polynomial,
    annotate its minimum (for quadratics), and optionally save the figure.
    Can also limit to the n lowest-energy points.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have 'structure' and 'engine_output' columns.
    plot_label : str, optional
        Label for this run.
    degree : int, optional
        Degree of polynomial fit.
    n_points : int or None, optional
        If set, only the n lowest-energy samples are used for plotting and fitting.
    save_filename : str or None, optional
        File name to save the figure. If None, figure is not saved.
    dpi : int, optional
        Dots per inch when saving.
    figsize : tuple, optional
        Figure size in inches (width, height).
    working_directory : str or None, optional
        Directory to save the figure in (used with save_filename).
    """
    # Prepare data
    df_copy = df.copy()
    df_copy["c"] = df_copy.structure.apply(lambda x: x.cell[-1][-1])
    df_copy["energy"] = df_copy.engine_output.apply(lambda r: r.final_energy)

    # Optionally select only the n smallest energy points
    if isinstance(n_points, int) and n_points > 0:
        df_fit = df_copy.nsmallest(n_points, "energy")
    else:
        df_fit = df_copy

    x = df_fit["c"].to_numpy()
    y = df_fit["energy"].to_numpy()

    # Polynomial fit
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = poly(x_fit)

    # Compute minimum for quadratic
    v = None
    if degree == 2:
        v = -coeffs[1] / (2 * coeffs[0])

    # Plot with fixed figure size
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df_copy["c"], df_copy["energy"], alpha=0.3, label="all points")
    ax.scatter(x, y, label=f"{plot_label} (n={len(x)})")
    ax.plot(x_fit, y_fit, label=f"fit {plot_label}")

    if v is not None:
        ax.axvline(v, linestyle="--", label=f"min {plot_label}")
        # offset in data units: 5% of x-range
        x_range = x.max() - x.min()
        offset_data = 0.05 * x_range
        ax.text(
            v + offset_data,
            0.95,
            f"{v:.3f}",
            rotation=90,
            va="top",
            ha="center",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel("c (Å)")
    ax.set_ylabel("energy (eV)")
    ax.legend()
    plt.tight_layout()

    # Save if requested
    if save_filename:
        fig.savefig(os.path.join(working_directory, save_filename), dpi=dpi)

    return fig


from ase import Atoms
from pyiron_snippets.logger import logger

from pyiron_workflow_atomistics.analysis.gb_plane import get_sites_on_plane


# Wrap‐aware difference in fractional space:
def _frac_dist(a, b):
    return abs(((a - b + 0.5) % 1.0) - 0.5)


@pwf.as_function_node
def find_viable_cleavage_planes_around_plane(
    structure: Atoms,
    axis: str,
    plane_coord: float | int,
    coord_tol: float | int,
    layer_tolerance: float = 1e-3,
    fractional: bool = False,
) -> list:
    # Convert axis string ("a"/"b"/"c") to numeric index
    ax = axis_to_index(axis)

    # 1) Get cell lengths along each axis
    cell = structure.get_cell()
    cell_lengths = np.linalg.norm(cell, axis=1)

    # 2) Gather coordinates along the axis (Cartesian or fractional)
    if fractional:
        all_coords = structure.get_scaled_positions(wrap=False)[:, ax] % 1.0
        tol = layer_tolerance / cell_lengths[ax]
        target = plane_coord % 1.0
        tol_plane = coord_tol / cell_lengths[ax]
        logger.info(f"Target fractional plane: {target} ± {tol_plane}")
    else:
        all_coords = structure.get_positions(wrap=False)[:, ax]
        tol = layer_tolerance
        target = plane_coord
        tol_plane = coord_tol
        logger.info(f"Target Cartesian plane: {target} Å ± {tol_plane} Å")

    # 3) Identify unique “layers” by merging coords within `tol`
    sorted_coords = np.sort(all_coords)
    unique_layers = []
    for v in sorted_coords:
        if not any(abs(v - u) < tol for u in unique_layers):
            unique_layers.append(v)
    logger.info(f"Unique layer positions: {unique_layers}")

    # 4) Compute midpoints between each pair of adjacent layers
    candidate_planes = [
        (unique_layers[i] + unique_layers[i + 1]) / 2.0
        for i in range(len(unique_layers) - 1)
    ]
    logger.info(f"Candidate cleavage midpoints: {candidate_planes}")

    # 5) Filter midpoints that lie within ±tol_plane of `target`
    if fractional:
        viable_planes = [
            cp for cp in candidate_planes if _frac_dist(cp, target) <= tol_plane
        ]
    else:
        viable_planes = [
            cp
            for cp in candidate_planes
            if (target - tol_plane) <= cp <= (target + tol_plane)
        ]

    return viable_planes


@pwf.as_function_node
def find_viable_cleavage_planes_around_site(
    structure: Atoms,
    axis: str,
    site_index: int,
    site_dist_threshold: float,
    layer_tolerance: float = 1e-3,
    fractional: bool = False,
) -> list:
    """
    Identify viable cleavage‐plane positions along a given axis around a specified site.

    Scans all atomic coordinates (either Cartesian or fractional)
    to find “layer” positions. It computes midpoints between adjacent layers,
    and filters those midpoints that lie within ±site_dist_threshold of the
    chosen site’s coordinate (within a small tolerance).

    Parameters
    ----------
    structure : ase.Atoms
        The ASE Atoms object to analyze.
    axis : str
        Which axis to cleave along: "a", "b", or "c".
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

    Logs
    ----
    - The min/max limits for filtering (min_lim and max_lim).
    - The site’s coordinate along `axis` (fractional or Å).
    - The list of unique layer positions.
    - The list of all candidate cleavage‐plane midpoints.
    - The final viable cleavage positions.
    """
    # Convert axis string ("a"/"b"/"c") to numeric index
    ax = axis_to_index(axis)

    # 2) Fetch cell and determine length along the chosen axis
    cell = structure.get_cell()
    cell_lengths = cell.diagonal()  # length along x, y, z

    # 3) Gather coordinates along the axis (Cartesian or fractional)
    if fractional:
        all_coords = structure.get_scaled_positions(wrap=False)[:, ax]
        site_coord = all_coords[site_index]
        tol = layer_tolerance / cell_lengths[ax]
        threshold_frac = site_dist_threshold / cell_lengths[ax]
        min_lim = site_coord - threshold_frac
        max_lim = site_coord + threshold_frac
    else:
        all_coords = structure.get_positions(wrap=False)[:, ax]
        site_coord = all_coords[site_index]
        tol = layer_tolerance
        min_lim = site_coord - site_dist_threshold
        max_lim = site_coord + site_dist_threshold

    logger.info(f"{min_lim} {max_lim}")
    logger.info(
        f"Site coordinate along axis {ax}: {site_coord} ({'fractional' if fractional else 'Å'})"
    )

    # 4) Identify unique “layers” by merging coordinates within `tol`
    sorted_coords = np.sort(all_coords)
    unique_layers = []
    for v in sorted_coords:
        if not any(abs(v - u) < tol for u in unique_layers):
            unique_layers.append(v)
    logger.info(f"Unique layer positions: {unique_layers}")

    # 5) Compute midpoints between each pair of adjacent layers
    candidate_planes = [
        (unique_layers[i] + unique_layers[i + 1]) / 2.0
        for i in range(len(unique_layers) - 1)
    ]
    logger.info(f"Candidate cleavage positions: {candidate_planes}")

    # 6) Filter midpoints to those within ± site_dist_threshold of the site
    viable_planes = [cp for cp in candidate_planes if (min_lim <= cp <= max_lim)]
    logger.info(f"Viable cleavage positions: {viable_planes}")

    return viable_planes


@pwf.as_function_node
def cleave_axis_aligned(
    structure: Atoms,
    axis: str,
    plane_coord: float | int,
    separation: float,
    use_fractional: bool = False,
) -> Atoms:
    """
    Cleave an ASE Atoms object by an axis‐aligned plane and move the two halves apart.
    You can specify the plane in Cartesian (Å) or fractional (0–1) coordinates.

    Parameters
    ----------
    structure : ase.Atoms
        The original system to be cleaved.
    axis : {'a','b','c'}
        The axis along which to cleave:
          - 'a' → plane normal = [1, 0, 0]
          - 'b' → plane normal = [0, 1, 0]
          - 'c' → plane normal = [0, 0, 1]
        (These correspond to the crystallographic axes a, b, c, which map to x, y, z in Cartesian.)
    plane_coord : float | int
        The coordinate along the chosen axis where the cleavage plane lies.
        If use_fractional=False, this is a Cartesian coordinate in Å (e.g. a = 3.2 Å).
        If use_fractional=True, this is a fractional coordinate (0 ≤ plane_coord < 1).
    separation : float
        The total distance (in Å) by which the two halves should be separated
        along the chosen axis. Atoms on the “+” side move by +separation/2,
        atoms on the “–” side move by –separation/2 along that axis.
    use_fractional : bool, optional (default=False)
        If False, compare each atom’s Cartesian coordinate to plane_coord (in Å).
        If True, compare each atom’s fractional (scaled) coordinate to plane_coord
        (in 0–1) to decide which side of the plane it lies on.

    Returns
    -------
    new_structure : ase.Atoms
        A deep‐copied Atoms object where:
          - any atom with coord(axis) ≥ plane_coord (cart or frac) has been shifted
            by +separation/2 along that axis (in Å),
          - any atom with coord(axis) <  plane_coord (cart or frac) has been shifted
            by –separation/2 along that axis (in Å).

    Notes
    -----
    1. Comparing in fractional mode does NOT modify cell vectors: it only
       uses scaled positions to classify atoms. The actual displacement is
       always in Cartesian (Å) by ±separation/2.
    2. Atoms exactly at coord == plane_coord (within floating‐point tolerance)
       are treated as “≥” → on the positive side.
    3. This function does NOT modify the cell geometry or PBC flags. If you
       need periodicity, you must enlarge the cell or turn off PBC manually.
    """
    # 1) Copy so original structure isn’t modified
    new_structure = structure.copy()

    # 2) Convert axis string ("a"/"b"/"c") to numeric index (0, 1, or 2)
    ax = axis_to_index(axis)

    # 3) Fetch positions or fractional coords along that axis
    if use_fractional:
        coords = new_structure.get_scaled_positions(wrap=False)[:, ax] % 1.0
        target = plane_coord % 1.0
        mask_positive = coords >= target
        mask_negative = coords < target
    else:
        coords = new_structure.get_positions(wrap=False)[:, ax]
        target = plane_coord
        mask_positive = coords >= target
        mask_negative = coords < target

    # 4) Determine shift distances (in Å)
    delta_pos = 0.5 * separation
    delta_neg = -0.5 * separation

    # 5) Build displacement array (N×3) all zeros except on the chosen axis
    positions = new_structure.get_positions(wrap=False)
    displacements = np.zeros_like(positions)
    displacements[mask_positive, ax] = delta_pos
    displacements[mask_negative, ax] = delta_neg

    # 6) Apply translations
    new_positions = positions + displacements
    new_structure.set_positions(new_positions)

    return new_structure


@pwf.as_function_node
def plot_structure_with_cleavage(
    structure: Atoms,
    cleavage_planes: list[float],
    projection=(0, 2),
    reps=(1, 1),
    figsize=(8, 6),
    atom_color="C0",
    plane_color="r",
    plane_linestyle="--",
    atom_size=30,
    save_path=None,
    dpi=300,
    show_fractional_axes: bool = True,
    ylims=None,
):
    """
    Plot a 2D projection of `structure` with cleavage planes overlaid as lines,
    and optionally add secondary axes showing fractional coordinates.

    Parameters
    ----------
    structure : ASE Atoms
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

    # Unpack projection
    p0, p1 = projection
    cell = structure.get_cell()

    # Compute tiling shifts (in Cartesian)
    shifts = [
        i * cell[p0] + j * cell[p1] for i in range(reps[0]) for j in range(reps[1])
    ]

    # Extract atomic positions
    pos = structure.get_positions()
    xs = pos[:, p0]
    ys = pos[:, p1]

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Plot atoms for each tile
    for shift in shifts:
        sx, sy = shift[p0], shift[p1]
        ax.scatter(
            xs + sx,
            ys + sy,
            s=atom_size,
            color=atom_color,
            label="Atoms" if shift is shifts[0] else None,
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
                xmin=x_min,
                xmax=x_max,
                colors=plane_color,
                linestyles=plane_linestyle,
                label=(
                    "Cleavage plane"
                    if (plane == cleavage_planes[0] and shift is shifts[0])
                    else None
                ),
            )

    # 3) Labels and aesthetics
    ax.set_xlabel(f"Axis {p0} (Å)")
    ax.set_ylabel(f"Axis {p1} (Å)")
    ax.set_title(f"2D Projection with Cleavage Planes (proj {p0}-{p1})")
    ax.set_aspect("equal")

    # 4) Apply user-specified y-limits if provided
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.autoscale()

    # 5) Add secondary fractional axes if requested
    if show_fractional_axes:
        # Secondary Y-axis (right): fractional along p1
        cell_len_p1 = np.linalg.norm(cell[p1])
        secax_y = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y: y / cell_len_p1,  # cart → frac
                lambda f: f * cell_len_p1,
            ),  # frac → cart
        )
        secax_y.set_ylabel(f"Axis {p1} (fractional)")

        # Secondary X-axis (top): fractional along p0
        cell_len_p0 = np.linalg.norm(cell[p0])
        secax_x = ax.secondary_xaxis(
            "top",
            functions=(
                lambda x: x / cell_len_p0,  # cart → frac
                lambda f: f * cell_len_p0,
            ),  # frac → cart
        )
        secax_x.set_xlabel(f"Axis {p0} (fractional)")

    # 6) Legend outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,
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
    base_structure: Atoms,
    axis_to_cleave: str,
    target_coord,
    tol=0.3,
    cleave_region_halflength=5.0,
    layer_tolerance=0.3,
    separation=8.0,
    use_fractional=False,
):
    """
    Find and cleave a grain‐boundary‐type slab (base_structure) into multiple
    “cleaved” pieces along all viable GB planes found near a given target_coord.

    Parameters
    ----------
    base_structure : ase.Atoms
        The input slab/Supercell (e.g. Fe+GB+vacuum) that you want to cleave.
    axis_to_cleave : str
        The crystallographic axis letter along which to cleave: "a", "b", or "c".
    target_coord : array‐like of length 3
        The Cartesian (x,y,z) coordinate of the GB plane (e.g. from gb_plane_extractor).
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
    cleavage_plane_coords : list of float
        The coordinates of each viable cleavage plane found.
    """
    # print("trying to find axis_to_index")
    # Convert axis letter ("a"/"b"/"c") to numeric index
    ax = axis_to_index(axis_to_cleave)
    # print("trying to find get_sites_on_plane")
    # 2) Identify which atom index sits “on/near” the GB plane.
    mid_site_indices = get_sites_on_plane.node_function(
        atoms=base_structure,
        axis=axis_to_cleave,
        target_coord=target_coord,
        tol=tol,
        use_fractional=use_fractional,
    )
    # print("succeeded")
    # if len(mid_site_indices) == 0:
    #     raise RuntimeError(
    #         f"No atoms found within tol={tol} of {axis_to_cleave}={target_coord}."
    #     )
    # print("trying to find mid_site_idx")
    if len(mid_site_indices) == 0:
        # fallback: compute distance along the chosen axis for every atom
        # print("fallback: computing distance along the chosen axis for every atom")
        positions = np.array(base_structure.get_positions())
        # print("positions", positions)
        # print("ax", ax)
        distances = np.abs(positions[:, ax] - target_coord)
        # print("distances", distances)
        mid_site_idx = int(np.argmin(distances))
        # print("mid_site_idx", mid_site_idx)
    else:
        # print("using mid_site_indices")
        mid_site_idx = int(mid_site_indices[0])
    # print("finished finding mid_site_idx")
    # print("trying to find viable cleavage planes")
    # 3) Find all viable cleavage planes around that site index.
    cleavage_plane_coords = find_viable_cleavage_planes_around_site.node_function(
        structure=base_structure,
        axis=ax,
        site_index=mid_site_idx,
        site_dist_threshold=cleave_region_halflength,
        layer_tolerance=layer_tolerance,
        fractional=use_fractional,
    )
    # print("finished finding viable cleavage planes")
    # print("trying to cleave structure")
    # 4) For each plane coord, call cleave_axis_aligned to get a “cleaved” slab.
    cleaved_structures = []
    for plane_c in cleavage_plane_coords:
        slab_structure = cleave_axis_aligned.node_function(
            structure=base_structure,
            axis=ax,
            plane_coord=plane_c,
            separation=separation,
            use_fractional=use_fractional,
        )
        cleaved_structures.append(slab_structure)
    # print("finished cleaving structure")
    return cleaved_structures, cleavage_plane_coords


@pwf.as_function_node
def get_cleavage_calc_names(parent_dir, cleavage_planes):
    folder_name_list = []
    for plane in cleavage_planes:
        calc_foldername = f"{os.path.basename(parent_dir)}_cp_{np.round(plane, 3)}"
        folder_name_list.append(os.path.join(parent_dir, calc_foldername))
    return folder_name_list


@pwf.as_function_node("df")
def get_results_df(
    df, cleavage_coords, cleaved_structures, uncleaved_energy, cleavage_axis: str = "c"
):
    from pyiron_workflow_atomistics._internal.engine_output import (
        extract_outputs_from_EngineOutputs,
    )

    extracted_dict = extract_outputs_from_EngineOutputs(
        engine_outputs=df.engine_output,
        keys=[
            "final_energy",
            "final_structure",
            "final_volume",
            "final_forces",
            "final_stress",
        ],
    )
    relaxed_structures = extracted_dict["final_structure"]
    energies = extracted_dict["final_energy"]

    axis_index = axis_to_index(cleavage_axis)

    cleavage_energies = []

    for E, struct in zip(energies, relaxed_structures):
        cell = struct.get_cell()
        # Get the 2 vectors that span the cleavage plane perpendicular to the cleavage axis
        a1, a2 = np.delete(cell, axis_index, axis=0)
        area = np.linalg.norm(np.cross(a1, a2))  # in Å²
        # print(area, struct.cell[-2][-2] *struct.cell[0][0])
        # Cleavage energy in J/m²
        E_cleave = (
            (E - uncleaved_energy) / (area) * 16.0218
        )  # eV/Å² → J/m² # Only 1 GB (for vacuum cells - which is as we do it here) so no 2 factor on bottom
        cleavage_energies.append(E_cleave)
    return pd.DataFrame(
        {
            "cleavage_coord": cleavage_coords,
            "initial_structure": cleaved_structures,
            "final_structure": relaxed_structures,
            "energy": energies,
            "cleavage_energy": cleavage_energies,
        }
    )


from pyiron_workflow_atomistics.physics._grain_boundary_helpers.dataclass_storage import (
    CleaveGBStructureInput,
    PlotCleaveInput,
)


@pwf.as_macro_node(
    "cleaved_structure_list",
    "cleaved_plane_coords_list",
    "cleavage_plane_plot_fig",
    "cleavage_plane_plot_ax",
    "cleavage_calcs_df",
)
def calc_cleavage_GB(
    wf,
    structure: Atoms,
    energy,
    input_cleave_gb_structure: CleaveGBStructureInput,
    input_plot_cleave: PlotCleaveInput,
    engine: Engine,
):
    wf.cleave_setup = cleave_gb_structure(
        base_structure=structure,
        axis_to_cleave=input_cleave_gb_structure.axis_to_cleave,
        target_coord=input_cleave_gb_structure.cleavage_target_coord,
        tol=input_cleave_gb_structure.tol,
        cleave_region_halflength=input_cleave_gb_structure.cleave_region_halflength,
        layer_tolerance=input_cleave_gb_structure.layer_tolerance,
        separation=input_cleave_gb_structure.separation,
        use_fractional=input_cleave_gb_structure.use_fractional,
    )
    wf.cleavage_structure_plot = plot_structure_with_cleavage(
        structure=structure,
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
    wf.cleave_structure_foldernames = get_cleavage_calc_names(
        parent_dir=engine.working_directory,
        cleavage_planes=wf.cleave_setup.outputs.cleavage_plane_coords,
    )
    wf.engines_per_plane = _make_engines_with_subdirs(
        engine=engine,
        subdirnames=wf.cleave_structure_foldernames,
    )
    wf.calculate_cleaved = pwf.api.for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.cleave_setup.outputs.cleaved_structures,
        engine=wf.engines_per_plane,
    )
    wf.collate_results = get_results_df(
        df=wf.calculate_cleaved.outputs.df,
        cleavage_coords=wf.cleave_setup.outputs.cleavage_plane_coords,
        cleaved_structures=wf.cleave_setup.outputs.cleaved_structures,
        uncleaved_energy=energy,
        cleavage_axis=input_cleave_gb_structure.axis_to_cleave,
    )

    return (
        wf.cleave_setup.outputs.cleaved_structures,
        wf.cleave_setup.outputs.cleavage_plane_coords,
        wf.cleavage_structure_plot.outputs.fig,
        wf.cleavage_structure_plot.outputs.ax,
        wf.collate_results.outputs.df,
    )


@pwf.api.as_macro_node("cleavage_results_rigid", "cleavage_results_relax")
def rigid_and_relaxed_cleavage_study(
    wf,
    gb_structure,
    gb_structure_energy,
    gb_plane_cart_loc,
    engine: Engine,
    static_engine: Engine,
    CleaveGBStructure_Input=None,
    PlotCleave_Input=None,
):
    from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dataclass

    wf.CleaveGBStructureInput = modify_dataclass(
        CleaveGBStructure_Input, "cleavage_target_coord", gb_plane_cart_loc
    )
    wf.rigid_engine = subengine(engine=static_engine, subdir="cleavage_rigid")
    wf.relax_engine = subengine(engine=engine, subdir="cleavage_relax")
    wf.calc_cleavage_rigid = calc_cleavage_GB(
        structure=gb_structure,
        energy=gb_structure_energy,
        engine=wf.rigid_engine,
        input_cleave_gb_structure=wf.CleaveGBStructureInput,
        input_plot_cleave=PlotCleave_Input,
    )
    wf.calc_cleavage_relax = calc_cleavage_GB(
        structure=gb_structure,
        energy=gb_structure_energy,
        engine=wf.relax_engine,
        input_cleave_gb_structure=wf.CleaveGBStructureInput,
        input_plot_cleave=PlotCleave_Input,
    )
    return (
        wf.calc_cleavage_rigid.outputs.cleavage_calcs_df,
        wf.calc_cleavage_relax.outputs.cleavage_calcs_df,
    )


@pwf.as_function_node("structure", "output_dir")
def create_seg_structure_and_output_dir(
    structure: Atoms,
    defect_site: int,
    element: str,
    structure_basename: str,
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
):
    # print("In create_seg_structure_and_output_dir")
    seg_structure = structure.copy()
    seg_structure[defect_site].symbol = element
    structure_name = f"{structure_basename}_{element}_{defect_site}"
    output_dir = os.path.join(parent_dir, structure_name)
    # print("Exiting create_seg_structure_and_output_dir")
    return seg_structure, output_dir


@pwf.as_function_node
def get_df_col_as_list(df, col):
    # print("In get_df_col_as_list")
    output_list = df[col].to_list()
    return output_list


@pwf.as_function_node("engines")
def _make_engines_from_dirs(engine: Engine, output_dirs: list) -> list:
    """Return a list of engines, one per output directory path.

    On POSIX, os.path.join(wd, absolute_path) == absolute_path, so passing
    absolute paths to with_working_directory is safe and sets the directory
    correctly.
    """
    return [engine.with_working_directory(d) for d in output_dirs]


@pwf.as_function_node("df")
def write_df(df, unique_sites_df, file_name, parent_dir):
    df_out = pd.concat([unique_sites_df, df], axis=1)
    df_out.to_pickle(os.path.join(parent_dir, file_name))
    return df_out


@pwf.as_function_node("unique_sites_list", "df")
def get_unique_sites_SOAP(
    structure: Atoms,
    defect_sites: list[int],
    r_cut: float = 6.0,
    n_max: int = 10,
    l_max: int = 10,
    n_jobs: int = -1,
    periodic: bool = True,
    pca_zca_model: dict | None = None,
    pca_variance_threshold: float = 0.999,
    similarity_threshold: float = 0.99999,
):
    from pyiron_workflow_atomistics.analysis.featurisers import (
        pca_whiten,
        soap_site_featuriser,
        summarize_cosine_groups,
    )

    a = soap_site_featuriser(
        atoms=structure,
        site_indices=defect_sites,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        n_jobs=n_jobs,
        periodic=periodic,
    )
    Z, model = pca_whiten(
        X=a, n_components=pca_variance_threshold, method="pca", model=pca_zca_model
    )
    df = summarize_cosine_groups(
        Z, threshold=similarity_threshold, ids=defect_sites, include_singletons=True
    )
    return df.rep.tolist(), df


@pwf.as_macro_node("gb_seg_calcs_df")
def calculate_substitutional_segregation_GB(
    wf,
    structure: Atoms,
    defect_sites: list[int],
    element: str,
    structure_basename: str,
    engine: Engine,
    unique_sites_df: pd.DataFrame | None = None,
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
    df_filename: str = "seg_calcs_df.pkl",
):
    wf.gb_seg_structure_generator = for_node(
        create_seg_structure_and_output_dir,
        structure=structure,
        iter_on=("defect_site"),
        defect_site=defect_sites,
        structure_basename=structure_basename,
        element=element,
        parent_dir=parent_dir,
    )
    wf.gb_seg_structure_list = get_df_col_as_list(
        wf.gb_seg_structure_generator.outputs.df, "structure"
    )
    wf.gb_seg_structure_dirs = get_df_col_as_list(
        wf.gb_seg_structure_generator.outputs.df, "output_dir"
    )
    wf.gb_seg_engines = _make_engines_from_dirs(
        engine=engine,
        output_dirs=wf.gb_seg_structure_dirs,
    )
    wf.gb_seg_calcs = for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.gb_seg_structure_list,
        engine=wf.gb_seg_engines,
    )
    wf.gb_seg_calcs_df = write_df(
        df=wf.gb_seg_calcs.outputs.df,
        unique_sites_df=unique_sites_df,
        file_name=df_filename,
        parent_dir=parent_dir,
    )
    return wf.gb_seg_calcs_df.outputs.df


from pyiron_workflow_atomistics.analysis.featurisers import voronoi_site_featuriser


@pwf.api.as_function_node("surface_energy")
def _get_surface_energy(total_energy_gb_vac, total_energy_gb_novac, area):
    surface_energy = (
        (total_energy_gb_vac - total_energy_gb_novac) / area * 16.021766208 / 2
    )
    return surface_energy


@pwf.api.as_function_node("area")
def _get_area(gb_with_vacuum_rel, axis="c"):
    from pyiron_workflow_atomistics.physics._grain_boundary_helpers.geometry import (
        axis_to_index,
    )

    axis = axis_to_index(axis)
    area = gb_with_vacuum_rel.cell.volume / gb_with_vacuum_rel.cell[axis][axis]
    return area


@pwf.api.as_macro_node(
    "final_pure_grain_boundary_structure",
    "final_pure_grain_boundary_structure_energy",
    "grain_boundary_length_optimisation_df",
    "grain_boundary_energy",
    "grain_boundary_excess_volume",
    "surface_energy",
    "pure_grain_boundary_structure_vacuum",
    "pure_grain_boundary_structure_vacuum_energy",
    "gb_plane_analysis_dict",
    "work_of_separation_rigid",
    "work_of_separation_rigid_df",
    "work_of_separation_relaxed",
    "work_of_separation_relaxed_df",
)
def pure_gb_study(
    wf,
    gb_structure,
    equil_bulk_volume,
    equil_bulk_energy,
    extensions_stage1,
    extensions_stage2,
    engine: Engine,
    static_engine: Engine,
    length_interpolate_min_n_points=5,
    gb_normal_axis="c",
    vacuum_length=20,
    min_inplane_cell_lengths=None,
    featuriser=voronoi_site_featuriser,
    approx_frac=0.5,
    tolerance=5.0,
    bulk_offset=10.0,
    slab_thickness=2.0,
    featuriser_kwargs=None,
    n_bulk=10,
    threshold_frac=0.3,
    CleaveGBStructure_Input=None,
    PlotCleave_Input=None,
):
    if min_inplane_cell_lengths is None:
        min_inplane_cell_lengths = [6, 6, None]
    wf.length_engine = subengine(engine=engine, subdir="gb_length_optimiser")
    wf.gb_vacuum_engine = subengine(engine=engine, subdir="gb_with_vacuum_rel")
    wf.gb_seg_engine = subengine(engine=engine, subdir="gb_seg_supercell")
    wf.cleavage_engine = subengine(engine=engine, subdir="cleavage_study")
    wf.cleavage_static_engine = subengine(engine=static_engine, subdir="cleavage_study")

    wf.gb_length_optimiser = full_gb_length_optimization(
        gb_structure=gb_structure,
        engine=wf.length_engine,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions_stage1=extensions_stage1,
        extensions_stage2=extensions_stage2,
        interpolate_min_n_points=length_interpolate_min_n_points,
        gb_normal_axis=gb_normal_axis,
    )
    from pyiron_workflow_atomistics.structure.transform import add_vacuum

    wf.gb_with_vacuum = add_vacuum(
        wf.gb_length_optimiser.outputs.gb_structure_final,
        vacuum_length=vacuum_length,
        axis=gb_normal_axis,
    )

    wf.gb_with_vacuum_rel = calculate(
        structure=wf.gb_with_vacuum,
        engine=wf.gb_vacuum_engine,
        label="gb_with_vacuum_rel_run",
    )
    from pyiron_workflow_atomistics.structure.transform import (
        create_supercell_with_min_dimensions,
    )

    wf.gb_seg_supercell = create_supercell_with_min_dimensions(
        wf.gb_with_vacuum_rel.outputs.engine_output.final_structure,
        min_dimensions=min_inplane_cell_lengths,
    )

    wf.gb_seg_supercell_rel = calculate(
        structure=wf.gb_seg_supercell,
        engine=wf.gb_seg_engine,
        label="gb_seg_supercell_rel_run",
    )
    wf.area = _get_area(
        wf.gb_with_vacuum_rel.outputs.engine_output.final_structure, gb_normal_axis
    )
    wf.surface_energy = _get_surface_energy(
        wf.gb_with_vacuum_rel.outputs.engine_output.final_energy,
        wf.gb_length_optimiser.outputs.gb_structure_final_energy,
        wf.area,
    )
    from pyiron_workflow_atomistics.analysis.gb_plane import (
        find_gb_plane,
        plot_gb_plane,
    )

    wf.gb_plane_extractor = find_gb_plane(
        atoms=wf.gb_with_vacuum_rel.outputs.engine_output.final_structure,
        featuriser=featuriser,
        axis=gb_normal_axis,
        approx_frac=approx_frac,
        tolerance=tolerance,
        bulk_offset=bulk_offset,
        slab_thickness=slab_thickness,
        featuriser_kwargs=featuriser_kwargs,
        n_bulk=n_bulk,
        threshold_frac=threshold_frac,
    )
    wf.gb_plane_extractor_plot = plot_gb_plane(
        atoms=wf.gb_with_vacuum_rel.outputs.engine_output.final_structure,
        res=wf.gb_plane_extractor.outputs.gb_plane_analysis_dict,
        projection=(0, 2),
        reps=(5, 1),
        figsize=(10, 6),
        bulk_color="C0",
        window_cmap="viridis",
        plane_linestyles=("--", "-"),
        axis=2,
        dpi=300,
        working_directory=engine.working_directory,
        save_filename="pureGB_plane_identifier.jpg",
    )
    from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dataclass

    wf.CleaveGBStructureInput = modify_dataclass(
        CleaveGBStructure_Input,
        "cleavage_target_coord",
        wf.gb_plane_extractor.outputs.gb_plane_analysis_dict["gb_cart"],
    )

    wf.cleavage_study = rigid_and_relaxed_cleavage_study(
        gb_structure=wf.gb_with_vacuum_rel.outputs.engine_output.final_structure,
        gb_structure_energy=wf.gb_with_vacuum_rel.outputs.engine_output.final_energy,
        gb_plane_cart_loc=wf.gb_plane_extractor.outputs.gb_plane_analysis_dict[
            "gb_cart"
        ],
        engine=wf.cleavage_engine,
        static_engine=wf.cleavage_static_engine,
        CleaveGBStructure_Input=wf.CleaveGBStructureInput,
        PlotCleave_Input=PlotCleave_Input,
    )
    wf.min_rigid_cleavage_energy = get_min_energy_from_cleavage_study(
        wf.cleavage_study.outputs.cleavage_results_rigid
    )
    wf.min_relaxed_cleavage_energy = get_min_energy_from_cleavage_study(
        wf.cleavage_study.outputs.cleavage_results_relax
    )
    return (
        wf.gb_length_optimiser.outputs.gb_structure_final,
        wf.gb_length_optimiser.outputs.gb_structure_final_energy,
        wf.gb_length_optimiser.outputs.results_df,
        wf.gb_length_optimiser.outputs.stage2_opt_GBEnergy,
        wf.gb_length_optimiser.outputs.stage2_opt_excvol,
        wf.surface_energy,
        wf.gb_with_vacuum_rel.outputs.engine_output.final_structure,
        wf.gb_with_vacuum_rel.outputs.engine_output.final_energy,
        wf.gb_plane_extractor.outputs.gb_plane_analysis_dict,
        wf.min_rigid_cleavage_energy,
        wf.cleavage_study.outputs.cleavage_results_rigid,
        wf.min_relaxed_cleavage_energy,
        wf.cleavage_study.outputs.cleavage_results_relax,
    )


@pwf.api.as_function_node("min_energy")
def get_min_energy_from_cleavage_study(cleavage_study_df):
    min_energy = cleavage_study_df.cleavage_energy.min()
    return min_energy
