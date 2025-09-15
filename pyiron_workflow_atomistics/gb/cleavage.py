import os
import numpy as np
import pandas as pd
from ase import Atoms
import matplotlib.pyplot as plt

import pyiron_workflow as pwf

from pyiron_workflow_atomistics.gb.analysis import get_sites_on_plane

from pyiron_workflow_atomistics.calculator import (
    fillin_default_calckwargs,
    calculate_structure_node,
)
from pyiron_workflow_atomistics.gb.utils import axis_to_index

from pyiron_snippets.logger import logger


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
    import matplotlib.pyplot as plt

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
        calc_foldername = f"{os.path.basename(parent_dir)}_cp_{np.round(plane,3)}"
        folder_name_list.append(os.path.join(parent_dir, calc_foldername))
    return folder_name_list


@pwf.as_function_node("df")
def get_results_df(
    df, cleavage_coords, cleaved_structures, uncleaved_energy, cleavage_axis: str = "c"
):
    from pyiron_workflow_atomistics.utils import extract_outputs_from_EngineOutputs
    extracted_dict = extract_outputs_from_EngineOutputs(
        engine_outputs=df.calc_output,
        keys=["final_energy", "final_structure", "final_volume", "final_forces", "final_stress"]
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

from pyiron_workflow_atomistics.gb.dataclass_storage import (
    CleaveGBStructureInput,
    PlotCleaveInput,
)
from typing import Callable, Any
from pyiron_workflow_atomistics.calculator import generate_kwargs_variants
from pyiron_workflow_atomistics.dataclass_storage import Engine
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
    calculation_engine: Engine | None = None,
    calc_structure_fn: Callable[..., Any] | None = None,
    calc_structure_fn_kwargs: dict[str, Any] | None = None,
    # parent_dir: str = "gb_cleavage",
):
    from pyiron_workflow_atomistics.calculator import validate_calculation_inputs   
    wf.validate = validate_calculation_inputs(calculation_engine = calculation_engine,
                                              calc_structure_fn = calc_structure_fn,
                                              calc_structure_fn_kwargs = calc_structure_fn_kwargs)
    from pyiron_workflow_atomistics.utils import get_calc_fn_calc_fn_kwargs_from_calculation_engine
    wf.calc_structure_fn_kwargs_cleavage_calc = get_calc_fn_calc_fn_kwargs_from_calculation_engine(calculation_engine = calculation_engine,
                                                                                   structure = structure,
                                                                                   calc_structure_fn = calc_structure_fn,
                                                                                   calc_structure_fn_kwargs = calc_structure_fn_kwargs)
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
        parent_dir=wf.calc_structure_fn_kwargs_cleavage_calc.outputs.calc_fn_kwargs["working_directory"],
        cleavage_planes=wf.cleave_setup.outputs.cleavage_plane_coords,
    )
    wf.kwargs_removed_working_directory = fillin_default_calckwargs(
        calc_kwargs=wf.calc_structure_fn_kwargs_cleavage_calc.outputs.calc_fn_kwargs, default_values=None, remove_keys=["working_directory"]
    )
    wf.cleavage_calcs_kwargs = generate_kwargs_variants(
        base_kwargs=wf.kwargs_removed_working_directory.outputs.full_calc_kwargs2,
        key="working_directory",
        values=wf.cleave_structure_foldernames,
    )
    wf.calculate_cleaved = pwf.api.for_node(
        calculate_structure_node,
        zip_on=("structure", "_calc_structure_fn_kwargs"),
        structure=wf.cleave_setup.outputs.cleaved_structures,
        _calc_structure_fn=wf.calc_structure_fn_kwargs_cleavage_calc.outputs.calc_fn,
        _calc_structure_fn_kwargs=wf.cleavage_calcs_kwargs,
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
def rigid_and_relaxed_cleavage_study(wf,
                    gb_structure,
                    gb_structure_energy,
                    gb_plane_cart_loc,
                    calculation_engine = None,
                    calc_structure_fn = None,
                    calc_structure_fn_kwargs = None,
                    static_engine = None,
                    static_calc_structure_fn_kwargs= None,
                    static_calc_structure_fn = None,
                    CleaveGBStructure_Input = None,  # Replace `Any` with actual type if known
                    PlotCleave_Input = None          # Replace `Any` with actual type if known
                   ):
        from pyiron_workflow_atomistics.utils import get_calc_fn_calc_fn_kwargs_from_calculation_engine
        wf.calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(calculation_engine = calculation_engine,
                                                                                        structure = gb_structure,
                                                                                        calc_structure_fn = calc_structure_fn,
                                                                                        calc_structure_fn_kwargs = calc_structure_fn_kwargs)
        wf.static_calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(calculation_engine = static_engine,
                                                                                        structure = gb_structure,
                                                                                        calc_structure_fn = static_calc_structure_fn,
                                                                                        calc_structure_fn_kwargs = static_calc_structure_fn_kwargs)
        from pyiron_workflow_atomistics.utils import get_working_subdir_kwargs  
        wf.calc_structure_fn_kwargs_cleavage_rigid = get_working_subdir_kwargs(calc_structure_fn_kwargs = wf.static_calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
                                                                        base_working_directory = wf.static_calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs["working_directory"],
                                                                        new_working_directory = "cleavage_rigid")
        from pyiron_workflow_atomistics.gb.cleavage import calc_cleavage_GB, PlotCleaveInput
        from pyiron_workflow_atomistics.utils import modify_dataclass
        wf.CleaveGBStructureInput = modify_dataclass(CleaveGBStructure_Input, "cleavage_target_coord", gb_plane_cart_loc)
        wf.calc_cleavage_rigid = calc_cleavage_GB(structure=gb_structure,
                                        energy = gb_structure_energy,
                                        calc_structure_fn = wf.static_calc_fn_calc_fn_kwargs.outputs.calc_fn,
                                        calc_structure_fn_kwargs = wf.calc_structure_fn_kwargs_cleavage_rigid,
                                        input_cleave_gb_structure = wf.CleaveGBStructureInput,
                                        input_plot_cleave = PlotCleave_Input)
        wf.calc_structure_fn_kwargs_cleavage_relax = get_working_subdir_kwargs(calc_structure_fn_kwargs = wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
                                                                        base_working_directory = wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs["working_directory"],
                                                                        new_working_directory = "cleavage_relax")
        wf.calc_cleavage_relax = calc_cleavage_GB(structure=gb_structure,
                                        energy = gb_structure_energy,
                                        calc_structure_fn = wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
                                        calc_structure_fn_kwargs = wf.calc_structure_fn_kwargs_cleavage_relax,
                                        input_cleave_gb_structure = wf.CleaveGBStructureInput,
                                        input_plot_cleave = PlotCleave_Input)
        return wf.calc_cleavage_rigid.outputs.cleavage_calcs_df,\
                wf.calc_cleavage_relax.outputs.cleavage_calcs_df