import os
import sys
import glob
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import pyiron_workflow as pwf
from pyiron_workflow_atomistics.gb.gb_code import gb_generator as gbc
from pyiron_workflow import Workflow
from pyiron_workflow.api import for_node
from pyiron_workflow_atomistics.calculator import (
    calculate_structure_node,
    fillin_default_calckwargs,
)
from typing import List, Tuple, Optional, Dict, Any, Callable
from pyiron_workflow_atomistics.gb.utils import axis_to_index
from pyiron_workflow_atomistics.dataclass_storage import Engine
from pyiron_workflow_atomistics.utils import extract_outputs_from_EngineOutputs
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
        engine_outputs=df.calc_output,
        keys=["final_energy", "final_structure", "final_volume"]
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
        engine_outputs=df.calc_output,
        keys=["final_energy", "final_structure", "final_volume"]
    )
    energies = extracted_dict["final_energy"]
    structs = extracted_dict["final_structure"]
    lengths = [np.linalg.norm(np.array(struct.cell)[axis_to_index(axis)]) for struct in structs]
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
    #print(interpolated_structure, energies, structs, lengths, interpolated_energy)
    return interpolated_structure, interpolated_energy


@pwf.as_function_node("GB_energy")
def get_GB_energy(atoms, total_energy, e0_per_atom, gb_normal_axis="c"):
    # print(atoms, total_energy, e0_per_atom, gb_normal_axis)
    idx = axis_to_index(gb_normal_axis)
    cell = np.array(atoms.get_cell())
    normals = [i for i in range(3) if i != idx]
    area = np.linalg.norm(np.cross(cell[normals[0]], cell[normals[1]]))
    deltaE = total_energy - (len(atoms) * e0_per_atom)
    # print(f"deltaE: {deltaE}, area: {area}, bulk_reference_energy: {len(atoms) * e0_per_atom}")
    gamma_GB = deltaE / (2 * area) * 16.021766208  # eV to J/m^2
    # print(f"gamma_GB: {gamma_GB}")
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
        extended_names.append((f"ext_{extension:.3f}"))
    return extended_names


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
    calculation_engine: Engine | None = None,
    calc_structure_fn_kwargs: dict[str, Any] | None = None,
    calc_structure_fn: Callable | None = None,
    gb_normal_axis: str = "c",
    calc_structure_fn_kwargs_defaults: dict[str, Any] | None = None,
):
    from pyiron_workflow_atomistics.calculator import validate_calculation_inputs   
    wf.validate = validate_calculation_inputs(calculation_engine = calculation_engine,
                                              calc_structure_fn = calc_structure_fn,
                                              calc_structure_fn_kwargs = calc_structure_fn_kwargs)
    from pyiron_workflow_atomistics.utils import get_calc_fn_calc_fn_kwargs_from_calculation_engine
    wf.calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(calculation_engine = calculation_engine,
                                                                                   structure = gb_structure,
                                                                                   calc_structure_fn = calc_structure_fn,
                                                                                   calc_structure_fn_kwargs = calc_structure_fn_kwargs)
    # 1. Generate extended structures
    wf.extended_GBs = get_extended_struct_list(gb_structure, extensions=extensions)
    wf.extended_GBs_subdirnames = get_extended_names(extensions=extensions)
    wf.full_calc_kwargs = fillin_default_calckwargs(
        calc_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        default_values=calc_structure_fn_kwargs_defaults,
    )
    from pyiron_workflow_atomistics.utils import get_subdirpaths

    wf.extended_GBs_dirnames = get_subdirpaths(
        parent_dir=wf.full_calc_kwargs.outputs.full_calc_kwargs2["working_directory"],
        output_subdirs=wf.extended_GBs_subdirnames,
    )
    wf.full_calc_kwargs_for_fornode = fillin_default_calckwargs(
        calc_kwargs=wf.full_calc_kwargs.outputs.full_calc_kwargs2,
        default_values=calc_structure_fn_kwargs_defaults,
        remove_keys=["working_directory"],
    )
    from pyiron_workflow_atomistics.calculator import generate_kwargs_variants

    wf.kwargs_variants = generate_kwargs_variants(
        base_kwargs=wf.full_calc_kwargs_for_fornode.outputs.full_calc_kwargs2,
        key="working_directory",
        values=wf.extended_GBs_dirnames,
    )
    # 2. Compute energies/volumes for extended structures
    wf.extended_GBs_calcs = for_node(
        calculate_structure_node,
        zip_on=("structure", "_calc_structure_fn_kwargs"),
        structure=wf.extended_GBs.outputs.extended_structure_list,
        _calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        _calc_structure_fn_kwargs=wf.kwargs_variants.outputs.kwargs_variants,
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
    # print("In generate_deepcopy executing")
    return deepcopy(input_obj)

@pwf.as_function_node("length")
def get_length(extensions):
    # print("In get_length executing")
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
    calculation_engine: Engine | None = None,
    calc_structure_fn_kwargs: dict[str, Any] | None = None,
    calc_structure_fn: Callable | None = None,
    calc_structure_fn_kwargs_defaults=None,
    interpolate_min_n_points=5,
    gb_normal_axis="c",
):
    from pyiron_workflow_atomistics.calculator import validate_calculation_inputs
    wf.validate = validate_calculation_inputs(calculation_engine = calculation_engine,
                                              calc_structure_fn = calc_structure_fn,
                                              calc_structure_fn_kwargs = calc_structure_fn_kwargs)
    from pyiron_workflow_atomistics.utils import get_calc_fn_calc_fn_kwargs_from_calculation_engine
    wf.calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(calculation_engine = calculation_engine,
                                                                                   structure = gb_structure,
                                                                                   calc_structure_fn = calc_structure_fn,
                                                                                   calc_structure_fn_kwargs = calc_structure_fn_kwargs)
    # 1. First length-scan + optimise
    wf.stage1_opt = gb_length_optimiser(
        gb_structure=gb_structure,
        calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions=extensions_stage1,
        gb_normal_axis=gb_normal_axis,
        calc_structure_fn_kwargs_defaults=calc_structure_fn_kwargs_defaults,
    )

    # 2. Second (refined) scan + optimise
    wf.stage2_opt = gb_length_optimiser(
        gb_structure=wf.stage1_opt.outputs.min_interp_energy_GB_struct,
        calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions=extensions_stage2,
        gb_normal_axis=gb_normal_axis,
        calc_structure_fn_kwargs_defaults=calc_structure_fn_kwargs_defaults,
    )
    wf.stage2_opt_struct_copy = generate_deepcopy(
        wf.stage2_opt.outputs.min_interp_energy_GB_struct
    )
    wf.stage1_plot_len = get_length(extensions_stage1)
    # 3. Plot each stage
    wf.stage1_plot = get_gb_length_optimiser_plot(
        df=wf.stage1_opt.outputs.extended_GB_results,
        n_points=wf.stage1_plot_len,
        working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs["working_directory"],
        save_filename="gb_optimiser_stage1.jpg",
    )
    wf.stage2_plot_len = get_length(extensions_stage2)
    wf.stage2_plot = get_gb_length_optimiser_plot(
        df=wf.stage2_opt.outputs.extended_GB_results,
        n_points=wf.stage2_plot_len,
        working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs["working_directory"],
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
        working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs["working_directory"],
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

import numpy as np
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
        Must have 'atoms' and 'results' columns.
    plot_label : str, optional
        Label for this run.
    degree : int, optional
        Degree of polynomial fit.
    n_points : int or None, optional
        If set, only the n lowest-energy samples are used for plotting and fitting.
    save_path : str or None, optional
        File path to save the figure. If None, figure is not saved.
    dpi : int, optional
        Dots per inch when saving.
    figsize : tuple, optional
        Figure size in inches (width, height).
    """
    # print("In get_gb_length_optimiser_plot executing")
    # Prepare data
    df_copy = df.copy()
    df_copy["c"] = df_copy.structure.apply(lambda x: x.cell[-1][-1])
    df_copy["energy"] = df_copy.calc_output.apply(lambda r: r.final_energy)

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
    ax.scatter(df_copy["c"], df_copy["energy"], alpha=0.3, label=f"all points")
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
