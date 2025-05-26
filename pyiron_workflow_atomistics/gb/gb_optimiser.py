import os
import sys
import glob
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import pyiron_workflow as pwf
from . import gb_generator as gbc
from pyiron_workflow import Workflow 
from pyiron_workflow.api import for_node
from pyiron_workflow_atomistics.calculator import calculate_structure_node
from typing import List, Tuple

@pwf.as_function_node
def get_extended_struct_list(
    structure,
    extensions=np.linspace(-0.2, 0.8, 11)
):
    """
    Generate ASE structures with varied cell lengths along the specified axis.

    Parameters:
    -----------
    structure : ASE Atoms
        Reference ASE Atoms object.
    extensions : array-like of float, optional
        Offsets to add to the 'c' vector. Default is np.linspace(-0.2, 0.8, 11).

    Returns:
    --------
    extended_structure_list : list of ASE Atoms
        List of extended ASE structures.
    extensions : array-like of float
        The array of extensions applied.
    """
    base_structure = structure.copy()
    extended_structure_list = []
    for ext in extensions:
        structure = base_structure.copy()
        a, b, c = structure.get_cell_lengths_and_angles()[:3]
        structure.set_cell([a, b, c + ext], scale_atoms=True)
        extended_structure_list.append(structure)
    return extended_structure_list, extensions

@pwf.as_function_node
def convert_structure(structure, target="ase"):
    """
    Convert structure between ASE Atoms and Pymatgen Structure.

    Parameters:
    -----------
    structure : ASE Atoms or Pymatgen Structure
        Input structure to convert.
    target : str, optional
        Target format: 'ase' for ASE Atoms, 'pmg' or 'pymatgen' for Pymatgen Structure. Default is 'ase'.

    Returns:
    --------
    result : ASE Atoms or Pymatgen Structure
        Converted structure.

    Raises:
    -------
    ValueError
        If the target format is unknown.
    """
    if target == "ase":
        converted_structure = AseAtomsAdaptor.get_atoms(structure)
    elif target in ("pmg", "pymatgen"):
        converted_structure =  AseAtomsAdaptor.get_structure(structure)
    else:
        raise ValueError(f"Unknown target: {target}")
    return converted_structure
    
@pwf.as_function_node
def extract_energy_volume_data_forloop_node(
    df, axis='c', check_orthorhombic=False, tol=1e-6
):
    """
    Extract energies, structures, and cell lengths for a specified axis from calculation results.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing calculation results with 'converged', 'structure', and 'results' columns.
    axis : {'a','b','c'}, optional
        Cell axis to extract lengths along. Default is 'c'.
    check_orthorhombic : bool, optional
        If True, verify the cell is orthogonal within tolerance. Default is False.
    tol : float, optional
        Tolerance for orthogonality check. Default is 1e-6.

    Returns:
    --------
    energies : list of float
        List of total energies for converged runs.
    structs : list of Structure
        List of Pymatgen Structure objects for converged runs.
    lengths : list of float
        List of cell lengths along the specified axis.
    """
    axis_map = dict(a=0,b=1,c=2)
    if axis not in axis_map:
        raise ValueError("axis must be 'a','b','c'")
    idx = axis_map[axis]
    energies, structs, lengths = [], [], []
    for _, row in df.iterrows():
        if not row.converged:
            continue
        struct = row.structure
        cell = np.array(struct.cell)
        if check_orthorhombic:
            for i in range(3):
                for j in range(i):
                    if abs(cell[i]@cell[j]) > tol:
                        raise ValueError("Non-orthogonal cell")
        energies.append(row.results['energy'])
        structs.append(struct)
        lengths.append(np.linalg.norm(cell[idx]))
    return energies, structs, lengths

@pwf.as_function_node
def get_min_energy_structure_from_forloop_df(df, axis='c'):
    """
    Identify the structure with minimum energy from a results DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with calculation results.
    axis : {'a','b','c'}, optional
        Cell axis used for length extraction. Default is 'c'.

    Returns:
    --------
    min_energy_structure : Structure
        Pymatgen Structure with the lowest energy.
    min_energy : float
        Minimum energy value.

    Raises:
    -------
    ValueError
        If no converged runs are found.
    """
    energies, structs, _ = extract_energy_volume_data_forloop_node.node_function(df, axis)
    if not energies:
        raise ValueError("No converged runs.")
    i = int(np.argmin(energies))
    min_energy_structure = structs[i]
    min_energy = energies[i]
    return min_energy_structure, min_energy

@pwf.as_function_node("modified_structure")
def get_modified_cell_structure(structure, cell):
    """
    Update a structure's cell parameters preserving fractional coordinates.

    Parameters:
    -----------
    structure : ASE Atoms or Pymatgen Structure
        Reference structure to modify.
    cell : array-like
        New cell matrix or lengths and angles.

    Returns:
    --------
    atoms : ASE Atoms or Pymatgen Structure
        Structure with updated cell.
    """
    modified_structure = structure.copy()
    modified_structure.set_cell(cell, scale_atoms=True)
    return modified_structure

@pwf.as_function_node()
def fit_polynomial_extremum(x_vals, y_vals, degree=2, num_points=None, extremum='min'):
    """
    Fit a polynomial to data and find its extremum.

    Parameters:
    -----------
    x_vals : array-like of float
        Independent variable data.
    y_vals : array-like of float
        Dependent variable data.
    degree : int, optional
        Degree of the polynomial fit (>=2). Default is 2.
    num_points : int, optional
        Number of points to use for fitting. Default is None.
    extremum : {'min','max'}, optional
        Type of extremum to find. Default is 'min'.

    Returns:
    --------
    ext_val : tuple (float, float)
        Extremum location and polynomial value at that point.
    coeffs : ndarray
        Coefficients of the fitted polynomial.

    Raises:
    -------
    ValueError
        If degree < 2.
    RuntimeError
        If no extremum is found.
    """
    x = np.array(x_vals, float)
    y = np.array(y_vals, float)
    if degree < 2:
        raise ValueError("Degree must be >= 2")
    if num_points and num_points < len(y):
        idxs = np.argsort(y) if extremum == 'min' else np.argsort(-y)
        idxs = idxs[:num_points]
        x, y = x[idxs], y[idxs]
    coeffs = np.polyfit(x, y, degree)
    roots = np.roots(np.polyder(coeffs))
    real_roots = roots[np.isreal(roots)].real
    second_derivative = np.polyder(coeffs, 2)
    candidates = [r for r in real_roots if (
        extremum=='min' and np.polyval(second_derivative, r) > 0
    ) or (
        extremum=='max' and np.polyval(second_derivative, r) < 0
    )]
    if not candidates:
        raise RuntimeError(f"No {extremum} found")
    vals = [(r, np.polyval(coeffs, r)) for r in candidates]
    ext_val = min(vals, key=lambda t: t[1]) if extremum=='min' else max(vals, key=lambda t: t[1])
    return ext_val, coeffs

@pwf.as_function_node("interpolated_structure", "interpolated_energy")
def get_interp_min_energy_structure_from_forloop_df(
    df, axis='c', check_orthorhombic=False, tol=1e-6,
    degree=2, num_points=None
):
    """
    Interpolate to find the minimum-energy structure from a dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with calculation results.
    axis : {'a','b','c'}, optional
        Axis for cell length interpolation. Default is 'c'.
    check_orthorhombic : bool, optional
        Verify the cell is orthogonal within tolerance. Default is False.
    tol : float, optional
        Tolerance for orthogonality check. Default is 1e-6.
    degree : int, optional
        Degree of polynomial for fitting. Default is 2.
    num_points : int, optional
        Number of data points to include in fit. Default is None.

    Returns:
    --------
    interpolated_structure : ASE Atoms or Pymatgen Structure
        Structure with cell set to interpolated minimum-energy.
    interpolated_energy : float
        Interpolated minimum energy value.
    """
    energies, structs, lengths = extract_energy_volume_data_forloop_node.node_function(
        df, axis, check_orthorhombic, tol
    )
    (length_min, interpolated_energy), _ = fit_polynomial_extremum.node_function(
        lengths, energies, degree, num_points, extremum='min'
    )
    ref_idx = int(np.argmin(energies))
    ref_struct = structs[ref_idx]
    cell = np.array(ref_struct.cell)
    idx = dict(a=0,b=1,c=2)[axis]
    unit_vec = cell[idx] / np.linalg.norm(cell[idx])
    cell[idx] = unit_vec * length_min
    interpolated_structure = get_modified_cell_structure.node_function(ref_struct, cell)
    return interpolated_structure, interpolated_energy

@pwf.as_function_node("GB_energy")
def get_GB_energy(atoms, total_energy, e0_per_atom, gb_normal_axis='c'):
    """
    Compute grain boundary energy per unit area for a bicrystal.

    Parameters:
    -----------
    atoms : ASE Atoms
        Bicrystal atoms object.
    total_energy : float
        Total energy of the bicrystal.
    e0_per_atom : float
        Bulk reference energy per atom.
    gb_normal_axis : {'a','b','c'}, optional
        Axis normal to grain boundary plane. Default is 'c'.

    Returns:
    --------
    gamma_GB : float
        Grain boundary energy per unit area.
    """
    idx = dict(a=0,b=1,c=2)[gb_normal_axis]
    cell = np.array(atoms.get_cell())
    normals = [i for i in range(3) if i != idx]
    area = np.linalg.norm(np.cross(cell[normals[0]], cell[normals[1]]))
    deltaE = total_energy - len(atoms) * e0_per_atom
    gamma_GB = deltaE / area / 2
    return gamma_GB

@pwf.as_function_node("excess_volume")
def get_GB_exc_volume(atoms, bulk_vol_per_atom, gb_normal_axis='c'):
    """
    Compute grain boundary excess volume per unit area.

    Parameters:
    -----------
    atoms : ASE Atoms
        Bicrystal atoms object.
    bulk_vol_per_atom : float
        Bulk reference volume per atom.
    gb_normal_axis : {'a','b','c'}, optional
        Axis normal to grain boundary plane. Default is 'c'.

    Returns:
    --------
    excess_volume : float
        Grain boundary excess volume per unit area.
    """
    idx = dict(a=0,b=1,c=2)[gb_normal_axis]
    cell = np.array(atoms.get_cell())
    normals = [i for i in range(3) if i != idx]
    area = np.linalg.norm(np.cross(cell[normals[0]], cell[normals[1]]))
    delta_vol = atoms.get_volume() - len(atoms) * bulk_vol_per_atom
    excess_volume = delta_vol / area / 2
    return excess_volume

@pwf.as_function_node("output_dirs")
def get_subdirpaths(
    parent_dir: str,
    output_subdirs: List[str]
):
    """
    Generate a list of calculation keyword dictionaries with extended output directories,
    and also return the original calc_kwargs without its output_dir.

    Parameters
    ----------
    calc_kwargs : dict
        Original calculation keyword arguments. Must include 'output_dir'.
    output_subdirs : list of str, optional
        List of subdirectory names to append to the base output_dir.
        Default: ['0.1', '0.2', ..., '0.7'].

    Returns
    -------
    extended_kwargs_list : list of dict
        Each dict is a copy of calc_kwargs (minus the original 'output_dir'),
        with a new 'output_dir' = os.path.join(base_output_dir, subdir).
    calc_kwargs_without_output_dir : dict
        The original calc_kwargs with 'output_dir' removed.
    """

    # Build extended kwargs
    dirpaths = []
    for sub in output_subdirs:
        output_subdir = os.path.join(parent_dir, sub)
        dirpaths.append(output_subdir)

    return dirpaths

@pwf.as_function_node("extended_dirnames")
def get_extended_names(extensions):
    extended_names = []
    for extension in extensions:
        extended_names.append((f"ext_{extension:.3f}"))
    return extended_names
    
@pwf.as_function_node("full_calc_kwargs2")
def fillin_default_calckwargs(calc_kwargs, default_values=None):
    """
    Take a partial calc_kwargs dict and fill in any missing entries
    with the standard defaults, but allow an optional `default_values`
    dict to override any of those built-in defaults.

    Parameters
    ----------
    calc_kwargs : dict
        User-provided kwargs for the calculation (may be partial).
    default_values : dict, optional
        If provided, these key→value pairs will override the built-in defaults.

    Returns
    -------
    full_calc_kwargs : dict
        A dict containing every argument, using user values when present,
        then `default_values`, then the built-in defaults.
    """
    # 1) define your built-in defaults
    built_in = {
        "output_dir":           "calc_dir",
        "fmax":                 0.01,
        "max_steps":            1000,
        "properties":           ('energy', 'forces', 'stresses'),
        "write_to_disk":        False,
        "initial_struct_path":  'initial_structure.xyz',
        "initial_results_path": 'initial_results.json',
        "traj_struct_path":     'trajectory.xyz',
        "traj_results_path":    'trajectory_results.json',
        "final_struct_path":    'final_structure.xyz',
        "final_results_path":   'final_results.json',
    }

    # 2) overlay any user-supplied default overrides
    if default_values:
        built_in.update(default_values)

    # 3) build the final dict: user → default_values → built-in
    full = {}
    for key, default in built_in.items():
        if key in calc_kwargs:
            full[key] = calc_kwargs[key]
        else:
            full[key] = default

    # 4) ensure tuple for properties
    full["properties"] = tuple(full["properties"])

    return full
    

@Workflow.wrap.as_macro_node(
    "extended_GB_results",
    "min_energy_GB_struct",
    "min_energy_GB_energy",
    "min_interp_energy_GB_struct",
    "min_interp_energy_GB_energy",
    "exc_volume",
    "gb_energy"
)
def gb_length_optimiser(
    wf,
    gb_structure,
    calc,
    calc_kwargs,
    equil_bulk_volume,
    equil_bulk_energy,
    extensions,
    gb_normal_axis="c",
    calc_kwargs_defaults = {
        "output_dir":           "gb_length_opt",
        "fmax":                 0.01,
        "max_steps":            1000,
        "properties":           ('energy', 'forces', 'stresses'),
        "write_to_disk":        False,
        "initial_struct_path":  'initial_structure.xyz',
        "initial_results_path": 'initial_results.json',
        "traj_struct_path":     'trajectory.xyz',
        "traj_results_path":    'trajectory_results.json',
        "final_struct_path":    'final_structure.xyz',
        "final_results_path":   'final_results.json',
    }
):
    """
    Macro node to extend GB structures over a range of lengths, compute energies/volumes, and extract GB excess properties.

    Parameters:
    -----------
    wf : pwf.Workflow
        The Workflow to which nodes are added.
    gb_structure : Structure
        (Ideally) Bulk-equilibrated GB structure.
    calc : Calc
        Calculation setup/node to apply to each extended structure.
    equil_bulk_volume : float
        Equilibrium volume per atom for GB excess volume calculation.
    equil_bulk_energy : float
        Bulk reference energy per atom for GB energy calculation.
    extensions : array-like of float
        Relative extensions to apply (e.g., [-0.2, 0.2, ...]).
    gb_normal_axis : {'a','b','c'}, optional
        Axis normal to the GB plane. Default is 'c'.

    Returns:
    --------
    extended_GBs_calcs : Node
        Node with calculation DataFrame for extended GBs.
    min_energy_GB_struct : Node
        Minimum-energy GB structure from scan.
    min_energy_GB_energy : Node
        Energy of the minimum-energy GB.
    min_interp_energy_GB_struct : Node
        Interpolated minimum-energy structure.
    min_interp_energy_GB_energy : Node
        Interpolated minimum energy value.
    exc_volume : Node
        Grain boundary excess volume per area node.
    gb_energy : Node
        Grain boundary energy per area node.
    """
    # 1. Generate extended structures
    wf.extended_GBs = get_extended_struct_list(
        gb_structure,
        extensions=extensions
    )
    wf.extended_GBs_subdirnames = get_extended_names(extensions = extensions)
    wf.full_calc_kwargs = fillin_default_calckwargs(calc_kwargs, calc_kwargs_defaults)
    wf.extended_GBs_dirnames = get_subdirpaths(parent_dir = wf.full_calc_kwargs.outputs.full_calc_kwargs2["output_dir"],
                                               output_subdirs = wf.extended_GBs_subdirnames)
    # 2. Compute energies/volumes for extended structures
    wf.extended_GBs_calcs = for_node(
        calculate_structure_node,
        #iter_on=("structure",),
        zip_on = ("structure", "output_dir"),
        structure = wf.extended_GBs.outputs.extended_structure_list,
        output_dir = wf.extended_GBs_dirnames.outputs.output_dirs,
        # These args are hardcoded (should be exactly the same as calculate_structure_node excluding structure and output_dir above)
        # because no unzipping of kwargs is possible in a macro
        calc = calc,
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
    # 3. Fit and extract minimum-energy structure
    wf.GB_min_energy_struct = get_min_energy_structure_from_forloop_df(
        wf.extended_GBs_calcs.outputs.df
    )
    # 4. Interpolate the min-energy GB from the datapoints
    wf.GB_min_energy_struct_interp = get_interp_min_energy_structure_from_forloop_df(
        wf.extended_GBs_calcs.outputs.df
    )
    # 5. Compute GB excess volume per area
    wf.exc_volume = get_GB_exc_volume(
        wf.GB_min_energy_struct_interp.outputs.interpolated_structure,
        equil_bulk_volume,
        gb_normal_axis=gb_normal_axis
    )
    # 6. Compute GB energy per area
    wf.gb_energy = get_GB_energy(
        atoms=wf.GB_min_energy_struct_interp.outputs.interpolated_structure,
        total_energy=wf.GB_min_energy_struct_interp.outputs.interpolated_energy,
        e0_per_atom=equil_bulk_energy,
        gb_normal_axis=gb_normal_axis
    )
    return (
        wf.extended_GBs_calcs,
        wf.GB_min_energy_struct.outputs.min_energy_structure,
        wf.GB_min_energy_struct.outputs.min_energy,
        wf.GB_min_energy_struct_interp.outputs.interpolated_structure,
        wf.GB_min_energy_struct_interp.outputs.interpolated_energy,
        wf.exc_volume,
        wf.gb_energy
    )

@pwf.as_function_node
def get_concat_df(df_list):
    concat_df = pd.concat(df_list)
    return concat_df
    
@Workflow.wrap.as_macro_node(
    "stage1_opt_struct",
    "stage1_opt_excvol",
    "stage1_opt_GBEnergy",
    "stage2_opt_struct",
    "stage2_opt_excvol",
    "stage2_opt_GBEnergy",
    "stage1_plot",
    "stage2_plot",
    "concat_results",
    "combined_plot",
    "gb_structure_final"
)
def full_gb_length_optimization(
    wf,
    gb_structure,
    calc,
    calc_kwargs,
    equil_bulk_volume,
    equil_bulk_energy,
    extensions_stage1,
    extensions_stage2,
    interpolate_min_n_points=5,
    gb_normal_axis="c"
):
    # 1. First length-scan + optimise
    wf.stage1_opt = gb_length_optimiser(
        gb_structure=gb_structure,
        calc=calc,
        calc_kwargs=calc_kwargs,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions=extensions_stage1,
        gb_normal_axis=gb_normal_axis
    )

    # 2. Second (refined) scan + optimise
    wf.stage2_opt = gb_length_optimiser(
        gb_structure=wf.stage1_opt.outputs.min_interp_energy_GB_struct,
        calc=calc,
        calc_kwargs=calc_kwargs,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions=extensions_stage2,
        gb_normal_axis=gb_normal_axis
    )
    wf.stage1_plot_len = pwf.api.std.Length(extensions_stage1)
    # 3. Plot each stage
    wf.stage1_plot = get_gb_length_optimiser_plot(
        df=wf.stage1_opt.outputs.extended_GB_results,
        n_points=wf.stage1_plot_len,
        save_path="gb_optimiser_whole/gb_optimiser_stage1.jpg"
    )
    wf.stage2_plot_len = pwf.api.std.Length(extensions_stage2)
    wf.stage2_plot = get_gb_length_optimiser_plot(
        df=wf.stage2_opt.outputs.extended_GB_results,
        n_points=wf.stage2_plot_len,
        save_path="gb_optimiser_whole/gb_optimiser_stage2.jpg"
    )

    # 4. Concatenate results and re-plot combined
    wf.concat_results = pwf.api.inputs_to_list(
        2,
        wf.stage1_opt.outputs.extended_GB_results,
        wf.stage2_opt.outputs.extended_GB_results
    )
    wf.concat_df = get_concat_df(wf.concat_results)

    wf.combined_plot = get_gb_length_optimiser_plot(
        df=wf.concat_df,
        n_points=interpolate_min_n_points,
        save_path="gb_optimiser_whole/gb_optimiser_combined.jpg"
    )

    # 5. Return the key outputs
    return (
        wf.stage1_opt.outputs.min_interp_energy_GB_struct,
        wf.stage1_opt.outputs.exc_volume,
        wf.stage1_opt.outputs.min_interp_energy_GB_energy,
        wf.stage2_opt.outputs.min_interp_energy_GB_struct,
        wf.stage2_opt.outputs.exc_volume,
        wf.stage2_opt.outputs.min_interp_energy_GB_energy,
        wf.stage1_plot,
        wf.stage2_plot,
        wf.concat_df,
        wf.combined_plot,
        wf.stage2_opt.outputs.min_interp_energy_GB_struct
    )


    
import numpy as np
import matplotlib.pyplot as plt
@pwf.as_function_node
def get_gb_length_optimiser_plot(df,
                                 plot_label='run',
                                 degree=2,
                                 n_points=None,
                                 save_path=None,
                                 dpi=300,
                                 figsize=(6, 4)):
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
    # Prepare data
    df_copy = df.copy()
    df_copy['c'] = df_copy.atoms.apply(lambda x: x.cell[-1][-1])
    df_copy['energy'] = df_copy.results.apply(lambda r: r['energy'])

    # Optionally select only the n smallest energy points
    if isinstance(n_points, int) and n_points > 0:
        df_fit = df_copy.nsmallest(n_points, 'energy')
    else:
        df_fit = df_copy

    x = df_fit['c'].to_numpy()
    y = df_fit['energy'].to_numpy()

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
    ax.scatter(df_copy['c'], df_copy['energy'], alpha=0.3, label=f'all points')
    ax.scatter(x, y, label=f'{plot_label} (n={len(x)})')
    ax.plot(x_fit, y_fit, label=f'fit {plot_label}')

    if v is not None:
        ax.axvline(v, linestyle='--', label=f'min {plot_label}')
        # offset in data units: 5% of x-range
        x_range = x.max() - x.min()
        offset_data = 0.05 * x_range
        ax.text(v + offset_data, 0.95, f"{v:.3f}", rotation=90,
                va="top", ha="center",
                transform=ax.get_xaxis_transform())

    ax.set_xlabel('c (Å)')
    ax.set_ylabel('energy (eV)')
    ax.legend()
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi)

    return fig
