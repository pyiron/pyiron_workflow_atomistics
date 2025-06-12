import numpy as np
from ase import Atoms
import pyiron_workflow as pwf
from pyiron_workflow import Workflow

from .calculator import extract_values, calculate_structure_node
import os
from typing import Callable, Tuple, Dict, Any, Optional, List
from .calculator import ase_calculate_structure_node_interface


@pwf.as_function_node("structure_list")
def generate_structures(
    base_structure: Atoms,
    axes: list[str] = ["iso"],
    strain_range: tuple[float, float] = (-0.2, 0.2),
    num_points: int = 11,
) -> list[Atoms]:
    """
    Generate a list of strained ASE Atoms structures.

    Parameters
    ----------
    base_structure
        ASE Atoms object to be strained.
    axes
        List of axes to strain simultaneously: any combination of "a", "b", "c".
        Use ["iso"] for isotropic (all axes) by default.
    strain_range
        (min_strain, max_strain), e.g. (-0.2, 0.2) for ±20%.
    num_points
        Number of steps in the strain grid.

    Returns
    -------
    List of ASE Atoms, one per epsilon value with specified axes strained.
    """
    structure_list: list[Atoms] = []
    start, end = strain_range

    for epsilon in np.linspace(start, end, num_points):
        s = base_structure.copy()
        cell = s.get_cell()

        # isotropic if requested
        if "iso" in [ax.lower() for ax in axes]:
            new_cell = cell * (1 + epsilon)
        else:
            new_cell = cell.copy()
            for ax in axes:
                ax_lower = ax.lower()
                if ax_lower == "a":
                    new_cell[0] = cell[0] * (1 + epsilon)
                elif ax_lower == "b":
                    new_cell[1] = cell[1] * (1 + epsilon)
                elif ax_lower == "c":
                    new_cell[2] = cell[2] * (1 + epsilon)
                else:
                    # ignore unknown axis labels
                    continue
        s.set_cell(new_cell, scale_atoms=True)
        #print(s)
        structure_list.append(s)

    return structure_list


@pwf.as_function_node("e0", "v0", "B")
def equation_of_state(energies, volumes, eos="sj"):
    from ase.eos import EquationOfState

    eos = EquationOfState(volumes, energies, eos=eos)
    v0, e0, B = eos.fit()  # v0, e0, B
    return e0, v0, B  # eos_results

@pwf.as_function_node("structures", "results_dict", "convergence_lst")
def evaluate_structures(
    structures: list[Atoms],
    calc_structure_fn: Callable[..., Any] = ase_calculate_structure_node_interface,
    calc_structure_fn_kwargs: dict[str, Any] | None = None,
):
    """
    Evaluate each structure, writing each one's results under its own subfolder.

    - structures: list of ASE Atoms
    - calc: ASE calculator
    - calc_kwargs: base kwargs for calc_structure_with_trajectory
    - working_directory: top‐level directory to dump each strain’s files into
    - write_to_disk: whether to write outputs to disk

    Returns
    -------
    rel_structs_lst : list of ASE Atoms
        The final relaxed structure for each input.
    results_lst : list of dict
        The final-results dict for each input.
    convergence_lst : list of bool
        Convergence flag for each calculation.
    """
    
    os.makedirs(calc_structure_fn_kwargs["working_directory"], exist_ok=True)

    rel_structs_lst = []
    results_lst = []
    convergence_lst = []

    for i, struct in enumerate(structures):
        # per-structure subfolder
        local_kwargs = calc_structure_fn_kwargs.copy()
        
        strain_dir = os.path.join(local_kwargs["working_directory"], f"strain_{i:03d}")
        #print(s)
        # start from the user’s calc_kwargs, preserving any keys they set
        local_kwargs["working_directory"] = strain_dir
        # run the full trajectory-enabled calculation
        #print(local_kwargs)
        atoms, final_results, converged = calculate_structure_node.node_function(structure = struct,
                                                     calc_structure_fn = calc_structure_fn,
                                                     calc_structure_fn_kwargs = local_kwargs)

        # unpack final results
        rel_structs_lst.append(atoms)
        results_lst.append(final_results)
        convergence_lst.append(converged)

    return rel_structs_lst, results_lst, convergence_lst


# @pwf.as_function_node("energies", "volumes")
# def extract_energies_volumes_from_output(results, energy_parser_func, energy_parser_func_kwargs, volume_parser_func, volume_parser_func_kwargs):
#     energies = energy_parser_func(results, **energy_parser_func_kwargs)
#     volumes = volume_parser_func(results, **volume_parser_func_kwargs)
#     return energies, volumes


@pwf.as_function_node("equil_struct")
def get_bulk_structure(
    name: str,
    crystalstructure=None,
    a=None,
    b=None,
    c=None,
    alpha=None,
    covera=None,
    u=None,
    orthorhombic=False,
    cubic=False,
    basis=None,
):
    from ase.build import bulk

    equil_struct = bulk(
        name=name,
        crystalstructure=crystalstructure,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        covera=covera,
        u=u,
        orthorhombic=orthorhombic,
        cubic=cubic,
        basis=basis,
    )
    return equil_struct


@pwf.as_function_node("a0")
def get_equil_lat_param(eos_output):
    a0 = eos_output ** (1 / 3)
    return a0

@Workflow.wrap.as_macro_node("v0", "e0", "B", "volumes", "energies")
def eos_volume_scan(
    wf,
    base_structure,
    calc_structure_fn = ase_calculate_structure_node_interface,
    calc_structure_fn_kwargs: dict[str, Any] | None = None,
    axes=["a", "b", "c"],
    strain_range=(-0.2, 0.2),
    num_points=11,
    
):
    # 1) generate strained structures
    wf.structures_list = generate_structures(
        base_structure,
        axes=axes,
        strain_range=strain_range,
        num_points=num_points,
    )

    # 2) evaluate them in subfolders under working_directory
    wf.evaluation = evaluate_structures(
        structures=wf.structures_list,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )

    # 3) extract energies and volumes
    wf.energies = extract_values(
        wf.evaluation.outputs.results_dict,
        key="energy",
    )
    wf.volumes = extract_values(
        wf.evaluation.outputs.results_dict,
        key="volume",
    )

    # 4) fit EOS
    wf.eos = equation_of_state(
        wf.energies,
        wf.volumes,
        eos="birchmurnaghan",
    )

    return (
        wf.eos.outputs.v0,
        wf.eos.outputs.e0,
        wf.eos.outputs.B,
        wf.volumes,
        wf.energies,
    )
