import numpy as np
from ase import Atoms
import pyiron_workflow as pwf
from pyiron_workflow import Workflow

from .calculator import calculate_structure_node
import os
from typing import Callable, Tuple, Dict, Any, Optional, List
from pyiron_workflow_atomistics.utils import get_per_atom_quantity
from pyiron_workflow_atomistics.dataclass_storage import Engine
from pyiron_workflow_atomistics.calculator import validate_calculation_inputs
import warnings


@pwf.as_function_node("structure_list")
def generate_structures(
    base_structure: Atoms,
    axes: list[str] = ["iso"],
    strain_range: tuple[float, float] = (-0.2, 0.2),
    num_points: int = 11,
) -> list[Atoms]:
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
                    warnings.warn(f"Unknown axis label: {ax}")
                    # ignore unknown axis labels
                    continue
        s.set_cell(new_cell, scale_atoms=True)
        # print(s)
        structure_list.append(s)

    return structure_list


@pwf.as_function_node("e0", "v0", "B")
def equation_of_state(energies, volumes, eos_type="sj"):
    from ase.eos import EquationOfState

    eos = EquationOfState(volumes, energies, eos=eos_type)
    v0, e0, B = eos.fit()  # v0, e0, B
    B_GPa = B * 160.21766208  # eV to GPa
    return e0, v0, B_GPa  # eos_results


@pwf.as_function_node("engine_output_lst")
def evaluate_structures(
    structures: list[Atoms],
    calculation_engine: Optional[Engine] = None,
    calc_structure_fn: Optional[Callable[..., Any]] = None,
    calc_structure_fn_kwargs: Optional[dict[str, Any]] = None,
):
    validate_calculation_inputs(
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
        calculation_engine=calculation_engine,
    )
    if calculation_engine is not None:
        working_directory = calculation_engine.working_directory
        os.makedirs(working_directory, exist_ok=True)
        calc_structure_fn, calc_structure_fn_kwargs = calculation_engine.calculate_fn(
            structure=structures[0]
        )
    else:
        working_directory = calc_structure_fn_kwargs["working_directory"]
        os.makedirs(working_directory, exist_ok=True)
        calc_structure_fn_kwargs = calc_structure_fn_kwargs.copy()

    engine_output_lst = []
    for i, struct in enumerate(structures):
        # per-structure subfolder
        local_kwargs = calc_structure_fn_kwargs.copy()

        strain_dir = os.path.join(working_directory, f"strain_{i:03d}")
        # print(s)
        # start from the user’s calc_kwargs, preserving any keys they set
        local_kwargs["working_directory"] = strain_dir
        # run the full trajectory-enabled calculation
        # print(local_kwargs)
        engine_output = calculate_structure_node.node_function(
            structure=struct,
            _calc_structure_fn=calc_structure_fn,
            _calc_structure_fn_kwargs=local_kwargs,
        )

        # unpack final results
        engine_output_lst.append(engine_output)

    return engine_output_lst


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


import ase


@pwf.api.as_function_node("rattle_structure")
def rattle_structure(structure, rattle=None):
    if rattle:
        base_structure = structure.copy()
        base_structure.rattle(rattle)
    else:
        base_structure = structure.copy()
    return base_structure


from pyiron_workflow_atomistics.utils import modify_dict, add_string


@pwf.api.as_macro_node(
    "equil_struct",
    "a0",
    "B",
    "equil_energy_per_atom",
    "equil_volume_per_atom",
    "volumes",
    "structures",
    "energies",
)
def optimise_cubic_lattice_parameter(
    wf,
    structure: Atoms,
    name: str,
    crystalstructure: str,
    calculation_engine=None,
    calc_structure_fn: Optional[Callable[..., Any]] = None,
    calc_structure_fn_kwargs: Optional[dict[str, Any]] = None,
    rattle: float = 0.0,
    strain_range=(-0.02, 0.02),
    num_points=11,
    parent_working_directory: str = "opt_cubic_cell",
):
    wf.rattle_structure = rattle_structure(structure, rattle)
    from pyiron_workflow_atomistics.utils import (
        get_calc_fn_calc_fn_kwargs_from_calculation_engine,
    )

    wf.calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(
        calculation_engine=calculation_engine,
        structure=wf.rattle_structure,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )
    from pyiron_workflow_atomistics.utils import get_working_subdir_kwargs

    wf.calc_fn_kwargs = get_working_subdir_kwargs(
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        base_working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        new_working_directory=parent_working_directory,
    )
    # print(wf.calc_fn_kwargs)
    # 3. Attach the macro node to the workflow, capturing all outputs
    wf.eos = eos_volume_scan(
        base_structure=wf.rattle_structure,
        # calculation_engine = calculation_engine,
        calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        axes=["a", "b", "c"],
        strain_range=strain_range,
        num_points=num_points,
    )
    wf.a0 = get_cubic_equil_lat_param(wf.eos.outputs.v0)
    wf.eq_bulk_struct = get_bulk_structure(
        name=name, crystalstructure=crystalstructure, a=wf.a0, cubic=True
    )

    wf.equil_energy_per_atom = get_per_atom_quantity(
        wf.eos.outputs.e0, wf.eq_bulk_struct.outputs.equil_struct
    )
    wf.equil_volume_per_atom = get_per_atom_quantity(
        wf.eos.outputs.v0, wf.eq_bulk_struct.outputs.equil_struct
    )

    return (
        wf.eq_bulk_struct.outputs.equil_struct,
        wf.a0.outputs.a0,
        wf.eos.outputs.B,
        wf.equil_energy_per_atom,
        wf.equil_volume_per_atom,
        wf.eos.outputs.volumes,
        wf.eos.outputs.structures,
        wf.eos.outputs.energies,
    )


@pwf.as_function_node("a0")
def get_cubic_equil_lat_param(eos_output):
    a0 = eos_output ** (1 / 3)
    return a0


@Workflow.wrap.as_macro_node("v0", "e0", "B", "volumes", "structures", "energies")
def eos_volume_scan(
    wf,
    base_structure,
    calculation_engine: Optional[Engine] = None,
    calc_structure_fn: Optional[Callable[..., Any]] = None,
    calc_structure_fn_kwargs: Optional[dict[str, Any]] = None,
    axes=["a", "b", "c"],
    strain_range=(-0.2, 0.2),
    num_points=11,
    eos_type="birchmurnaghan",
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
        calculation_engine=calculation_engine,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )
    from pyiron_workflow_atomistics.calculator import (
        extract_output_values_from_EngineOutput,
    )

    # 3) extract energies and volumes
    wf.energies = extract_output_values_from_EngineOutput(
        wf.evaluation.outputs.engine_output_lst,
        key="final_energy",
    )
    wf.volumes = extract_output_values_from_EngineOutput(
        wf.evaluation.outputs.engine_output_lst,
        key="final_volume",
    )
    wf.structures = extract_output_values_from_EngineOutput(
        wf.evaluation.outputs.engine_output_lst,
        key="final_structure",
    )
    # 4) fit EOS
    wf.eos = equation_of_state(
        wf.energies,
        wf.volumes,
        eos_type=eos_type,
    )

    return (
        wf.eos.outputs.v0,
        wf.eos.outputs.e0,
        wf.eos.outputs.B,
        wf.volumes,
        wf.structures,
        wf.energies,
    )
