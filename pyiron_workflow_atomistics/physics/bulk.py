import warnings
from typing import Optional

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from pyiron_workflow import Workflow

from pyiron_workflow_atomistics.engine import Engine, run
from pyiron_workflow_atomistics.structure.build import get_bulk
from pyiron_workflow_atomistics.structure.transform import rattle
from pyiron_workflow_atomistics.analysis.quantities import get_per_atom_quantity


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
    engine: Engine,
    parent_working_directory: str = ".",
):
    engine_output_lst = []
    for i, struct in enumerate(structures):
        sub_engine = engine.with_working_directory(f"strain_{i:03d}")
        engine_output_lst.append(run.node_function(structure=struct, engine=sub_engine))
    return engine_output_lst


@pwf.as_function_node("energies")
def _extract_energies(engine_outputs):
    return [o.final_energy for o in engine_outputs]


@pwf.as_function_node("volumes")
def _extract_volumes(engine_outputs):
    return [o.final_volume for o in engine_outputs]


@pwf.as_function_node("structures")
def _extract_structures(engine_outputs):
    return [o.final_structure for o in engine_outputs]


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
    engine: Engine,
    rattle_amount: float = 0.0,
    strain_range=(-0.02, 0.02),
    num_points=11,
    parent_working_directory: str = "opt_cubic_cell",
    eos_type="birchmurnaghan",
):
    wf.rattle_structure = rattle(structure, rattle=rattle_amount)
    wf.eos = eos_volume_scan(
        base_structure=wf.rattle_structure,
        engine=engine,
        axes=["a", "b", "c"],
        strain_range=strain_range,
        num_points=num_points,
        eos_type=eos_type,
    )
    wf.a0 = get_cubic_equil_lat_param(wf.eos.outputs.v0)
    wf.eq_bulk_struct = get_bulk(
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
    engine: Engine,
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
        engine=engine,
    )

    # 3) extract energies and volumes
    wf.energies = _extract_energies(wf.evaluation.outputs.engine_output_lst)
    wf.volumes = _extract_volumes(wf.evaluation.outputs.engine_output_lst)
    wf.structures = _extract_structures(wf.evaluation.outputs.engine_output_lst)

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
