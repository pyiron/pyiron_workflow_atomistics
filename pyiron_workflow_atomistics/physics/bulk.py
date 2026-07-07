from __future__ import annotations

import warnings

import numpy as np
import pyiron_workflow._wfms.api as pwf
from ase import Atoms

from pyiron_workflow_atomistics.analysis.quantities import get_per_atom_quantity
from pyiron_workflow_atomistics.engine import Engine, calculate
from pyiron_workflow_atomistics.structure.build import get_bulk
from pyiron_workflow_atomistics.structure.transform import rattle


@pwf.atomic("structure_list")
def generate_structures(
    base_structure: Atoms,
    axes: list[str] | tuple[str] | None = None,
    strain_range: tuple[float, float] = (-0.2, 0.2),
    num_points: int = 11,
) -> list[Atoms]:
    if axes is None:
        axes = ["iso"]
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
                    warnings.warn(f"Unknown axis label: {ax}", stacklevel=2)
                    # ignore unknown axis labels
                    continue
        s.set_cell(new_cell, scale_atoms=True)
        structure_list.append(s)

    return structure_list


@pwf.atomic("e0", "v0", "B")
def equation_of_state(energies, volumes, eos_type="sj"):
    from ase.eos import EquationOfState

    eos = EquationOfState(volumes, energies, eos=eos_type)
    v0, e0, B = eos.fit()  # v0, e0, B
    B_GPa = B * 160.21766208  # eV to GPa
    return e0, v0, B_GPa  # eos_results


@pwf.atomic("engine_output_lst")
def evaluate_structures(
    structures: list[Atoms],
    engine: Engine,
    parent_working_directory: str = ".",
):
    engine_output_lst = []
    for i, struct in enumerate(structures):
        sub_engine = engine.with_working_directory(f"strain_{i:03d}")
        engine_output_lst.append(
            calculate.node_function(structure=struct, engine=sub_engine)
        )
    return engine_output_lst


@pwf.atomic("energies")
def _extract_energies(engine_outputs):
    return [o.final_energy for o in engine_outputs]


@pwf.atomic("volumes")
def _extract_volumes(engine_outputs):
    return [o.final_volume for o in engine_outputs]


@pwf.atomic("structures")
def _extract_structures(engine_outputs):
    return [o.final_structure for o in engine_outputs]


@pwf.atomic("a0")
def get_cubic_equil_lat_param(eos_output):
    a0 = eos_output ** (1 / 3)
    return a0


@pwf.workflow("v0", "e0", "B", "volumes", "structures", "energies")
def eos_volume_scan(
    base_structure,
    engine: Engine,
    axes=("a", "b", "c"),
    strain_range=(-0.2, 0.2),
    num_points=11,
    eos_type="birchmurnaghan",
):
    # 1) generate strained structures
    structures_list = generate_structures(
        base_structure,
        axes=axes,
        strain_range=strain_range,
        num_points=num_points,
    )

    # 2) evaluate them in subfolders under working_directory
    engine_output_lst = evaluate_structures(
        structures=structures_list,
        engine=engine,
    )

    # 3) extract energies and volumes
    volumes = _extract_volumes(engine_output_lst)
    structures = _extract_structures(engine_output_lst)
    energies = _extract_energies(engine_output_lst)

    # 4) fit EOS
    e0, v0, B_GPa = equation_of_state(energies, volumes, eos_type=eos_type)

    return v0, e0, B_GPa, volumes, structures, energies



@pwf.workflow(
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
    structure: Atoms,
    name: str,
    crystalstructure: str,
    engine: Engine,
    rattle_amount: float = 0.0,
    strain_range=(-0.02, 0.02),
    num_points=11,
    parent_working_directory: str = "opt_cubic_cell",
    eos_type="birchmurnaghan",
    axes=("a", "b", "c"),
    cubic: bool = True,
):
    rattle_structure = rattle(structure=structure, rattle=rattle_amount)
    v0, e0, B, volumes, structures, energies = eos_volume_scan(
        base_structure=rattle_structure,
        engine=engine,
        axes=axes,
        strain_range=strain_range,
        num_points=num_points,
        eos_type=eos_type,
    )
    a0 = get_cubic_equil_lat_param(eos_output=v0)
    eq_bulk_struct = get_bulk(
        name=name, crystalstructure=crystalstructure, a=a0, cubic=cubic
    )

    equil_energy_per_atom = get_per_atom_quantity(quantity=e0, structure=eq_bulk_struct)
    equil_volume_per_atom = get_per_atom_quantity(quantity=v0, structure=eq_bulk_struct)

    return (
        eq_bulk_struct,
        a0,
        B,
        equil_energy_per_atom,
        equil_volume_per_atom,
        volumes,
        structures,
        energies,
    )