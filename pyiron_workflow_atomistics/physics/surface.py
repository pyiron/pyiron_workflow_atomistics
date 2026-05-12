"""Surface energy workflow."""

from typing import Optional, Tuple, Union

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, run
from pyiron_workflow_atomistics.structure.build import create_surface_slab


@pwf.as_function_node("mu_bulk_out")
def _calculate_if_not_present_(input_structure, engine: Engine, mu_bulk=None):
    if mu_bulk is None:
        output = run.node_function(input_structure, engine=engine)
        mu_bulk_out = output.final_energy / len(input_structure)
    else:
        mu_bulk_out = mu_bulk
    return mu_bulk_out


@pwf.as_function_node("surface_energy")
def get_surface_energy(E_slab, E_bulk_per_atom, N_slab, area_one_side):
    gamma_fs = (E_slab - N_slab * E_bulk_per_atom) / (2.0 * area_one_side)
    gamma_J_per_m2 = gamma_fs * 16.021766208
    return gamma_J_per_m2


@pwf.as_function_node("area_one_side")
def area_one_side(slab):
    cell = slab.cell
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    return area


@pwf.as_function_node("n_atoms")
def get_n_atoms(atoms):
    n = len(atoms)
    return n


@pwf.as_macro_node(
    "unrelaxed_surface",
    "relaxed_surface",
    "relaxed_surface_calc_output",
    "mu_bulk",
    "surface_energy",
)
def calculate_surface_energy(
    wf,
    bulk_structure: Atoms,
    engine: Engine,
    miller_indices: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (1, 1, 1),
    layers: int = 3,
    vacuum: float = 10.0,
    periodic: bool = True,
    mu_bulk: Optional[float] = None,
):
    """Calculate the surface energy (J/m^2) of a slab cut from ``bulk_structure``.

    Builds a vacuum-free reference slab and a vacuum-padded slab from the
    same bulk crystal, relaxes the latter with ``engine``, and divides the
    excess energy by twice the in-plane area.
    """
    wf.slab_novac = create_surface_slab(
        bulk_structure=bulk_structure,
        miller_indices=miller_indices,
        layers=layers,
        vacuum=0.0,
        periodic=periodic,
    )
    wf.slab_vac = create_surface_slab(
        bulk_structure=bulk_structure,
        miller_indices=miller_indices,
        layers=layers,
        vacuum=vacuum,
        periodic=periodic,
    )
    wf.calc_slab = run(wf.slab_vac, engine=engine, label="calc_slab")
    wf.mu_bulk_out = _calculate_if_not_present_(
        wf.slab_novac, engine=engine, mu_bulk=mu_bulk
    )
    wf.n_atoms_slab = get_n_atoms(wf.slab_vac)
    wf.area_one_side = area_one_side(wf.slab_novac)
    wf.surface_energy = get_surface_energy(
        E_slab=wf.calc_slab.outputs.engine_output.final_energy,
        E_bulk_per_atom=wf.mu_bulk_out,
        N_slab=wf.n_atoms_slab,
        area_one_side=wf.area_one_side,
    )
    return (
        wf.slab_vac,
        wf.calc_slab.outputs.engine_output.final_structure,
        wf.calc_slab,
        wf.mu_bulk_out,
        wf.surface_energy,
    )
