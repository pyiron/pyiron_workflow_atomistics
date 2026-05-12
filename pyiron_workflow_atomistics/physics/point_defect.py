"""Point-defect formation energies (vacancy, substitutional)."""
from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, run
from pyiron_workflow_atomistics.structure.defects import (
    create_vacancy,
    substitutional_swap,
)
from pyiron_workflow_atomistics.structure.transform import (
    create_supercell_with_min_dimensions,
)


@pwf.as_function_node("formation_energy")
def calculate_vacancy_formation_energy(
    vacancy_energy: float, supercell_energy: float
) -> float:
    """Vacancy formation energy as the bare difference (preserves the
    existing semantics in ``bulk_defect/vacancy.py``).

    NOTE: the textbook formula uses the per-atom chemical potential
    ``mu_bulk`` rather than the bulk supercell energy
    (E_f = E_vac − (N−1)·mu_bulk). The current macro intentionally keeps
    the simpler ``vacancy − supercell`` form for backwards compatibility;
    switching to the (N−1)/N normalisation is tracked as a separate
    physics improvement — out of scope for the cleanup.
    """
    formation_energy = vacancy_energy - supercell_energy
    return formation_energy


@pwf.as_macro_node("supercell_calc", "vacancy_calc", "vacancy_formation_energy")
def get_vacancy_formation_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    remove_atom_index: int = 0,
    min_dimensions: list = [12, 12, 12],
    vacancy_subdir: str = "vacancy",
    supercell_subdir: str = "supercell",
):
    """Standard vacancy formation energy macro.

    Examples
    --------
    See ``notebooks/vacancy_formation_energy.ipynb``.
    """
    wf.structure_supercell = create_supercell_with_min_dimensions(
        structure, min_dimensions=min_dimensions
    )
    wf.structure_with_vacancy = create_vacancy(
        wf.structure_supercell, remove_atom_index=remove_atom_index
    )
    wf.supercell_calc = run(
        wf.structure_supercell,
        engine=engine.with_working_directory(supercell_subdir),
    )
    wf.vacancy_calc = run(
        wf.structure_with_vacancy,
        engine=engine.with_working_directory(vacancy_subdir),
    )
    wf.vacancy_formation_energy = calculate_vacancy_formation_energy(
        vacancy_energy=wf.vacancy_calc.outputs.engine_output.final_energy,
        supercell_energy=wf.supercell_calc.outputs.engine_output.final_energy,
    )
    return wf.supercell_calc, wf.vacancy_calc, wf.vacancy_formation_energy


@pwf.as_function_node("E_f")
def _substitutional_formation_energy(E_sub, E_bulk, mu_solute, mu_host):
    E_f = E_sub - E_bulk - mu_solute + mu_host
    return E_f


@pwf.as_macro_node(
    "supercell_calc",
    "substitutional_calc",
    "substitutional_formation_energy",
)
def get_substitutional_formation_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    defect_site: int = 0,
    new_symbol: str = "Ni",
    mu_solute: float = 0.0,
    mu_host: float = 0.0,
    min_dimensions: list = [12, 12, 12],
    sub_subdir: str = "substitutional",
    supercell_subdir: str = "supercell",
):
    """Dilute substitutional formation energy:
    ``E_f = E_sub - E_supercell - mu_solute + mu_host``.
    """
    wf.structure_supercell = create_supercell_with_min_dimensions(
        structure, min_dimensions=min_dimensions
    )
    wf.structure_with_substitute = substitutional_swap(
        wf.structure_supercell, defect_site=defect_site, new_symbol=new_symbol
    )
    wf.supercell_calc = run(
        wf.structure_supercell,
        engine=engine.with_working_directory(supercell_subdir),
    )
    wf.substitutional_calc = run(
        wf.structure_with_substitute,
        engine=engine.with_working_directory(sub_subdir),
    )
    wf.substitutional_formation_energy = _substitutional_formation_energy(
        E_sub=wf.substitutional_calc.outputs.engine_output.final_energy,
        E_bulk=wf.supercell_calc.outputs.engine_output.final_energy,
        mu_solute=mu_solute,
        mu_host=mu_host,
    )
    return (
        wf.supercell_calc,
        wf.substitutional_calc,
        wf.substitutional_formation_energy,
    )
