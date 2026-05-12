"""Point-defect formation energies (vacancy, substitutional)."""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, calculate, subengine
from pyiron_workflow_atomistics.structure.defects import (
    create_vacancy,
    substitutional_swap,
)
from pyiron_workflow_atomistics.structure.transform import (
    create_supercell_with_min_dimensions,
)


@pwf.as_function_node("n_atoms")
def _count_atoms(structure: Atoms) -> int:
    n_atoms = len(structure)
    return n_atoms


@pwf.as_function_node("formation_energy")
def calculate_vacancy_formation_energy(
    vacancy_energy: float,
    supercell_energy: float,
    n_atoms_supercell: int,
) -> float:
    """Vacancy formation energy with the correct (N−1)/N normalisation.

    .. math:: E_\\mathrm{f} = E_\\mathrm{vac} - \\frac{N-1}{N} E_\\mathrm{bulk}

    where ``E_bulk`` is the perfect-supercell total energy with ``N`` atoms,
    and ``E_vac`` is the supercell energy with one atom removed. Equivalent
    to the textbook form ``E_vac − (N−1)·mu_bulk`` where
    ``mu_bulk = E_bulk / N``.

    The previous bare ``E_vac − E_bulk`` form was off by exactly ``mu_bulk``
    (~3.7 eV/atom for typical foundation MLIPs in DFT-PBE; ~0.005 eV/atom
    for EMT, which is why the bug was invisible against classical
    references).
    """
    formation_energy = (
        vacancy_energy - (n_atoms_supercell - 1) / n_atoms_supercell * supercell_energy
    )
    return formation_energy


@pwf.as_macro_node("supercell_calc", "vacancy_calc", "vacancy_formation_energy")
def get_vacancy_formation_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    remove_atom_index: int = 0,
    min_dimensions: list | None = None,
    vacancy_subdir: str = "vacancy",
    supercell_subdir: str = "supercell",
):
    """Standard vacancy formation energy macro.

    Examples
    --------
    See ``notebooks/vacancy_formation_energy.ipynb``.
    """
    if min_dimensions is None:
        min_dimensions = [12, 12, 12]
    wf.structure_supercell = create_supercell_with_min_dimensions(
        structure, min_dimensions=min_dimensions
    )
    wf.structure_with_vacancy = create_vacancy(
        wf.structure_supercell, remove_atom_index=remove_atom_index
    )
    wf.supercell_engine = subengine(engine=engine, subdir=supercell_subdir)
    wf.vacancy_engine = subengine(engine=engine, subdir=vacancy_subdir)
    wf.supercell_calc = calculate(
        wf.structure_supercell, engine=wf.supercell_engine, label="supercell_calc"
    )
    wf.vacancy_calc = calculate(
        wf.structure_with_vacancy, engine=wf.vacancy_engine, label="vacancy_calc"
    )
    wf.n_atoms_supercell = _count_atoms(wf.structure_supercell)
    wf.vacancy_formation_energy = calculate_vacancy_formation_energy(
        vacancy_energy=wf.vacancy_calc.outputs.engine_output.final_energy,
        supercell_energy=wf.supercell_calc.outputs.engine_output.final_energy,
        n_atoms_supercell=wf.n_atoms_supercell,
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
    min_dimensions: list | None = None,
    sub_subdir: str = "substitutional",
    supercell_subdir: str = "supercell",
):
    """Dilute substitutional formation energy:
    ``E_f = E_sub - E_supercell - mu_solute + mu_host``.
    """
    if min_dimensions is None:
        min_dimensions = [12, 12, 12]
    wf.structure_supercell = create_supercell_with_min_dimensions(
        structure, min_dimensions=min_dimensions
    )
    wf.structure_with_substitute = substitutional_swap(
        wf.structure_supercell, defect_site=defect_site, new_symbol=new_symbol
    )
    wf.supercell_engine = subengine(engine=engine, subdir=supercell_subdir)
    wf.substitutional_engine = subengine(engine=engine, subdir=sub_subdir)
    wf.supercell_calc = calculate(
        wf.structure_supercell, engine=wf.supercell_engine, label="supercell_calc"
    )
    wf.substitutional_calc = calculate(
        wf.structure_with_substitute,
        engine=wf.substitutional_engine,
        label="substitutional_calc",
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
