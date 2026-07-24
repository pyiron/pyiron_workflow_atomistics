"""Point-defect formation energies (vacancy, substitutional)."""

from __future__ import annotations

import flowrep as fr
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import (
    Engine,
    calculate,
    subengine,
)
from pyiron_workflow_atomistics.structure.defects import (
    create_vacancy,
    substitutional_swap,
)
from pyiron_workflow_atomistics.structure.transform import (
    create_supercell_with_min_dimensions,
)


@pwf.atomic
def _count_atoms(structure: Atoms) -> int:
    n_atoms = len(structure)
    return n_atoms


@pwf.atomic
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


@pwf.workflow
def get_vacancy_formation_energy(
    structure: Atoms,
    engine: Engine,
    remove_atom_index: int = 0,
    min_dimensions: list | tuple = (12, 12, 12),
    vacancy_subdir: str = "vacancy",
    supercell_subdir: str = "supercell",
):
    """Standard vacancy formation energy macro.

    Examples
    --------
    See ``notebooks/vacancy_formation_energy.ipynb``.
    """
    structure_supercell = create_supercell_with_min_dimensions(
        structure, min_dimensions=min_dimensions
    )
    structure_with_vacancy = create_vacancy(
        structure_supercell, remove_atom_index=remove_atom_index
    )
    supercell_engine = subengine(engine=engine, subdir=supercell_subdir)
    vacancy_engine = subengine(engine=engine, subdir=vacancy_subdir)
    supercell_calc = calculate(structure_supercell, engine=supercell_engine)
    vacancy_calc = calculate(structure_with_vacancy, engine=vacancy_engine)
    n_atoms_supercell = _count_atoms(structure_supercell)

    supercell_final_energy = fr.std.get_attr(supercell_calc, "final_energy")
    vacancy_final_energy = fr.std.get_attr(vacancy_calc, "final_energy")

    vacancy_formation_energy = calculate_vacancy_formation_energy(
        vacancy_energy=vacancy_final_energy,
        supercell_energy=supercell_final_energy,
        n_atoms_supercell=n_atoms_supercell,
    )
    return supercell_calc, vacancy_calc, vacancy_formation_energy


@pwf.atomic
def _substitutional_formation_energy(E_sub, E_bulk, mu_solute, mu_host):
    E_f = E_sub - E_bulk - mu_solute + mu_host
    return E_f


@pwf.workflow
def get_substitutional_formation_energy(
    structure: Atoms,
    engine: Engine,
    defect_site: int = 0,
    new_symbol: str = "Ni",
    mu_solute: float = 0.0,
    mu_host: float = 0.0,
    min_dimensions: list | tuple = (12, 12, 12),
    sub_subdir: str = "substitutional",
    supercell_subdir: str = "supercell",
):
    """Dilute substitutional formation energy:
    ``E_f = E_sub - E_supercell - mu_solute + mu_host``.
    """
    structure_supercell = create_supercell_with_min_dimensions(
        structure, min_dimensions=min_dimensions
    )
    structure_with_substitute = substitutional_swap(
        structure_supercell,
        defect_site=defect_site,
        new_symbol=new_symbol,
    )
    supercell_engine = subengine(engine=engine, subdir=supercell_subdir)
    substitutional_engine = subengine(engine=engine, subdir=sub_subdir)
    supercell_calc = calculate(structure_supercell, engine=supercell_engine)
    substitutional_calc = calculate(
        structure_with_substitute, engine=substitutional_engine
    )

    supercell_final_energy = fr.std.get_attr(supercell_calc, "final_energy")
    substitutional_final_energy = fr.std.get_attr(substitutional_calc, "final_energy")

    substitutional_formation_energy = _substitutional_formation_energy(
        E_sub=substitutional_final_energy,
        E_bulk=supercell_final_energy,
        mu_solute=mu_solute,
        mu_host=mu_host,
    )
    return (
        supercell_calc,
        substitutional_calc,
        substitutional_formation_energy,
    )
