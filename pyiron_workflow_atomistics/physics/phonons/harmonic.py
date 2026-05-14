"""phonopy FC2 helpers: supercell generation + ASE/PhonopyAtoms conversion.

Harmonic-observable nodes (band structure, DOS, free energy) land in
Task 13 once the synthesis node exists to expose them.
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.physics.phonons._compat import (
    require_phono3py,
    require_phonopy,
)


def _normalise_supercell_matrix(m: ArrayLike) -> np.ndarray:
    """Accept int / list[int] of length 3 / (3,3) ndarray; return (3,3) int."""
    arr = np.asarray(m)
    if arr.ndim == 0:
        return (int(arr) * np.eye(3, dtype=int))
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(
                f"supercell_matrix 1d shape must be (3,), got {arr.shape}"
            )
        return np.diag(arr.astype(int))
    if arr.ndim == 2:
        if arr.shape != (3, 3):
            raise ValueError(
                f"supercell_matrix 2d shape must be (3,3), got {arr.shape}"
            )
        return arr.astype(int)
    raise ValueError(f"supercell_matrix must be int / (3,) / (3,3); got {arr.shape}")


def _ase_to_phonopy(ase_atoms: Atoms):
    """Convert ASE Atoms → PhonopyAtoms (phonopy's own structure type)."""
    require_phonopy()  # noqa: F841 — only needed for the import side-effect
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=list(ase_atoms.get_chemical_symbols()),
        positions=ase_atoms.get_positions(),
        cell=np.asarray(ase_atoms.get_cell()),
        masses=ase_atoms.get_masses(),
    )


def _phonopy_to_ase(phonopy_atoms) -> Atoms:
    """Convert PhonopyAtoms → ASE Atoms. pbc=True (supercells are always periodic)."""
    return Atoms(
        symbols=list(phonopy_atoms.symbols),
        positions=np.asarray(phonopy_atoms.positions),
        cell=np.asarray(phonopy_atoms.cell),
        pbc=True,
        masses=np.asarray(phonopy_atoms.masses),
    )


def _build_phono3py(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
):
    """Construct a Phono3py instance with both supercell matrices.

    Note: phono3py's `supercell_matrix` is the FC3 supercell and
    `phonon_supercell_matrix` is the FC2 supercell. We expose them under the
    physics-level names (`fc2_*`, `fc3_*`) and translate here.
    """
    phono3py_mod = require_phono3py()
    unitcell = _ase_to_phonopy(structure)
    return phono3py_mod.Phono3py(
        unitcell=unitcell,
        supercell_matrix=_normalise_supercell_matrix(fc3_supercell_matrix),
        phonon_supercell_matrix=_normalise_supercell_matrix(fc2_supercell_matrix),
    )


@pwf.as_function_node("fc2_supercells")
def _generate_fc2_supercells(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    displacement_distance: float = 0.03,
    is_plusminus: str | bool = "auto",
) -> list[Atoms]:
    """FC2 displaced supercells via phono3py.generate_fc2_displacements (FD).

    Returns a list of ASE Atoms. The same kwargs reconstruct an identical
    Phono3py object inside the synthesis node — FD is deterministic in
    structure + supercell + distance + symmetry.
    """
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc2_supercell_matrix,  # placeholder; FC3 grid not used here
    )
    ph3.generate_fc2_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    fc2_supercells = [
        _phonopy_to_ase(s) for s in ph3.phonon_supercells_with_displacements
    ]
    return fc2_supercells
