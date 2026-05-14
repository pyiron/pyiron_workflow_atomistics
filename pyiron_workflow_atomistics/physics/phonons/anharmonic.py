"""phono3py FC3 + lattice thermal conductivity workflow.

The single user-facing entry point is :func:`calculate_phonon_thermal_conductivity`.
Everything else in this module is a private node or helper.
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _build_phono3py,
    _phonopy_to_ase,
)


def _check_polar_unsupported(
    *,
    born_charges: np.ndarray | None,
    epsilon_inf: np.ndarray | None,
) -> None:
    """Raise NotImplementedError if the caller asked for polar-material support.

    v1 is metals/non-polar only. The follow-up that adds NAC is tracked under
    "NAC / BORN effective charges" in the design spec.
    """
    if born_charges is not None or epsilon_inf is not None:
        raise NotImplementedError(
            "Non-analytic correction (BORN + ε∞) is not supported in v1; "
            "see the 'NAC / BORN effective charges' follow-up at the end of "
            "docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md."
        )


@pwf.as_function_node("fc3_supercells")
def _generate_fc3_supercells(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
    displacement_distance: float = 0.03,
    is_plusminus: str | bool = "auto",
    cutoff_pair_distance: float | None = None,
    number_of_snapshots: int | None = None,
    random_seed: int | None = None,
) -> list[Atoms]:
    """FC3 displaced supercells via phono3py.generate_displacements.

    Finite-difference path is used when number_of_snapshots is None;
    random-displacement path (and symfc fitting) lands in Task 11.
    """
    if number_of_snapshots is not None:
        # Filled in by Task 11. Until then, refuse random mode loudly so a
        # user who passes the kwarg too early gets a clear message.
        raise NotImplementedError(
            "Random-displacement FC3 sampling is added in a later task; "
            "set number_of_snapshots=None for the FD path."
        )
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )
    fc3_supercells = [
        _phonopy_to_ase(s) for s in ph3.supercells_with_displacements
    ]
    return fc3_supercells
