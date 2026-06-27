from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase.build import bulk
from ase.constraints import FixAtoms


@pwf.as_function_node("structure")
def create_coexistence_supercell(
    element: str,
    crystalstructure: str | None = None,
    a: float | None = None,
    n_atoms: int = 8000,
):
    """Bulk repeated i x i x i so the atom count is closest to ``n_atoms/2``.

    Uses a cubic conventional cell where possible; hcp falls back to an
    orthorhombic cell (cubic=True is invalid for hcp), matching the notebook.
    """
    cs = (crystalstructure or "").lower()
    if cs == "hcp":
        base = bulk(element, crystalstructure, a=a, orthorhombic=True)
    elif crystalstructure is None:
        base = bulk(element, a=a)
    else:
        base = bulk(element, crystalstructure, a=a, cubic=True)
    target = n_atoms / 2.0
    reps = range(2, 30)
    cells = [base.repeat((i, i, i)) for i in reps]
    structure = cells[int(np.argmin([abs(len(c) - target) for c in cells]))]
    return structure


@pwf.as_function_node("structure")
def freeze_half(structure, axis: int = 2, fraction: float = 0.5):
    """Fix atoms whose scaled coordinate along ``axis`` is below ``fraction``."""
    s = structure.copy()
    scaled = s.get_scaled_positions()[:, axis]
    s.set_constraint(FixAtoms(indices=np.where(scaled < fraction)[0]))
    return s


@pwf.as_function_node("structure")
def unfreeze(structure):
    """Remove all constraints."""
    s = structure.copy()
    s.set_constraint()
    return s


@pwf.as_function_node("structure")
def strain_cell_along_z(structure, strain: float):
    """Scale cell vector c by ``strain`` (scale_atoms=True)."""
    s = structure.copy()
    cell = s.cell.copy()
    cell[2, 2] *= strain
    s.set_cell(cell, scale_atoms=True)
    return s
