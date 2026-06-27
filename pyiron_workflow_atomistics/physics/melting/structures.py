from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pyiron_workflow as pwf
from ase.build import bulk
from ase.constraints import FixAtoms

# Conventional-cell atom counts and ideal volume-per-atom prefactors used to map a
# relaxed reference atomic volume onto a seed lattice constant for a target phase.
# ASE ``bulk`` only knows ``a`` for an element's reference state, so seeding fcc/bcc/
# hcp candidates for a scan needs an explicit estimate (cell-relax refines it after).
_ATOMS_PER_CUBIC = {"fcc": 4, "bcc": 2, "sc": 1}


@pwf.as_function_node("a")
def estimate_lattice_constant(element, engine, crystalstructure):
    """Seed lattice constant ``a`` for ``crystalstructure`` from a relaxed reference.

    Relaxes the element's ASE reference-state bulk (cell relaxed) to get the
    per-atom volume, then inverts the ideal volume-per-atom relation for the
    requested phase. fcc/bcc/sc use the cubic conventional cell; hcp assumes the
    ideal c/a = 1.633. Any other phase falls back to the fcc map. This is only a
    seed — ``calculate``'s cell relaxation corrects it.
    """
    from pyiron_workflow_atomistics.engine import CalcInputMinimize, calculate

    ref = bulk(element)
    relax_engine = replace(
        engine, EngineInput=CalcInputMinimize(relax_cell=True)
    ).with_working_directory("a_ref")
    relaxed = calculate.node_function(ref, engine=relax_engine).final_structure
    v_atom = relaxed.get_volume() / len(relaxed)
    cs = (crystalstructure or "fcc").lower()
    if cs == "hcp":
        # V_atom = (sqrt(3)/2) * a^2 * c / 2, with c = 1.633 a
        a = (v_atom / (math.sqrt(3.0) / 2.0 * 1.633)) ** (1.0 / 3.0)
    else:
        n = _ATOMS_PER_CUBIC.get(cs, 4)
        a = (n * v_atom) ** (1.0 / 3.0)
    return float(a)


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
