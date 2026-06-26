import numpy as np
from ase.constraints import FixAtoms

from pyiron_workflow_atomistics.physics.melting.structures import (
    create_coexistence_supercell,
    freeze_half,
    strain_cell_along_z,
    unfreeze,
)


def test_supercell_targets_half_n_atoms():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=8000)
    assert len(s) == 4000  # fcc cubic = 4 atoms; 4*10^3 = 4000 ~ 8000/2


def test_freeze_half_fixes_lower_half():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=2000)
    frozen = freeze_half.node_function(s, axis=2, fraction=0.5)
    cons = [c for c in frozen.constraints if isinstance(c, FixAtoms)]
    assert len(cons) == 1
    fixed = set(cons[0].get_indices())
    zsc = frozen.get_scaled_positions()[:, 2]
    expected = set(np.where(zsc < 0.5)[0])
    assert fixed == expected


def test_unfreeze_clears_constraints():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=2000)
    assert len(unfreeze.node_function(freeze_half.node_function(s)).constraints) == 0


def test_strain_scales_only_c():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=2000)
    c0 = s.cell[2, 2]
    strained = strain_cell_along_z.node_function(s, 1.05)
    assert abs(strained.cell[2, 2] - 1.05 * c0) < 1e-8
    assert abs(strained.cell[0, 0] - s.cell[0, 0]) < 1e-8
