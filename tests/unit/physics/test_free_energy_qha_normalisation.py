"""Regression tests for the per-atom normalisation of the QHA static-energy grid.

``_static_energies_per_volume`` feeds ``phonopy.qha.QHA(volumes=...,
electronic_energies=...)``. Those two arrays must share a basis. They did not:
``volumes`` was per atom while ``energies`` was the total cell energy from
``EngineOutput.final_energy``. Because B = V d²E/dV², mixing the two inflates
the fitted bulk modulus by exactly the unit-cell atom count.

Both tests use cubic fcc cells only, so they run inside the crystal-system
support the workflow already has.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.bulk import generate_structures
from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
    _static_energies_per_volume,
)

EV_PER_ANG3_TO_GPA = 160.21766208


def _emt_engine(tmp_path):
    return ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )


def _strained(base, num_points=5):
    return generate_structures(
        base_structure=base,
        axes=["iso"],
        strain_range=(-0.03, 0.03),
        num_points=num_points,
    )


def _bulk_modulus_GPa(energies, volumes):
    _, _, B = EquationOfState(
        np.asarray(volumes).tolist(),
        np.asarray(energies).tolist(),
        eos="birchmurnaghan",
    ).fit()
    return B * EV_PER_ANG3_TO_GPA


def test_static_energies_are_per_atom(tmp_path):
    """Returned energies must be per atom, matching the per-atom volumes.

    Pre-patch this fails by a factor of 4 for the 4-atom conventional cell.
    """
    base = bulk("Al", "fcc", a=4.05, cubic=True)
    structures = _strained(base)

    energies, volumes = _static_energies_per_volume(
        strained_structures=structures, engine=_emt_engine(tmp_path)
    )

    expected_energies = []
    expected_volumes = []
    for s in structures:
        ref = s.copy()
        ref.calc = EMT()
        expected_energies.append(ref.get_potential_energy() / len(s))
        expected_volumes.append(s.get_volume() / len(s))

    assert np.allclose(energies, expected_energies, rtol=1e-10)
    assert np.allclose(volumes, expected_volumes, rtol=1e-10)


def test_bulk_modulus_is_independent_of_unit_cell_size(tmp_path):
    """The same crystal in a bigger unit cell must give the same bulk modulus.

    This is the physically meaningful statement of the bug: nothing observable
    may depend on how many atoms the caller chose to put in the cell. With the
    per-cell/per-atom mismatch the two cells below disagree by exactly 2x
    (B scales as the atom count: 4 vs 8).
    """
    engine = _emt_engine(tmp_path)

    base_4 = bulk("Al", "fcc", a=4.05, cubic=True)
    base_8 = bulk("Al", "fcc", a=4.05, cubic=True).repeat((1, 1, 2))
    assert len(base_4) == 4 and len(base_8) == 8

    B = []
    for base in (base_4, base_8):
        energies, volumes = _static_energies_per_volume(
            strained_structures=_strained(base), engine=engine
        )
        B.append(_bulk_modulus_GPa(energies, volumes))

    assert B[0] == pytest.approx(B[1], rel=1e-6)
    # Sanity: EMT Al is a ~40 GPa material, not a ~160 GPa one. Guards against
    # both cells being wrong by the same factor, which the ratio check alone
    # would not catch.
    assert 25.0 < B[0] < 60.0
