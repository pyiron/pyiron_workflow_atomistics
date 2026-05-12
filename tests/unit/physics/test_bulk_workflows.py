"""Smoke tests for the macro workflows in ``physics/bulk.py``.

The existing ``tests/unit/test_bulk.py`` covers ``generate_structures``,
``equation_of_state``, and ``get_cubic_equil_lat_param`` in isolation. This
module exercises:

* ``optimise_cubic_lattice_parameter`` end-to-end (transitively covers
  ``eos_volume_scan``, ``evaluate_structures``, ``_extract_energies``,
  ``_extract_volumes``, ``_extract_structures``)
* ``generate_structures`` ``axes=None`` default branch
"""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


@pytest.mark.slow
def test_optimise_cubic_lattice_parameter_runs_end_to_end(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.bulk import (
        optimise_cubic_lattice_parameter,
    )

    # Slight off-equilibrium start so the EOS scan has a real minimum to find.
    structure = bulk("Cu", "fcc", a=3.5, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = optimise_cubic_lattice_parameter(
        structure=structure,
        name="Cu",
        crystalstructure="fcc",
        engine=engine,
        rattle_amount=0.0,
        strain_range=(-0.05, 0.05),
        num_points=5,
        eos_type="birchmurnaghan",
    )
    out = wf.run()

    a0 = out["a0"]
    B = out["B"]
    e0_per_atom = out["equil_energy_per_atom"]
    v0_per_atom = out["equil_volume_per_atom"]

    # EMT-Cu equilibrium lattice parameter is in the 3.6 Å range.
    assert 3.4 < a0 < 3.8, f"a0={a0} out of EMT-Cu range"
    # Bulk modulus positive and order-of-magnitude reasonable (EMT-Cu ~ 130 GPa).
    assert B > 0
    # Energies per atom finite and negative for cohesive solid.
    assert e0_per_atom < 0
    assert v0_per_atom > 0

    # The macro also exposes the raw EOS samples.
    assert len(out["energies"]) == 5
    assert len(out["volumes"]) == 5
    assert len(out["structures"]) == 5


def test_generate_structures_defaults_to_iso_axes():
    """``generate_structures(axes=None)`` should fall back to the iso branch."""
    from pyiron_workflow_atomistics.physics.bulk import generate_structures

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    structures = generate_structures.node_function(
        base_structure=structure, axes=None, strain_range=(-0.05, 0.05), num_points=3
    )
    assert len(structures) == 3
    # iso strain scales volume by (1+eps)^3; check the volumes are monotone.
    vols = [s.get_volume() for s in structures]
    assert vols[0] < vols[1] < vols[2]


def test_generate_structures_with_unknown_axis_warns():
    """An axis label outside {a, b, c, iso} should warn but not crash."""
    from pyiron_workflow_atomistics.physics.bulk import generate_structures

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    with pytest.warns(UserWarning, match="Unknown axis label"):
        out = generate_structures.node_function(
            base_structure=structure,
            axes=["a", "garbage"],
            strain_range=(-0.05, 0.05),
            num_points=3,
        )
    # axis 'a' still gets applied to each structure, so they're still produced.
    assert len(out) == 3
