"""Tests for pyiron_workflow_atomistics.physics.free_energy.harmonic."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_harmonic_free_energy_emt_al_2x2x2(tmp_path):
    pytest.importorskip("phonopy", reason="phonopy not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
        harmonic_free_energy,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = harmonic_free_energy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(0.0, 300.0),
        working_directory=str(tmp_path),
        subdir="harmonic",
    )
    out = wf.run()
    out = out["free_energy_output"] if isinstance(out, dict) else out

    assert out.mode == "harmonic"
    assert out.reference_phase == "solid"
    # ZPE > 0 at T=0
    assert out.free_energy_array[0] > 0.0
    # F decreases with T (entropy dominates)
    assert out.free_energy_array[1] < out.free_energy_array[0]
    # Entropy at T=0 is zero
    assert out.entropy_array[0] == pytest.approx(0.0, abs=1e-6)
