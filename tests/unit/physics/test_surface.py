"""Smoke test for physics.surface using ASEEngine + EMT (Cu)."""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


@pytest.mark.slow
def test_calculate_surface_energy_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.surface import calculate_surface_energy

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=50
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    cu_bulk = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = calculate_surface_energy(
        bulk_structure=cu_bulk,
        engine=engine,
        miller_indices=(1, 1, 1),
        layers=3,
        vacuum=8.0,
    )
    out.run()
    se = out.outputs.surface_energy.value
    assert se > 0
    assert se < 5  # Cu(111) ~ 1.5 J/m^2 — generous upper bound
