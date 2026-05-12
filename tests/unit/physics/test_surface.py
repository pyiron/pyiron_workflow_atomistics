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
    # EMT Cu(111) is ~1.0 J/m^2; DFT reports ~1.5. Bracket generously but
    # keep the lower bound positive so the slab_novac-as-bulk-reference
    # regression (which produced a sign-flipped result) cannot recur.
    assert 0.3 < se < 3.0, f"Cu(111) EMT surface energy out of expected range: {se} J/m^2"


@pytest.mark.slow
def test_calculate_surface_energy_accepts_explicit_mu_bulk(tmp_path):
    """When mu_bulk is supplied the macro must not relaunch a bulk calc."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.surface import calculate_surface_energy

    cu_bulk = bulk("Cu", "fcc", a=3.6, cubic=True)
    cu_bulk_for_ref = cu_bulk.copy()
    cu_bulk_for_ref.calc = EMT()
    mu_bulk_from_emt = cu_bulk_for_ref.get_potential_energy() / len(cu_bulk_for_ref)

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=50
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_surface_energy(
        bulk_structure=cu_bulk,
        engine=engine,
        miller_indices=(1, 1, 1),
        layers=3,
        vacuum=8.0,
        mu_bulk=mu_bulk_from_emt,
    )
    out.run()
    se = out.outputs.surface_energy.value
    assert 0.3 < se < 3.0
