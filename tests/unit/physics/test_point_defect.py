"""Smoke test for physics.point_defect using ASEEngine + EMT (Cu)."""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


@pytest.mark.slow
def test_vacancy_formation_energy_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.point_defect import (
        get_vacancy_formation_energy,
    )

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=50
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = get_vacancy_formation_energy(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        engine=engine,
        min_dimensions=[8, 8, 8],
    )
    wf.run()
    e_f = wf.outputs.vacancy_formation_energy.value
    assert 0.5 < e_f < 2.5  # EMT Cu vacancy ~ 0.9–1.3 eV


@pytest.mark.slow
def test_substitutional_formation_energy_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.point_defect import (
        get_substitutional_formation_energy,
    )

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=50
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = get_substitutional_formation_energy(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        engine=engine,
        new_symbol="Ni",
        min_dimensions=[8, 8, 8],
    )
    wf.run()
    e_f = wf.outputs.substitutional_formation_energy.value
    assert e_f is not None
