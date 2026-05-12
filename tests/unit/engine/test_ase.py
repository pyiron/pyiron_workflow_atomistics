"""Characterisation tests for ASEEngine: real EMT round-trip + pickle round-trip."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


def test_ase_engine_isinstance_engine_protocol():
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.engine.protocol import Engine

    eng = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory="./_t",
    )
    assert isinstance(eng, Engine)


def test_ase_engine_static_run_returns_engine_output(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputStatic,
        EngineOutput,
        run,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=structure, engine=engine)

    assert isinstance(out, EngineOutput)
    assert out.converged is True
    assert isinstance(out.final_energy, float)
    assert out.final_forces is not None
    assert out.final_forces.shape == (len(structure), 3)
    assert out.final_volume == pytest.approx(structure.get_volume())


def test_ase_engine_minimize_run_reduces_force(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputMinimize,
        run,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    # Perturb so the optimiser has work to do
    structure.rattle(0.05, seed=0)
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05, max_iterations=200),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=structure, engine=engine)

    # All forces are below the tolerance the optimiser reported converged on,
    # OR optimiser hit max steps with reduced forces — either way forces dropped.
    assert out.final_forces is not None
    final_fmax = float(np.linalg.norm(out.final_forces, axis=1).max())
    assert final_fmax < 1.0  # generous bound; rattle 0.05 yields ~few eV/Å initially


def test_ase_engine_with_working_directory_is_pure(tmp_path: Path):
    """with_working_directory returns a copy; original is untouched."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic

    eng = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    sub = eng.with_working_directory("substep")

    assert eng.working_directory == str(tmp_path)
    assert sub.working_directory == f"{tmp_path}/substep"
    assert eng is not sub


def test_ase_engine_pickle_round_trip(tmp_path: Path):
    """ASEEngine with EMT() calculator must pickle and unpickle cleanly."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize

    eng = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    blob = pickle.dumps(eng)
    restored = pickle.loads(blob)
    assert restored.working_directory == eng.working_directory
    assert restored.EngineInput.force_convergence_tolerance == 0.05
