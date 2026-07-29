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
        calculate,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate(structure=structure, engine=engine)

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
        calculate,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    # Perturb so the optimiser has work to do
    structure.rattle(0.05, seed=0)
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.05, max_iterations=200
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate(structure=structure, engine=engine)

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


def test_ase_engine_shared_calculator_is_thread_safe(tmp_path: Path):
    """Concurrent runs on one calculator instance must serialize, not race.

    pyiron_workflow >= 0.19 runs DAG-layer peers in threads by default, and
    with_working_directory() shares the calculator across sub-engines. Without
    the per-calculator lock this crashes inside ase.PrimitiveNeighborList or
    silently corrupts energies.
    """
    import threading

    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputStatic,
        calculate,
    )

    structures = [
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2),
        bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2),
    ]
    ref = []
    for i, structure in enumerate(structures):
        engine = ASEEngine(
            EngineInput=CalcInputStatic(),
            calculator=EMT(),
            working_directory=str(tmp_path / f"ref{i}"),
        )
        ref.append(calculate(structure=structure, engine=engine).final_energy)

    shared = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    n = 16
    results: list[float | None] = [None] * n
    errors: list[BaseException] = []

    def run(i: int) -> None:
        try:
            out = calculate(
                structure=structures[i % 2],
                engine=shared.with_working_directory(f"t{i}"),
            )
            results[i] = out.final_energy
        except BaseException as exc:  # noqa: BLE001 — record, assert in main thread
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent shared-calculator runs raised: {errors[:3]}"
    for i in range(n):
        assert results[i] == pytest.approx(ref[i % 2], abs=1e-10)
