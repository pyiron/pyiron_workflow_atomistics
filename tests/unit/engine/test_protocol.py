"""Tests for the Engine Protocol, EngineOutput dataclass, and run() node."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


def test_engine_protocol_is_runtime_checkable():
    """A duck-typed object with the right attrs should isinstance() as Engine."""
    from pyiron_workflow_atomistics.engine.protocol import Engine

    @dataclass
    class FakeEngine:
        working_directory: str = "fake"

        def get_calculate_fn(self, structure: Atoms):
            return (lambda **kw: None, {})

        def with_working_directory(self, subdir: str) -> "FakeEngine":
            return FakeEngine(working_directory=f"{self.working_directory}/{subdir}")

    assert isinstance(FakeEngine(), Engine)


def test_engine_output_is_dataclass_with_required_fields():
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    output = EngineOutput(
        final_structure=structure,
        final_energy=-1.23,
        converged=True,
    )
    assert output.final_energy == pytest.approx(-1.23)
    assert output.converged is True
    # Optional fields default to None
    assert output.final_forces is None
    assert output.final_stress is None
    assert output.final_stress_voigt is None


def test_engine_output_to_dict_round_trip():
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput

    output = EngineOutput(
        final_structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        final_energy=-1.23,
        converged=True,
        final_forces=np.zeros((1, 3)),
    )
    d = output.to_dict()
    assert d["final_energy"] == pytest.approx(-1.23)
    assert d["converged"] is True
    assert d["final_forces"].shape == (1, 3)


def test_run_node_dispatches_to_engine():
    """run(structure, engine) calls engine.get_calculate_fn and invokes the result."""
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput, run

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    sentinel_output = EngineOutput(
        final_structure=structure, final_energy=42.0, converged=True
    )

    @dataclass
    class StubEngine:
        working_directory: str = "."

        def get_calculate_fn(self, structure: Atoms):
            def fn(structure, **kwargs):
                return sentinel_output
            return fn, {"some": "kwarg"}

        def with_working_directory(self, subdir: str) -> "StubEngine":
            return StubEngine(working_directory=f"./{subdir}")

    out = run.node_function(structure=structure, engine=StubEngine())
    assert out is sentinel_output
