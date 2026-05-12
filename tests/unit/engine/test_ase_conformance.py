"""In-tree conformance test: prove EngineConformanceTests is correct
by running it against the canonical ASEEngine (EMT calculator)."""

from __future__ import annotations

from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.testing import EngineConformanceTests


class TestASEEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        return ASEEngine(
            EngineInput=CalcInputStatic(),
            calculator=EMT(),
            working_directory=str(tmp_path),
        )
