"""ASEEngine routes the new cell_relaxation enum to the right optimiser."""

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import (
    ASEEngine,
    CalcInputMinimize,
    calculate,
)


def _make_engine(tmp_path, mode):
    return ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation=mode, max_iterations=5),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )


class TestASEMinimizeRouting:
    def test_none_runs(self, tmp_path):
        eng = _make_engine(tmp_path, "none")
        out = calculate.node_function(structure=bulk("Cu", "fcc", a=3.6, cubic=True), engine=eng)
        assert out.final_energy is not None

    def test_full_runs(self, tmp_path):
        eng = _make_engine(tmp_path, "full")
        out = calculate.node_function(structure=bulk("Cu", "fcc", a=3.6, cubic=True), engine=eng)
        assert out.final_energy is not None

    @pytest.mark.parametrize("mode", ["volume", "shape"])
    def test_volume_and_shape_raise(self, tmp_path, mode):
        eng = _make_engine(tmp_path, mode)
        with pytest.raises(NotImplementedError, match=mode):
            calculate.node_function(
                structure=bulk("Cu", "fcc", a=3.6, cubic=True), engine=eng
            )
