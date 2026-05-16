"""Unit tests for CalcInput* dataclasses, focused on the cell_relaxation
enum added in feat/cell-relaxation-enum."""

import warnings

import pytest

from pyiron_workflow_atomistics.engine.inputs import (
    CalcInputMinimize,
    CalcInputStatic,
    CalcInputMD,
)


# ---------------------------------------------------------------------------
# Legacy tests (preserved from original test_inputs.py)
# ---------------------------------------------------------------------------


def test_calc_input_static_is_empty_dataclass():
    inp = CalcInputStatic()
    assert hasattr(inp, "__dataclass_fields__")


def test_calc_input_minimize_defaults():
    inp = CalcInputMinimize()
    assert inp.force_convergence_tolerance > 0
    assert inp.energy_convergence_tolerance > 0
    assert inp.max_iterations > 0
    assert inp.cell_relaxation == "none"
    assert inp.relax_cell is False


def test_calc_input_md_renamed_field():
    """`thermostat_time_constant` replaces the old `temperature_damping_timescale`."""
    inp = CalcInputMD()
    assert hasattr(inp, "thermostat_time_constant")
    assert not hasattr(inp, "temperature_damping_timescale")


def test_calc_input_md_time_step_in_fs():
    """time_step is in fs (default ~1 fs), not ps."""
    inp = CalcInputMD()
    # 1 fs is the conventional ASE/LAMMPS default — anything < 100 means fs
    assert inp.time_step < 100.0


def test_calc_input_md_no_dropped_fields():
    """delta_temp and delta_press are removed."""
    inp = CalcInputMD()
    assert not hasattr(inp, "delta_temp")
    assert not hasattr(inp, "delta_press")


def test_calc_input_md_no_lammps_jargon_in_docstring():
    assert "LAMMPS units style" not in (CalcInputMD.__doc__ or "")


# ---------------------------------------------------------------------------
# New tests: cell_relaxation enum
# ---------------------------------------------------------------------------


class TestCellRelaxationField:
    def test_default_is_none(self):
        ci = CalcInputMinimize()
        assert ci.cell_relaxation == "none"

    @pytest.mark.parametrize(
        "value", ["none", "volume", "shape", "full"]
    )
    def test_accepts_all_four_modes(self, value):
        ci = CalcInputMinimize(cell_relaxation=value)
        assert ci.cell_relaxation == value


class TestRelaxCellShim:
    def test_relax_cell_true_maps_to_full(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ci = CalcInputMinimize(relax_cell=True)
        assert ci.cell_relaxation == "full"

    def test_relax_cell_false_maps_to_none(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ci = CalcInputMinimize(relax_cell=False)
        assert ci.cell_relaxation == "none"

    def test_relax_cell_property_reflects_cell_relaxation(self):
        assert CalcInputMinimize(cell_relaxation="full").relax_cell is True
        assert CalcInputMinimize(cell_relaxation="none").relax_cell is False
        assert CalcInputMinimize(cell_relaxation="volume").relax_cell is True
        assert CalcInputMinimize(cell_relaxation="shape").relax_cell is True

    def test_relax_cell_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="cell_relaxation"):
            CalcInputMinimize(relax_cell=True)

    def test_relax_cell_and_cell_relaxation_together_is_an_error(self):
        with pytest.raises(ValueError, match="both"):
            CalcInputMinimize(relax_cell=True, cell_relaxation="shape")
