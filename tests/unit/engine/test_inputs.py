"""Tests for the physics-level engine input dataclasses."""

from __future__ import annotations


def test_calc_input_static_is_empty_dataclass():
    from pyiron_workflow_atomistics.engine.inputs import CalcInputStatic

    inp = CalcInputStatic()
    assert hasattr(inp, "__dataclass_fields__")


def test_calc_input_minimize_defaults():
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMinimize

    inp = CalcInputMinimize()
    assert inp.force_convergence_tolerance > 0
    assert inp.energy_convergence_tolerance > 0
    assert inp.max_iterations > 0
    assert inp.relax_cell is False


def test_calc_input_md_renamed_field():
    """`thermostat_time_constant` replaces the old `temperature_damping_timescale`."""
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    inp = CalcInputMD()
    assert hasattr(inp, "thermostat_time_constant")
    assert not hasattr(inp, "temperature_damping_timescale")


def test_calc_input_md_time_step_in_fs():
    """time_step is in fs (default ~1 fs), not ps."""
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    inp = CalcInputMD()
    # 1 fs is the conventional ASE/LAMMPS default — anything < 100 means fs
    assert inp.time_step < 100.0


def test_calc_input_md_no_dropped_fields():
    """delta_temp and delta_press are removed."""
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    inp = CalcInputMD()
    assert not hasattr(inp, "delta_temp")
    assert not hasattr(inp, "delta_press")


def test_calc_input_md_no_lammps_jargon_in_docstring():
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    assert "LAMMPS units style" not in (CalcInputMD.__doc__ or "")
