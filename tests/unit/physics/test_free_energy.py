"""Tests for pyiron_workflow_atomistics.physics.free_energy."""

from __future__ import annotations

import sys

import pytest


def test_require_calphy_raises_actionable_when_missing(monkeypatch):
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    monkeypatch.setitem(sys.modules, "calphy", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        _compat._require_calphy()
    assert "pip install 'pyiron_workflow_atomistics[free-energy]'" in str(exc.value)


def test_require_lammps_engine_raises_actionable_when_missing(monkeypatch):
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    monkeypatch.setitem(sys.modules, "pyiron_workflow_lammps", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        _compat._require_lammps_engine()
    assert "pip install 'pyiron_workflow_atomistics[free-energy]'" in str(exc.value)


def test_require_calphy_returns_module_when_present():
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    calphy = pytest.importorskip("calphy")
    assert _compat._require_calphy() is calphy


# ---------------------------------------------------------------------------
# LammpsPotential
# ---------------------------------------------------------------------------


def test_lammps_potential_required_fields():
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /path/to/Cu.eam.alloy Cu")
    assert pot.pair_style == "eam/alloy"
    assert pot.pair_coeff == "* * /path/to/Cu.eam.alloy Cu"
    assert pot.potential_file is None


def test_lammps_potential_optional_file():
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(
        pair_style="pace",
        pair_coeff="* * /path/to/pot.yace Cu",
        potential_file="/path/to/extra.txt",
    )
    assert pot.potential_file == "/path/to/extra.txt"


def test_lammps_potential_picklable():
    import pickle
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /path/to/Cu.eam.alloy Cu")
    restored = pickle.loads(pickle.dumps(pot))
    assert restored == pot


# ---------------------------------------------------------------------------
# FreeEnergyOutput
# ---------------------------------------------------------------------------


def test_free_energy_output_required_fields():
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="fe",
        reference_phase="solid",
        free_energy=-3.5,
        free_energy_error=0.01,
        temperature=300.0,
        pressure=0.0,
        n_atoms=108,
        elements=["Cu"],
        simfolder="/tmp/fe",
        report={"results": {"free_energy": -3.5}},
    )
    assert out.mode == "fe"
    assert out.free_energy == -3.5
    assert out.temperature_array is None
    assert out.melting_temperature is None


def test_free_energy_output_optional_fields_default_to_none():
    from dataclasses import fields, MISSING
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    optional_names = {
        "temperature_array",
        "free_energy_array",
        "pressure_array",
        "melting_temperature",
        "melting_temperature_error",
        "composition_path",
        "einstein_free_energy",
    }
    for f in fields(FreeEnergyOutput):
        if f.name in optional_names:
            assert f.default is None, f"{f.name} should default to None"


def test_free_energy_output_to_dict_round_trip():
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="ts",
        reference_phase="solid",
        free_energy=-3.5,
        free_energy_error=0.01,
        temperature=300.0,
        pressure=0.0,
        n_atoms=108,
        elements=["Cu"],
        simfolder="/tmp/ts",
        report={"results": {"free_energy": -3.5}},
    )
    d = out.to_dict()
    assert d["mode"] == "ts"
    assert d["temperature_array"] is None


def test_free_energy_output_picklable():
    import pickle
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="fe",
        reference_phase="solid",
        free_energy=-3.5,
        free_energy_error=0.01,
        temperature=300.0,
        pressure=0.0,
        n_atoms=108,
        elements=["Cu"],
        simfolder="/tmp/fe",
        report={},
    )
    restored = pickle.loads(pickle.dumps(out))
    assert restored.free_energy == -3.5


# ---------------------------------------------------------------------------
# _split_lammps_command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd, expected", [
    ("lmp", ("lmp", None, 1)),
    ("lmp -in in.lmp -log log.lammps", ("lmp", None, 1)),
    ("mpirun -np 4 lmp", ("lmp", "mpirun", 4)),
    ("mpiexec -n 8 lmp -in in.lmp -log log.lammps", ("lmp", "mpiexec", 8)),
    ("srun -n 16 lmp", ("lmp", "srun", 16)),
    ("mpirun --bind-to none -np 2 lmp",
     ("lmp", "mpirun --bind-to none", 2)),
    ("/opt/lammps/bin/lmp_mpi -in in.lmp",
     ("/opt/lammps/bin/lmp_mpi", None, 1)),
])
def test_split_lammps_command_valid(cmd, expected):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _split_lammps_command,
    )

    assert _split_lammps_command(cmd) == expected


@pytest.mark.parametrize("cmd", [
    "mpirun -np 4 lmp -unknown-flag x",
    "lmp -partition 2x2",
    "lmp -screen none",
])
def test_split_lammps_command_rejects_unknown_tokens(cmd):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _split_lammps_command,
    )

    with pytest.raises(ValueError, match="Unrecognized tokens"):
        _split_lammps_command(cmd)


def test_split_lammps_command_rejects_empty():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _split_lammps_command,
    )

    with pytest.raises(ValueError):
        _split_lammps_command("")
