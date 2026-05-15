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


# ---------------------------------------------------------------------------
# _validate_engine_only_command
# ---------------------------------------------------------------------------


lammps_engine = pytest.importorskip("pyiron_workflow_lammps.engine")


def _make_minimal_engine():
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_lammps.engine import LammpsEngine

    return LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")


def test_validate_engine_only_command_passes_for_minimal_engine():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    _validate_engine_only_command(eng)  # should not raise


def test_validate_engine_only_command_rejects_raw_script():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.raw_script = "run 1000"
    with pytest.raises(ValueError, match=r"raw_script"):
        _validate_engine_only_command(eng)


def test_validate_engine_only_command_rejects_path_to_model():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.path_to_model = "/real/model"
    with pytest.raises(ValueError, match=r"path_to_model"):
        _validate_engine_only_command(eng)


def test_validate_engine_only_command_rejects_pair_style():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.input_script_pair_style = "eam/alloy"
    with pytest.raises(ValueError, match=r"input_script_pair_style"):
        _validate_engine_only_command(eng)


def test_validate_engine_only_command_command_can_be_changed():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.command = "mpirun -np 8 lmp"
    _validate_engine_only_command(eng)  # should not raise


def test_validate_engine_only_command_working_directory_carveout():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.working_directory = "/somewhere/else"
    _validate_engine_only_command(eng)  # working_directory is in the carve-out set


# ---------------------------------------------------------------------------
# _validate_structure
# ---------------------------------------------------------------------------


def test_validate_structure_accepts_cubic_bulk(fcc_al_atoms):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    _validate_structure(fcc_al_atoms)  # should not raise


def test_validate_structure_rejects_empty():
    from ase import Atoms
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    with pytest.raises(ValueError, match=r"empty"):
        _validate_structure(Atoms())


def test_validate_structure_rejects_mixed_pbc(fcc_al_atoms):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    a = fcc_al_atoms.copy()
    a.pbc = (True, True, False)
    with pytest.raises(ValueError, match=r"PBC"):
        _validate_structure(a)


def test_validate_structure_rejects_zero_volume(fcc_al_atoms):
    from ase import Atoms
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    a = Atoms("Cu", positions=[[0, 0, 0]], cell=[[0, 0, 0]] * 3, pbc=True)
    with pytest.raises(ValueError, match=r"non-positive volume"):
        _validate_structure(a)


# ---------------------------------------------------------------------------
# _build_calphy_calculation — fe mode
# ---------------------------------------------------------------------------


def test_build_calphy_calculation_fe_basic(tmp_path, fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _build_calphy_calculation,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    eng = _make_minimal_engine()
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff="* * /tmp/Al.eam.alloy Al",
    )
    calc = _build_calphy_calculation(
        mode="fe",
        structure=fcc_al_atoms,
        potential=pot,
        lammps_engine=eng,
        working_directory=str(tmp_path),
        temperature=300.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
        n_iterations=1,
        npt=True,
        equilibration_control="nose-hoover",
    )
    assert calc.mode == "fe"
    assert calc.element == ["Al"]
    assert calc.pair_style == ["eam/alloy"]
    assert calc.pair_coeff == ["* * /tmp/Al.eam.alloy Al"]
    assert calc.script_mode is True
    assert calc.lammps_executable == "lmp"
    assert calc.mpi_executable is None
    assert calc.reference_phase == "solid"
    assert calc.npt is True
    # data file must exist in working_directory
    assert (tmp_path / "lammps.data").exists()
    assert str(tmp_path / "lammps.data") in calc.lattice
    assert calc.file_format == "lammps-data"


def test_build_calphy_calculation_fe_passes_mpi_command(tmp_path, fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _build_calphy_calculation,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    eng = _make_minimal_engine()
    eng.command = "mpirun -np 8 lmp"
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    calc = _build_calphy_calculation(
        mode="fe",
        structure=fcc_al_atoms,
        potential=pot,
        lammps_engine=eng,
        working_directory=str(tmp_path),
        temperature=300.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
        n_iterations=1,
        npt=True,
        equilibration_control="nose-hoover",
    )
    assert calc.lammps_executable == "lmp"
    assert calc.mpi_executable == "mpirun"
    # cores is captured on queue.cores
    assert calc.queue.cores == 8


def test_build_calphy_calculation_writes_data_file_matches_structure(
    tmp_path, fcc_al_atoms
):
    pytest.importorskip("calphy")
    from ase.io import read as ase_read
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _build_calphy_calculation,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    eng = _make_minimal_engine()
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    _build_calphy_calculation(
        mode="fe",
        structure=fcc_al_atoms,
        potential=pot,
        lammps_engine=eng,
        working_directory=str(tmp_path),
        temperature=300.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
        n_iterations=1,
        npt=True,
        equilibration_control="nose-hoover",
    )
    round_trip = ase_read(
        str(tmp_path / "lammps.data"),
        format="lammps-data",
        style="atomic",
    )
    import numpy as np
    np.testing.assert_allclose(round_trip.get_positions(),
                               fcc_al_atoms.get_positions(),
                               atol=1e-8)
    np.testing.assert_allclose(round_trip.get_cell(),
                               fcc_al_atoms.get_cell(),
                               atol=1e-8)


# ---------------------------------------------------------------------------
# _pack_free_energy_output
# ---------------------------------------------------------------------------


def test_pack_free_energy_output_fe_minimal(fcc_al_atoms, tmp_path):
    from types import SimpleNamespace
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _pack_free_energy_output,
    )

    fake_report = {
        "results": {
            "free_energy": -3.5,
            "error": 0.01,
            "einstein_crystal": -3.7,
        },
        "input": {
            "temperature": 300.0,
            "pressure": 0.0,
        },
    }
    fake_job = SimpleNamespace(simfolder=str(tmp_path))
    out = _pack_free_energy_output(
        mode="fe",
        job=fake_job,
        report=fake_report,
        simfolder=str(tmp_path),
        structure=fcc_al_atoms,
        reference_phase="solid",
        temperature=300.0,
        pressure=0.0,
    )
    assert out.mode == "fe"
    assert out.free_energy == -3.5
    assert out.free_energy_error == 0.01
    assert out.einstein_free_energy == -3.7
    assert out.temperature == 300.0
    assert out.pressure == 0.0
    assert out.n_atoms == len(fcc_al_atoms)
    assert out.elements == ["Al"]
    assert out.simfolder == str(tmp_path)
    assert out.report is fake_report


def test_pack_free_energy_output_melting_temperature(fcc_al_atoms, tmp_path):
    from types import SimpleNamespace
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _pack_free_energy_output,
    )

    fake_job = SimpleNamespace(simfolder=str(tmp_path), tm=1357.0, dtm=15.0)
    fake_report = {"results": {"free_energy": 0.0, "error": 0.0}}
    out = _pack_free_energy_output(
        mode="melting_temperature",
        job=fake_job,
        report=fake_report,
        simfolder=str(tmp_path),
        structure=fcc_al_atoms,
        reference_phase="both",
        temperature=1357.0,
        pressure=0.0,
    )
    assert out.melting_temperature == 1357.0
    assert out.melting_temperature_error == 15.0


# ---------------------------------------------------------------------------
# _run_calphy_job — mocked
# ---------------------------------------------------------------------------


def test_run_calphy_job_dispatches_and_reads_report(monkeypatch, tmp_path):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.physics.free_energy import _calphy_adapter

    # Fake job with a simfolder containing a report.yaml
    import yaml
    sim = tmp_path / "sim"
    sim.mkdir()
    fake_report = {"results": {"free_energy": -3.5, "error": 0.01}}
    (sim / "report.yaml").write_text(yaml.safe_dump(fake_report))

    from types import SimpleNamespace
    fake_job = SimpleNamespace(simfolder=str(sim),
                               calc=SimpleNamespace(mode="fe"))

    captured = {}

    def fake_setup(calc):
        captured["setup"] = True
        return fake_job

    def fake_run(job):
        captured["run"] = True
        return job

    monkeypatch.setattr(_calphy_adapter, "_setup_calculation", fake_setup)
    monkeypatch.setattr(_calphy_adapter, "_run_calculation", fake_run)

    fake_calc = SimpleNamespace(mode="fe")
    job, report = _calphy_adapter._run_calphy_job(fake_calc)
    assert captured == {"setup": True, "run": True}
    assert report == fake_report
    assert job is fake_job


# ---------------------------------------------------------------------------
# _load_rs_curve
# ---------------------------------------------------------------------------


def test_load_rs_curve_temperature(tmp_path):
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _load_rs_curve,
    )

    # calphy writes "temperature_sweep.dat" with cols [T, F]
    data = np.array([[100.0, -3.50],
                     [200.0, -3.55],
                     [300.0, -3.60]])
    np.savetxt(tmp_path / "temperature_sweep.dat", data)

    t, f = _load_rs_curve(str(tmp_path))
    np.testing.assert_allclose(t, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(f, [-3.50, -3.55, -3.60])


def test_load_rs_curve_pressure(tmp_path):
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _load_rs_curve,
    )

    data = np.array([[0.0, -3.50],
                     [5000.0, -3.45],
                     [10000.0, -3.40]])
    np.savetxt(tmp_path / "pressure_sweep.dat", data)

    p, f = _load_rs_curve(str(tmp_path), axis="pressure")
    np.testing.assert_allclose(p, [0.0, 5000.0, 10000.0])
    np.testing.assert_allclose(f, [-3.50, -3.45, -3.40])


def test_load_rs_curve_missing_raises(tmp_path):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _load_rs_curve,
    )

    with pytest.raises(FileNotFoundError):
        _load_rs_curve(str(tmp_path))


# ---------------------------------------------------------------------------
# free_energy public node — Tier 1 (no calphy / no LAMMPS)
# ---------------------------------------------------------------------------


def test_free_energy_node_raises_when_calphy_missing(monkeypatch, fcc_al_atoms):
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    monkeypatch.setitem(sys.modules, "calphy", None)
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ModuleNotFoundError, match=r"pip install"):
        free_energy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            reference_phase="solid",
        )


def test_free_energy_node_rejects_non_default_engine_field(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    eng.raw_script = "run 1000"
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"raw_script"):
        free_energy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            reference_phase="solid",
        )


def test_free_energy_node_rejects_non_periodic_structure(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    s = fcc_al_atoms.copy()
    s.pbc = (True, True, False)
    with pytest.raises(ValueError, match=r"PBC"):
        free_energy.node_function(
            structure=s,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            reference_phase="solid",
        )


def test_free_energy_node_restores_cwd_on_error(monkeypatch, tmp_path,
                                                fcc_al_atoms):
    pytest.importorskip("calphy")
    import os
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy import _calphy_adapter
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")

    def boom(_calc):
        raise RuntimeError("calphy exploded mid-run")

    monkeypatch.setattr(_calphy_adapter, "_run_calphy_job", boom)
    monkeypatch.chdir(tmp_path)
    cwd_before = os.getcwd()
    with pytest.raises(RuntimeError, match="calphy exploded"):
        free_energy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            working_directory=str(tmp_path),
            temperature=300.0,
            reference_phase="solid",
        )
    assert os.getcwd() == cwd_before


# ---------------------------------------------------------------------------
# Tier 2 integration — needs calphy + lmp binary
# ---------------------------------------------------------------------------

import shutil
from pathlib import Path

LAMMPS_BIN = shutil.which("lmp") or shutil.which("lmp_mpi")
RESOURCES = Path(__file__).resolve().parent.parent.parent / "resources" / "free_energy"

requires_lammps = pytest.mark.skipif(
    LAMMPS_BIN is None or not (RESOURCES / "Cu01.eam.alloy").exists(),
    reason="needs lmp binary on PATH and tests/resources/free_energy/Cu01.eam.alloy",
)


@requires_lammps
def test_free_energy_fcc_cu_smoke(tmp_path):
    pytest.importorskip("calphy")
    import os
    from ase.build import bulk
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    cu = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff=f"* * {RESOURCES / 'Cu01.eam.alloy'} Cu",
    )
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command=LAMMPS_BIN)

    out = free_energy.node_function(
        structure=cu,
        lammps_engine=eng,
        potential=pot,
        working_directory=str(tmp_path),
        subdir="fe",
        temperature=100.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
    )
    assert out.mode == "fe"
    assert out.reference_phase == "solid"
    assert out.n_atoms == 108
    assert out.elements == ["Cu"]
    assert -4.5 < out.free_energy < -3.0  # Cu EAM, eV/atom @ 100 K
    assert out.free_energy_error >= 0
    assert os.path.isabs(out.simfolder)
    assert (tmp_path / "fe" / "report.yaml").exists()


# ---------------------------------------------------------------------------
# reversible_scaling_temperature
# ---------------------------------------------------------------------------


def test_reversible_scaling_temperature_validates_tuple_shape(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        reversible_scaling_temperature,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"temperature_range"):
        reversible_scaling_temperature.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature_range=300.0,  # scalar — must be 2-tuple
            reference_phase="solid",
        )


@requires_lammps
def test_reversible_scaling_temperature_returns_curve(tmp_path):
    pytest.importorskip("calphy")
    from ase.build import bulk
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        reversible_scaling_temperature,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    cu = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff=f"* * {RESOURCES / 'Cu01.eam.alloy'} Cu",
    )
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command=LAMMPS_BIN)

    out = reversible_scaling_temperature.node_function(
        structure=cu,
        lammps_engine=eng,
        potential=pot,
        working_directory=str(tmp_path),
        subdir="ts",
        temperature_range=(100.0, 300.0),
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
    )
    assert out.mode == "ts"
    assert out.temperature_array is not None
    assert out.free_energy_array is not None
    assert out.temperature_array.shape == out.free_energy_array.shape
    import numpy as np
    assert np.all(np.diff(out.temperature_array) >= 0)


# ---------------------------------------------------------------------------
# reversible_scaling_pressure
# ---------------------------------------------------------------------------


def test_reversible_scaling_pressure_validates_tuple_shape(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        reversible_scaling_pressure,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"pressure_range"):
        reversible_scaling_pressure.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            pressure_range=1000.0,  # scalar — must be 2-tuple
            reference_phase="solid",
        )


# ---------------------------------------------------------------------------
# melting_temperature
# ---------------------------------------------------------------------------


def test_melting_temperature_validates_positive_guess(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import melting_temperature
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"positive"):
        melting_temperature.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature_guess=-100.0,
        )


@requires_lammps
@pytest.mark.slow
def test_melting_temperature_runs(tmp_path):
    pytest.importorskip("calphy")
    from ase.build import bulk
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import melting_temperature
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    cu = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff=f"* * {RESOURCES / 'Cu01.eam.alloy'} Cu",
    )
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command=LAMMPS_BIN)
    out = melting_temperature.node_function(
        structure=cu,
        lammps_engine=eng,
        potential=pot,
        working_directory=str(tmp_path),
        temperature_guess=1300.0,
        step=400,
        max_attempts=3,
        n_equilibration_steps=2000,
        n_switching_steps=2000,
    )
    assert out.mode == "melting_temperature"
    assert out.reference_phase == "both"
    assert 800.0 < out.melting_temperature < 2000.0


# ---------------------------------------------------------------------------
# alchemy
# ---------------------------------------------------------------------------


def test_alchemy_requires_target_potential_strings(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import alchemy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/A.eam.alloy Al")
    with pytest.raises(ValueError, match=r"pair_style_target"):
        alchemy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            pair_style_target=None,
            pair_coeff_target=None,
        )


# ---------------------------------------------------------------------------
# composition_scaling
# ---------------------------------------------------------------------------


def test_composition_scaling_requires_output_composition(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        composition_scaling,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/AB.eam.alloy A B")
    with pytest.raises(ValueError, match=r"output_chemical_composition"):
        composition_scaling.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            output_chemical_composition=None,
        )
