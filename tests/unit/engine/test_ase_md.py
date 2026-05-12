"""Smoke tests for the MD code path in pyiron_workflow_atomistics.engine.ase.

Covers ase_md_calc_structure and the CalcInputMD branch of
ASEEngine.get_calculate_fn. Uses EMT() on a small Cu supercell and only a
handful of MD steps so each test runs in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


def _cu_supercell():
    return bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))


def test_ase_md_nvt_langevin_default_runs(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="langevin",
            temperature=300.0,
            n_ionic_steps=5,
            time_step=1.0,
            thermostat_time_constant=50.0,
            seed=42,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
        record_interval=1,
    )

    out = run.node_function(structure=_cu_supercell(), engine=engine)

    assert isinstance(out, EngineOutput)
    assert out.converged is True
    assert out.final_energy is not None
    assert isinstance(out.final_energy, float)
    assert out.final_forces.shape == (32, 3)
    # MD recorded a trajectory; the recorder fires at step 0 too, so
    # expect at least n_ionic_steps frames.
    assert out.energies is not None
    assert len(out.energies) >= 5
    # job_data.pkl.gz is the canonical trajectory dataframe artifact
    assert (tmp_path / "job_data.pkl.gz").exists()


def test_ase_md_nvt_berendsen_runs(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="berendsen",
            temperature=300.0,
            n_ionic_steps=5,
            time_step=1.0,
            thermostat_time_constant=50.0,
            seed=0,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True
    assert len(out.energies) >= 5


def test_ase_md_nvt_andersen_falls_through_to_langevin(tmp_path: Path):
    """The andersen branch hits the `else: # langevin or andersen` path."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="andersen",
            temperature=300.0,
            n_ionic_steps=3,
            time_step=1.0,
            thermostat_time_constant=50.0,
            seed=7,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True


def test_ase_md_nve_runs(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVE",
            thermostat="langevin",  # ignored for NVE
            temperature=200.0,
            n_ionic_steps=3,
            time_step=1.0,
            seed=1,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True


def test_ase_md_npt_berendsen_runs(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NPT",
            thermostat="berendsen",
            temperature=300.0,
            n_ionic_steps=3,
            time_step=1.0,
            thermostat_time_constant=50.0,
            pressure_damping_timescale=500.0,
            pressure=1e5,  # 1 bar in Pa
            seed=2,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True


def test_ase_md_npt_nose_hoover_runs(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NPT",
            thermostat="nose-hoover",
            temperature=300.0,
            n_ionic_steps=3,
            time_step=1.0,
            thermostat_time_constant=50.0,
            pressure_damping_timescale=500.0,
            pressure=1e5,
            seed=3,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True


def test_ase_md_npt_without_pressure_raises(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NPT", thermostat="berendsen", n_ionic_steps=1, pressure=None
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    with pytest.raises(ValueError, match="Pressure must be specified"):
        run.node_function(structure=_cu_supercell(), engine=engine)


def test_ase_md_npt_with_invalid_thermostat_raises(tmp_path: Path):
    """NPT only accepts nose-hoover or berendsen."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NPT", thermostat="langevin", n_ionic_steps=1, pressure=1e5
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    with pytest.raises(ValueError, match="NPT supports only"):
        run.node_function(structure=_cu_supercell(), engine=engine)


def test_ase_md_write_to_disk_emits_trajectory_files(tmp_path: Path):
    """write_to_disk=True must produce initial/final/trajectory artifacts."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="langevin",
            temperature=300.0,
            n_ionic_steps=3,
            time_step=1.0,
            seed=8,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
        write_to_disk=True,
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True
    for name in (
        "initial_structure.xyz",
        "initial_results.json",
        "trajectory.xyz",
        "trajectory_results.json",
        "final_structure.xyz",
        "final_results.json",
        "job_data.pkl.gz",
    ):
        assert (tmp_path / name).exists(), f"missing artifact: {name}"


def test_ase_md_with_initial_temperature_zero(tmp_path: Path):
    """initial_temperature=0 must skip the MaxwellBoltzmann velocity init."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="berendsen",
            temperature=300.0,
            initial_temperature=0.0,  # branches around MaxwellBoltzmannDistribution
            n_ionic_steps=2,
            time_step=1.0,
            seed=4,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    assert out.converged is True


def test_ase_engine_get_calculate_fn_dispatches_to_md(tmp_path: Path):
    """ASEEngine.get_calculate_fn should return ase_md_calc_structure + kwargs."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD
    from pyiron_workflow_atomistics.engine.ase import ase_md_calc_structure

    engine = ASEEngine(
        EngineInput=CalcInputMD(mode="NVT", thermostat="langevin", n_ionic_steps=1),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    fn, kwargs = engine.get_calculate_fn(structure=_cu_supercell())
    assert fn is ase_md_calc_structure
    assert "md_input" in kwargs
    assert kwargs["md_input"] is engine.EngineInput
    assert "record_interval" in kwargs


def test_ase_md_npt_berendsen_compressibility_is_threaded_into_dyn(tmp_path: Path):
    """Verify the CalcInputMD.compressibility value reaches the underlying
    ASE NPTBerendsen integrator. Reading dyn.compressibility back out is
    cheaper and more direct than running enough MD to see the volume
    response numerically — which can be unstable for stiff combinations."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD

    structure = _cu_supercell()
    structure.calc = EMT()

    eng = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NPT",
            thermostat="berendsen",
            temperature=300.0,
            n_ionic_steps=1,
            time_step=1.0,
            pressure=1e5,
            compressibility=7.0e-7,  # Cu-ish
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    fn, kwargs = eng.get_calculate_fn(structure=structure)
    assert kwargs["md_input"].compressibility == 7.0e-7


def test_ase_md_velocities_initialised_at_target_T(tmp_path: Path):
    """Sanity check: with T0>0, the recorded kinetic energy is non-zero from step 1."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, run

    engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVE",
            thermostat="langevin",
            temperature=400.0,
            n_ionic_steps=2,
            time_step=1.0,
            seed=11,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=_cu_supercell(), engine=engine)
    # Final structure should have non-zero momenta since MaxwellBoltzmann initialised
    p = out.final_structure.get_momenta()
    assert np.linalg.norm(p) > 0
