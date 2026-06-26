import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, calculate


@pytest.mark.slow
def test_npt_berendsen_metal_runs(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    md = CalcInputMD(
        mode="NPT",
        thermostat="berendsen",
        temperature=300.0,
        pressure=0.0,
        n_ionic_steps=20,
        n_print=5,
        time_step=2.0,
        thermostat_time_constant=100.0,
        pressure_damping_timescale=1000.0,
        compressibility=1e-6,
        seed=1,
        initial_temperature=600.0,
    )
    eng = ASEEngine(EngineInput=md, calculator=EMT(), working_directory=str(tmp_path))
    out = calculate.node_function(atoms, engine=eng)
    assert out.converged is True
    assert out.structures and out.structures[-1].get_temperature() > 0
