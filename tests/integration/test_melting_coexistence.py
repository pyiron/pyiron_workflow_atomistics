import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting.coexistence import coexistence_iteration


@pytest.mark.slow
def test_one_coexistence_iteration_runs(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 6))
    eng = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )
    rec = coexistence_iteration.node_function(
        s, eng, temperature=900.0, crystalstructure="fcc", fit_range=0.05,
        n_strain_points=5, nvt_steps=20, nve_steps=20, npt_steps=20, timestep=2.0,
        delta_t_melt=1000.0, ratio_boundary=0.4, boundary_value=0.25, seed=1,
        subdir="iter0",
    )
    assert rec.temperature_in == 900.0
    assert isinstance(rec.temperature_next, float)
    assert len(rec.strains) == 5
