import pytest
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting import (
    MeltingInput,
    calculate_melting_point,
)


@pytest.mark.slow
def test_calculate_melting_point_end_to_end(tmp_path):
    mi = MeltingInput(
        element="Al",
        crystalstructure="fcc",
        a=4.05,
        n_atoms=500,
        temperature_right=1400.0,
        strain_run_steps=40,
        timestep_lst=[2.0],
        fit_range_lst=[0.05],
        nve_steps_lst=[20],
        nvt_run_steps=20,
        npt_run_steps=20,
        n_strain_points=5,
        ratio_boundary=0.4,
        max_coexistence_iterations=2,
        seed=1,
    )
    eng = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )
    res = calculate_melting_point.node_function(eng, mi)
    assert res.element == "Al"
    assert res.initial_guess >= 0
    assert isinstance(res.melting_temperature, float)
    assert res.n_iterations >= 1
