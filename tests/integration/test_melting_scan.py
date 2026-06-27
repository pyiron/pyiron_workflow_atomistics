import pytest
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting import (
    MeltingInput,
    melting_point_scan,
)
from pyiron_workflow_atomistics.physics.melting.outputs import MeltingScanResult


@pytest.mark.slow
def test_melting_point_scan_end_to_end(tmp_path):
    # Restrict to fcc/bcc so the scan refines two polymorphs and picks the higher Tm.
    mi = MeltingInput(
        element="Al",
        candidate_phases=["fcc", "bcc"],
        n_refine=2,
        n_atoms=120,
        temperature_right=1400.0,
        strain_run_steps=40,
        timestep_lst=[2.0],
        fit_range_lst=[0.05],
        nve_steps_lst=[20],
        nvt_run_steps=20,
        npt_run_steps=20,
        n_strain_points=5,
        ratio_boundary=0.4,
        max_coexistence_iterations=1,
        seed=1,
    )
    eng = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )
    res = melting_point_scan.node_function(eng, mi)
    assert isinstance(res, MeltingScanResult)
    assert res.element == "Al"
    assert len(res.screened) == 2
    # selected phase achieves the reported melting temperature ...
    assert isinstance(res.melting_temperature, float)
    by_phase = {r.crystalstructure: r.melting_temperature for r in res.refined}
    assert by_phase[res.selected_phase] == res.melting_temperature
    assert res.melting_temperature == max(by_phase.values())
    # ... and the runner-up gap is non-negative when a runner-up exists.
    if res.runner_up_phase is not None:
        assert res.delta_runner_up >= 0
