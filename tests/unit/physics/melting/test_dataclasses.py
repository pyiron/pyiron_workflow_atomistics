from pyiron_workflow_atomistics.physics.melting.inputs import MeltingInput
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingIterationRecord,
    MeltingResult,
)


def test_melting_input_defaults():
    mi = MeltingInput(element="Al")
    assert mi.n_atoms == 8000
    assert mi.timestep_lst == [2.0, 2.0, 1.0]
    assert mi.convergence_goal == 1.0


def test_melting_result_to_dict():
    rec = MeltingIterationRecord(
        temperature_in=900.0,
        temperature_next=905.0,
        strains=[1.0],
        ratios=[0.5],
        pressures=[0.0],
        temperatures=[905.0],
        converged=True,
    )
    res = MeltingResult(
        melting_temperature=905.0,
        converged=True,
        n_iterations=1,
        element="Al",
        crystalstructure="fcc",
        n_atoms=4000,
        initial_guess=914.0,
        iterations=[rec],
        report={},
    )
    d = res.to_dict()
    assert d["melting_temperature"] == 905.0
    assert len(d["iterations"]) == 1
