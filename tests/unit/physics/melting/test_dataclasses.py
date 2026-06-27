from pyiron_workflow_atomistics.physics.melting.inputs import MeltingInput
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingIterationRecord,
    MeltingResult,
    MeltingScanResult,
    PhaseScreenRecord,
)


def test_melting_input_defaults():
    mi = MeltingInput(element="Al")
    assert mi.n_atoms == 8000
    assert mi.timestep_lst == [2.0, 2.0, 1.0]
    assert mi.convergence_goal == 1.0
    assert mi.candidate_phases is None
    assert mi.n_refine == 2


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


def _melting_result(phase, tm):
    return MeltingResult(
        melting_temperature=tm,
        converged=True,
        n_iterations=1,
        element="Fe",
        crystalstructure=phase,
        n_atoms=2000,
        initial_guess=tm,
    )


def test_phase_screen_record_to_dict():
    rec = PhaseScreenRecord(
        crystalstructure="bcc",
        observed_phase="bcc",
        lattice_constant=2.87,
        t_guess=1800.0,
        held=True,
        refined=True,
    )
    d = rec.to_dict()
    assert d["observed_phase"] == "bcc"
    assert d["held"] is True
    assert d["refined"] is True


def test_melting_scan_result_to_dict_nested():
    res = MeltingScanResult(
        element="Fe",
        melting_temperature=1820.0,
        selected_phase="bcc",
        runner_up_phase="fcc",
        delta_runner_up=40.0,
        screened=[
            PhaseScreenRecord("bcc", "bcc", 2.87, 1800.0, True, True),
            PhaseScreenRecord("fcc", "fcc", 3.6, 1700.0, True, True),
        ],
        refined=[_melting_result("bcc", 1820.0), _melting_result("fcc", 1780.0)],
    )
    d = res.to_dict()
    assert d["selected_phase"] == "bcc"
    assert d["delta_runner_up"] == 40.0
    # nested dataclasses are serialised to plain dicts
    assert isinstance(d["screened"][0], dict)
    assert isinstance(d["refined"][0], dict)
    assert d["refined"][0]["melting_temperature"] == 1820.0
