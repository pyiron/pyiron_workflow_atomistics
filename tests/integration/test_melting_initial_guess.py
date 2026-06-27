import math

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    analyse_reference_structure,
)
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting.initial_guess import (
    estimate_melting_temperature,
)


@pytest.mark.slow
def test_initial_guess_brackets_a_temperature(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    key_max, _, half = analyse_reference_structure.node_function(s)
    eng = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )
    t_guess, struct = estimate_melting_temperature.node_function(
        s, eng, key_max=key_max, distribution_half=half, crystalstructure="fcc",
        temperature_left=0.0, temperature_right=1400.0, strain_run_steps=40,
        timestep=2.0, seed=1, t_step_min=200.0, max_iterations=8,
    )
    assert math.isfinite(t_guess)
    assert t_guess >= 0
    assert len(struct) == len(s)
