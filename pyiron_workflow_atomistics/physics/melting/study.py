from __future__ import annotations

import pyiron_workflow as pwf
from ase.data import atomic_numbers, reference_states

from pyiron_workflow_atomistics.physics.melting.coexistence import refine_melting_point
from pyiron_workflow_atomistics.physics.melting.screen import screen_phase
from pyiron_workflow_atomistics.physics.melting.structures import (
    estimate_lattice_constant,
)


def _default_crystalstructure(element):
    return reference_states[atomic_numbers[element]]["symmetry"]


@pwf.as_function_node("result")
def calculate_melting_point(engine, melting_input):
    """Full interface-method melting point for ONE phase: screen -> refine.

    Builds, relaxes and runs the Step-1 superheating estimate (``screen_phase``),
    then refines with the Step-2 coexistence loop. Use ``melting_point_scan`` to
    discover the pre-melt phase across polymorphs instead of fixing it here.
    """
    mi = melting_input
    crystalstructure = mi.crystalstructure or _default_crystalstructure(mi.element)
    a = mi.a
    if a is None and crystalstructure != _default_crystalstructure(mi.element):
        a = estimate_lattice_constant.node_function(
            mi.element, engine, crystalstructure
        )
    t_guess, struct_at_guess, observed = screen_phase.node_function(
        engine,
        mi.element,
        crystalstructure,
        a,
        n_atoms=mi.n_atoms,
        temperature_left=mi.temperature_left,
        temperature_right=mi.temperature_right,
        strain_run_steps=mi.strain_run_steps,
        timestep=mi.timestep_lst[0],
        seed=mi.seed,
        npt_thermostat=mi.npt_thermostat,
        subdir="single",
    )
    result = refine_melting_point.node_function(
        struct_at_guess,
        engine,
        t_guess=t_guess,
        melting_input=mi,
        crystalstructure=observed,
    )
    return result
