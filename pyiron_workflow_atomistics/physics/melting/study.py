from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf
from ase.data import atomic_numbers, reference_states

from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    analyse_reference_structure,
)
from pyiron_workflow_atomistics.engine import CalcInputMinimize, calculate
from pyiron_workflow_atomistics.physics.melting.coexistence import refine_melting_point
from pyiron_workflow_atomistics.physics.melting.initial_guess import (
    estimate_melting_temperature,
)
from pyiron_workflow_atomistics.physics.melting.structures import (
    create_coexistence_supercell,
)


def _default_crystalstructure(element):
    return reference_states[atomic_numbers[element]]["symmetry"]


@pwf.as_function_node("result")
def calculate_melting_point(engine, melting_input):
    """Full interface-method melting point: build -> relax -> Step 1 -> Step 2."""
    mi = melting_input
    crystalstructure = mi.crystalstructure or _default_crystalstructure(mi.element)
    structure = create_coexistence_supercell.node_function(
        mi.element, crystalstructure, a=mi.a, n_atoms=mi.n_atoms
    )
    relax_engine = replace(
        engine, EngineInput=CalcInputMinimize(relax_cell=True)
    ).with_working_directory("minimize")
    relaxed = calculate.node_function(structure, engine=relax_engine).final_structure
    key_max, _, distribution_half = analyse_reference_structure.node_function(relaxed)
    t_guess, struct_at_guess = estimate_melting_temperature.node_function(
        relaxed,
        engine,
        key_max=key_max,
        distribution_half=distribution_half,
        crystalstructure=crystalstructure,
        temperature_left=mi.temperature_left,
        temperature_right=mi.temperature_right,
        strain_run_steps=mi.strain_run_steps,
        timestep=mi.timestep_lst[0],
        seed=mi.seed,
        npt_thermostat=mi.npt_thermostat,
    )
    result = refine_melting_point.node_function(
        struct_at_guess,
        engine,
        t_guess=t_guess,
        melting_input=mi,
        crystalstructure=crystalstructure,
    )
    return result
