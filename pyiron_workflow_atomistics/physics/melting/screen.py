"""Across-the-table melting scan: cheap Step-1 screen of candidate polymorphs,
expensive Step-2 coexistence only on the survivors, ``Tm = max`` over them.

The melting point is a property of the *potential*: the relevant pre-melt solid is
whichever polymorph that potential makes most stable just below melting. Rather than
imposing an experimental phase, this module discovers it — Step 1 (the superheating
NPT bisection) is cheap enough to run on every candidate as a screen, and the
polymorph with the highest coexistence ``Tm`` (lowest free energy ⇒ melts last) is
the one the potential selects. Seeding a metastable phase is self-correcting: it
either transforms during equilibration (caught by CNA) or yields a lower ``Tm`` that
``max`` discards.
"""

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
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingScanResult,
    PhaseScreenRecord,
)
from pyiron_workflow_atomistics.physics.melting.structures import (
    create_coexistence_supercell,
    estimate_lattice_constant,
)

# CNA-classifiable solids the coexistence solid-fraction analysis can label.
_REFINABLE_PHASES = ("fcc", "bcc", "hcp")


def _default_candidate_phases(element):
    """{fcc, bcc, hcp} plus the element's ASE reference state (order-preserving)."""
    phases = ["fcc", "bcc", "hcp"]
    ref = reference_states[atomic_numbers[element]]["symmetry"]
    if ref not in phases:
        phases.append(ref)
    return phases


@pwf.as_function_node("t_guess", "structure", "observed_phase")
def screen_phase(
    engine,
    element,
    crystalstructure,
    a,
    n_atoms=8000,
    temperature_left=0.0,
    temperature_right=1000.0,
    strain_run_steps=1000,
    timestep=2.0,
    seed=None,
    npt_thermostat="berendsen",
    subdir="screen",
):
    """Build -> relax -> Step-1 estimate for ONE candidate polymorph (cheap screen).

    Returns the Step-1 superheating estimate ``t_guess``, the warm near-melt
    structure to seed refinement, and the relaxed cell's dominant CNA phase
    (``observed_phase``) — which differs from ``crystalstructure`` when the seeded
    polymorph transforms on relaxation.
    """
    structure = create_coexistence_supercell.node_function(
        element, crystalstructure, a=a, n_atoms=n_atoms
    )
    relax_engine = replace(
        engine, EngineInput=CalcInputMinimize(relax_cell=True)
    ).with_working_directory(f"{subdir}_min")
    relaxed = calculate.node_function(structure, engine=relax_engine).final_structure
    observed_phase, _, distribution_half = analyse_reference_structure.node_function(
        relaxed
    )
    t_guess, struct_at_guess = estimate_melting_temperature.node_function(
        relaxed,
        engine,
        key_max=observed_phase,
        distribution_half=distribution_half,
        crystalstructure=observed_phase,
        temperature_left=temperature_left,
        temperature_right=temperature_right,
        strain_run_steps=strain_run_steps,
        timestep=timestep,
        seed=seed,
        npt_thermostat=npt_thermostat,
        subdir=subdir,
    )
    return t_guess, struct_at_guess, observed_phase


def _dedupe_by_observed(screened):
    """One record per observed phase: prefer ``held``, then the higher ``t_guess``."""
    best: dict[str, PhaseScreenRecord] = {}
    for rec in screened:
        cur = best.get(rec.observed_phase)
        if cur is None or (rec.held, rec.t_guess) > (cur.held, cur.t_guess):
            best[rec.observed_phase] = rec
    return list(best.values())


def _select_for_refinement(screened, n_refine):
    """Pick the polymorphs worth the expensive coexistence step.

    Dedupe by observed phase, keep only CNA-refinable solids (fcc/bcc/hcp), then
    take the ``n_refine`` highest Step-1 estimates (most-stable-first). Falls back
    to the deduped set if nothing is refinable so the scan still returns a number.
    """
    deduped = _dedupe_by_observed(screened)
    refinable = [r for r in deduped if r.observed_phase in _REFINABLE_PHASES]
    pool = refinable or deduped
    pool = sorted(pool, key=lambda r: r.t_guess, reverse=True)
    return pool[: max(1, n_refine)]


@pwf.as_function_node("result")
def melting_point_scan(engine, melting_input):
    """Discover the pre-melt phase and melting point across candidate polymorphs.

    Screens every candidate phase with the cheap Step-1 estimate, refines the top
    ``n_refine`` survivors with the full coexistence method, and reports
    ``Tm = max`` over them with the selected phase and the runner-up gap.
    """
    mi = melting_input
    phases = mi.candidate_phases or _default_candidate_phases(mi.element)

    screened: list[PhaseScreenRecord] = []
    artifacts: dict[str, tuple] = {}  # requested phase -> (t_guess, struct)
    for cs in phases:
        a_cs = (
            mi.a
            if mi.a is not None
            else estimate_lattice_constant.node_function(mi.element, engine, cs)
        )
        t_guess, struct, observed = screen_phase.node_function(
            engine,
            mi.element,
            cs,
            a_cs,
            n_atoms=mi.n_atoms,
            temperature_left=mi.temperature_left,
            temperature_right=mi.temperature_right,
            strain_run_steps=mi.strain_run_steps,
            timestep=mi.timestep_lst[0],
            seed=mi.seed,
            npt_thermostat=mi.npt_thermostat,
            subdir=f"screen_{cs}",
        )
        screened.append(
            PhaseScreenRecord(
                crystalstructure=cs,
                observed_phase=observed,
                lattice_constant=float(a_cs),
                t_guess=float(t_guess),
                held=bool(observed == cs.lower()),
            )
        )
        artifacts[cs] = (t_guess, struct)

    selected = _select_for_refinement(screened, mi.n_refine)
    selected_keys = {r.crystalstructure for r in selected}
    for rec in screened:
        rec.refined = rec.crystalstructure in selected_keys

    refined = []
    for rec in selected:
        t_guess, struct = artifacts[rec.crystalstructure]
        # Label refinement by the OBSERVED phase: that is the crystal actually
        # present in the warm seed, and the solid-fraction CNA target must match it.
        res = refine_melting_point.node_function(
            struct,
            engine,
            t_guess=t_guess,
            melting_input=mi,
            crystalstructure=rec.observed_phase,
        )
        refined.append(res)

    refined_sorted = sorted(refined, key=lambda r: r.melting_temperature, reverse=True)
    best = refined_sorted[0]
    runner = refined_sorted[1] if len(refined_sorted) > 1 else None
    result = MeltingScanResult(
        element=mi.element,
        melting_temperature=best.melting_temperature,
        selected_phase=best.crystalstructure,
        runner_up_phase=runner.crystalstructure if runner else None,
        delta_runner_up=(
            best.melting_temperature - runner.melting_temperature if runner else None
        ),
        screened=screened,
        refined=refined,
        report={
            "candidate_phases": list(phases),
            "n_refine": mi.n_refine,
            "refined_phases": [r.crystalstructure for r in refined_sorted],
        },
    )
    return result
