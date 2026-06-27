from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf

from pyiron_workflow_atomistics.analysis.structure_descriptors import holes_mask
from pyiron_workflow_atomistics.physics.melting.fitting import (
    predict_melting_point,
    ratio_selection,
)
from pyiron_workflow_atomistics.physics.melting.md_steps import (
    build_solid_liquid_interface,
    npt_relax_solid,
    strain_scan_nvt_nve,
)
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingIterationRecord,
    MeltingResult,
)


def _strain_grid(center, fit_range, n_points):
    return [
        round(float(s), 4)
        for s in np.linspace(center - fit_range, center + fit_range, n_points)
    ]


def _next_center(strains, pressures, fallback=1.0):
    """Zero-pressure strain from a linear P(strain) fit (reference get_center_point).

    Returns ``fallback`` if the fit is degenerate or the root is unphysical so the
    strain grid never wanders far from the equilibrium c-length.
    """
    if len(strains) < 2:
        return fallback
    try:
        slope, intercept = np.polyfit(strains, pressures, 1)
        if abs(slope) < 1e-12:
            return fallback
        center = abs(-intercept / slope)
    except Exception:
        return fallback
    if not np.isfinite(center) or not (0.5 < center < 1.5):
        return fallback
    return float(round(center, 4))


@pwf.as_function_node("record")
def coexistence_iteration(
    structure,
    engine,
    temperature,
    crystalstructure,
    fit_range=0.05,
    n_strain_points=21,
    nvt_steps=10000,
    nve_steps=20000,
    npt_steps=50000,
    timestep=2.0,
    delta_t_melt=1000.0,
    ratio_boundary=0.25,
    boundary_value=0.25,
    seed=None,
    npt_thermostat="berendsen",
    center=1.0,
    subdir="iter",
):
    """One interface-method temperature iteration -> next-T estimate.

    ``center`` is the strain the grid is built around (the previous iteration's
    fitted zero-pressure strain); the returned record carries the newly fitted
    ``center`` for the next iteration.
    """
    solid, _ = npt_relax_solid.node_function(
        structure, engine, temperature=temperature, n_steps=npt_steps,
        timestep=timestep, seed=seed, npt_thermostat=npt_thermostat,
        subdir=f"{subdir}_npt",
    )
    interface = build_solid_liquid_interface.node_function(
        solid, engine, t_solid=temperature, t_liquid=temperature + delta_t_melt,
        n_steps=npt_steps, timestep=timestep, seed=seed, subdir=f"{subdir}_iface",
    )
    strains = _strain_grid(center, fit_range, n_strain_points)
    records = strain_scan_nvt_nve.node_function(
        interface, engine, temperature=temperature, strains=strains,
        crystalstructure=crystalstructure, nvt_steps=nvt_steps, nve_steps=nve_steps,
        timestep=timestep, seed=seed, subdir=f"{subdir}_strain",
    )
    ratios = [r["solid_fraction"] for r in records]
    pressures = [r["mean_P"] for r in records]
    temps = [r["mean_T"] for r in records]
    sel_s, sel_r, sel_p, sel_t, flag = ratio_selection.node_function(
        strains, ratios, pressures, temps, ratio_boundary=ratio_boundary
    )
    if len(sel_s) > 2:
        sel_index = [strains.index(s) for s in sel_s]
        vmax = [records[i]["voronoi_max"] for i in sel_index]
        vmean = [records[i]["voronoi_mean"] for i in sel_index]
        keep = holes_mask.node_function(vmax, vmean, factor=2.0)
        sel_s = [s for s, k in zip(sel_s, keep) if k]
        sel_p = [p for p, k in zip(sel_p, keep) if k]
        sel_t = [t for t, k in zip(sel_t, keep) if k]
    if len(sel_s) > 2:
        t_next, _, _, _ = predict_melting_point.node_function(
            sel_s, sel_p, sel_t, boundary_value=boundary_value
        )
        next_center = _next_center(sel_s, sel_p, fallback=center)
    else:
        t_next = temperature * (1.10 if flag > 0 else 0.90)
        next_center = center
    record = MeltingIterationRecord(
        temperature_in=float(temperature),
        temperature_next=float(t_next),
        strains=list(strains),
        ratios=list(ratios),
        pressures=list(pressures),
        temperatures=list(temps),
        converged=False,
        center=float(next_center),
    )
    return record


@pwf.as_function_node("result")
def refine_melting_point(structure, engine, t_guess, melting_input, crystalstructure):
    """Iterate coexistence steps until |dT| <= convergence_goal.

    Refinement schedules (timestep, fit_range, nve_steps) advance one entry per
    iteration and then HOLD at the finest entry, repeating it until the
    temperature change between consecutive iterations meets the goal (or the
    ``max_coexistence_iterations`` safety cap is hit). Each iteration re-centres
    the strain grid on the previous iteration's fitted zero-pressure strain.
    """
    mi = melting_input
    schedules = list(zip(mi.timestep_lst, mi.fit_range_lst, mi.nve_steps_lst))
    temperature = float(t_guess)
    center = 1.0
    iterations: list[MeltingIterationRecord] = []
    converged = False
    for step_idx in range(mi.max_coexistence_iterations):
        timestep, fit_range, nve_steps = schedules[min(step_idx, len(schedules) - 1)]
        rec = coexistence_iteration.node_function(
            structure, engine, temperature=temperature,
            crystalstructure=crystalstructure, fit_range=fit_range,
            n_strain_points=mi.n_strain_points, nvt_steps=mi.nvt_run_steps,
            nve_steps=nve_steps, npt_steps=mi.npt_run_steps, timestep=timestep,
            delta_t_melt=mi.delta_t_melt, ratio_boundary=mi.ratio_boundary,
            boundary_value=mi.boundary_value, seed=mi.seed,
            npt_thermostat=mi.npt_thermostat, center=center, subdir=f"iter_{step_idx}",
        )
        iterations.append(rec)
        delta = abs(rec.temperature_next - temperature)
        temperature = rec.temperature_next
        center = rec.center
        # only declare convergence once the finest schedule is in effect
        if delta <= mi.convergence_goal and step_idx >= len(schedules) - 1:
            converged = True
            break
    result = MeltingResult(
        melting_temperature=float(temperature),
        converged=converged,
        n_iterations=len(iterations),
        element=mi.element,
        crystalstructure=crystalstructure,
        n_atoms=len(structure),
        initial_guess=float(t_guess),
        iterations=iterations,
        report={"schedules": schedules,
                "max_coexistence_iterations": mi.max_coexistence_iterations},
    )
    return result
