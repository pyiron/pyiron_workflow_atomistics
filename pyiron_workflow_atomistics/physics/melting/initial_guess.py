from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf

from pyiron_workflow_atomistics.analysis.structure_descriptors import cna_fractions
from pyiron_workflow_atomistics.engine import CalcInputMD, calculate


def _fraction(structure, key_max):
    """Population fraction of the dominant crystalline phase ``key_max``."""
    counts = cna_fractions.node_function(structure)
    return counts.get(key_max, 0) / len(structure)


def _heated_solid(structure, engine, temperature, strain_run_steps, timestep, seed,
                  subdir, npt_thermostat="berendsen"):
    """NPT-heat ``structure`` at ``temperature`` and return the final structure.

    ``npt_thermostat`` must keep the cell isotropic/orthorhombic ("berendsen" for
    ASE, "nose-hoover" for LAMMPS) so CNA classification stays valid.
    """
    md = CalcInputMD(
        mode="NPT",
        thermostat=npt_thermostat,
        temperature=temperature,
        pressure=0.0,
        n_ionic_steps=strain_run_steps,
        n_print=max(1, strain_run_steps // 10),
        time_step=timestep,
        initial_temperature=2.0 * temperature,
        seed=seed,
        compressibility=1e-6,
    )
    tag = f"{subdir}_{int(round(temperature))}"
    eng = replace(engine, EngineInput=md).with_working_directory(tag)
    return calculate.node_function(structure, engine=eng).final_structure


@pwf.as_function_node("t_guess", "structure")
def estimate_melting_temperature(
    structure,
    engine,
    key_max,
    distribution_half,
    crystalstructure,
    temperature_left=0.0,
    temperature_right=1000.0,
    strain_run_steps=1000,
    timestep=2.0,
    seed=None,
    t_step_min=10.0,
    max_iterations=40,
    t_ceiling=None,
    npt_thermostat="berendsen",
    subdir="guess",
):
    """Bisection: heat the bulk solid in NPT; raise T while it stays crystalline.

    Port of ``get_initial_melting_temperature_guess`` + ``next_step_funct`` with
    a hard ``max_iterations`` guard and ``t_ceiling`` so an undersampled MD run
    (where nothing melts) cannot expand the bracket without bound.
    """
    ceiling = t_ceiling if t_ceiling is not None else max(
        temperature_right * 3.0, temperature_right + 1.0
    )
    t_left, t_right = temperature_left, temperature_right
    struct_left = structure
    struct_right = _heated_solid(
        structure, engine, t_right, strain_run_steps, timestep, seed, subdir,
        npt_thermostat=npt_thermostat,
    )
    step = t_right - t_left
    iteration = 0
    while step > t_step_min and iteration < max_iterations:
        iteration += 1
        f_left = _fraction(struct_left, key_max)
        f_right = _fraction(struct_right, key_max)
        diff = t_right - t_left
        if f_left > distribution_half and f_right > distribution_half:
            if t_right >= ceiling:
                break
            struct_left, t_left = struct_right.copy(), t_right
            t_right = min(t_right + diff, ceiling)
            struct_right = _heated_solid(
                structure, engine, t_right, strain_run_steps, timestep, seed, subdir,
                npt_thermostat=npt_thermostat,
            )
        elif f_left > distribution_half >= f_right:
            diff /= 2.0
            t_left += diff
            struct_left = _heated_solid(
                structure, engine, t_left, strain_run_steps, timestep, seed, subdir,
                npt_thermostat=npt_thermostat,
            )
        elif f_left <= distribution_half and f_right <= distribution_half:  # both molten
            diff /= 2.0
            t_right, struct_right = t_left, struct_left.copy()
            t_left -= diff
            struct_left = _heated_solid(
                structure, engine, t_left, strain_run_steps, timestep, seed, subdir,
                npt_thermostat=npt_thermostat,
            )
        else:  # inverted (left molten, right solid): non-physical/noisy CNA — shrink
            # the bracket from the right and re-sample nearer the middle rather than
            # silently treating it as both-molten.
            diff /= 2.0
            t_right -= diff
            struct_right = _heated_solid(
                structure, engine, t_right, strain_run_steps, timestep, seed, subdir,
                npt_thermostat=npt_thermostat,
            )
        step = t_right - t_left
    t_guess = int(round(t_left))
    return t_guess, struct_left
