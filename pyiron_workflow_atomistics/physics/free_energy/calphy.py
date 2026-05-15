"""Public function-nodes — one per calphy mode.

Each node:
  1. Asserts the [free-energy] extra is installed via _require_*.
  2. Validates the engine has only its `command` set, and the structure
     is a 3D fully-periodic supercell.
  3. Creates ``working_directory/subdir/`` and chdirs into it.
  4. Builds a calphy.input.Calculation, runs it, packs a FreeEnergyOutput.
  5. Restores the previous cwd in a try/finally.

All calphy and pyiron_workflow_lammps imports happen inside node bodies
so importing this subpackage works without the [free-energy] extra.
"""

from __future__ import annotations

import os
from typing import Literal

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
    _build_calphy_calculation,
    _pack_free_energy_output,
    _run_calphy_job,
    _validate_engine_only_command,
    _validate_structure,
)
from pyiron_workflow_atomistics.physics.free_energy._compat import (
    _require_calphy,
    _require_lammps_engine,
)
from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput


def _run_one(
    *,
    mode: str,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str,
    subdir: str,
    reference_phase: str,
    temperature: float,
    pressure: float,
    builder_kwargs: dict,
) -> FreeEnergyOutput:
    """Shared body for every public node: validate → chdir → run → pack."""
    _require_calphy()
    _require_lammps_engine()
    _validate_engine_only_command(lammps_engine)
    _validate_structure(structure)

    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(simfolder)
        calc = _build_calphy_calculation(
            mode=mode,
            structure=structure,
            potential=potential,
            lammps_engine=lammps_engine,
            working_directory=simfolder,
            **builder_kwargs,
        )
        job, report = _run_calphy_job(calc)
        return _pack_free_energy_output(
            mode=mode,
            job=job,
            report=report,
            simfolder=simfolder,
            structure=structure,
            reference_phase=reference_phase,
            temperature=temperature,
            pressure=pressure,
        )
    finally:
        os.chdir(prev_cwd)


@pwf.as_function_node("free_energy_output")
def free_energy(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "free_energy",
    temperature: float,
    pressure: float = 0.0,
    reference_phase: Literal["solid", "liquid"],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Helmholtz/Gibbs free energy at one (T, P) via Frenkel-Ladd / UF reference.

    Pressure is in **bar** (calphy native). Temperature in K. Free energy
    returned in eV/atom.
    """
    return _run_one(
        mode="fe",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase=reference_phase,
        temperature=temperature,
        pressure=pressure,
        builder_kwargs=dict(
            temperature=temperature,
            pressure=pressure,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )


@pwf.as_function_node("free_energy_output")
def reversible_scaling_temperature(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "reversible_scaling_temperature",
    temperature_range: tuple[float, float],
    pressure: float = 0.0,
    reference_phase: Literal["solid", "liquid"],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Free energy along an isobar by reversible scaling in temperature.

    ``temperature_range`` is (lo, hi) in K. Pressure is in bar. The
    ``FreeEnergyOutput.temperature_array`` and ``free_energy_array``
    fields are populated with the integrated curve.
    """
    if (
        temperature_range is None
        or not hasattr(temperature_range, "__len__")
        or len(temperature_range) != 2
    ):
        raise ValueError(
            "reversible_scaling_temperature requires "
            "`temperature_range=(lo, hi)` (length-2 tuple)"
        )
    return _run_one(
        mode="ts",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase=reference_phase,
        temperature=float(temperature_range[0]),
        pressure=pressure,
        builder_kwargs=dict(
            temperature_range=temperature_range,
            pressure=pressure,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )


@pwf.as_function_node("free_energy_output")
def reversible_scaling_pressure(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "reversible_scaling_pressure",
    temperature: float,
    pressure_range: tuple[float, float],
    reference_phase: Literal["solid", "liquid"],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Free energy along an isotherm by reversible scaling in pressure.

    ``pressure_range`` is (lo, hi) in bar (calphy native).
    """
    if (
        pressure_range is None
        or not hasattr(pressure_range, "__len__")
        or len(pressure_range) != 2
    ):
        raise ValueError(
            "reversible_scaling_pressure requires "
            "`pressure_range=(lo, hi)` (length-2 tuple)"
        )
    return _run_one(
        mode="pscale",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase=reference_phase,
        temperature=temperature,
        pressure=float(pressure_range[0]),
        builder_kwargs=dict(
            temperature=temperature,
            pressure_range=pressure_range,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )


@pwf.as_function_node("free_energy_output")
def melting_temperature(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "melting_temperature",
    temperature_guess: float | None = None,
    pressure: float = 0.0,
    step: int = 200,
    max_attempts: int = 5,
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Automated solid+liquid free-energy crossover via calphy's MeltingTemp.

    ``temperature_guess`` is calphy's starting temperature in K (if
    ``None``, calphy guesses from ``mendeleev``). ``step`` (K) and
    ``max_attempts`` route to ``Calculation.melting_temperature``.
    Result has ``reference_phase="both"`` and populates
    ``melting_temperature`` + ``melting_temperature_error``.
    """
    if temperature_guess is not None and temperature_guess <= 0:
        raise ValueError(
            f"`temperature_guess` must be positive, got {temperature_guess}"
        )
    return _run_one(
        mode="melting_temperature",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase="both",
        temperature=temperature_guess or 0.0,
        pressure=pressure,
        builder_kwargs=dict(
            temperature_guess=temperature_guess,
            pressure=pressure,
            melting_step=step,
            melting_max_attempts=max_attempts,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )


@pwf.as_function_node("free_energy_output")
def alchemy(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "alchemy",
    temperature: float,
    pressure: float = 0.0,
    pair_style_target: str,
    pair_coeff_target: str,
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Alchemical free-energy difference between two potentials.

    The starting potential is the supplied ``potential``; the target
    potential is supplied as raw ``pair_style_target`` /
    ``pair_coeff_target`` strings.
    """
    missing = []
    if not pair_style_target:
        missing.append("pair_style_target")
    if not pair_coeff_target:
        missing.append("pair_coeff_target")
    if missing:
        raise ValueError(
            f"alchemy requires {missing} (raw LAMMPS strings for the "
            f"target potential)"
        )
    return _run_one(
        mode="alchemy",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase="solid",
        temperature=temperature,
        pressure=pressure,
        builder_kwargs=dict(
            temperature=temperature,
            pressure=pressure,
            pair_style_target=pair_style_target,
            pair_coeff_target=pair_coeff_target,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )


@pwf.as_function_node("free_energy_output")
def composition_scaling(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "composition_scaling",
    temperature: float,
    pressure: float = 0.0,
    output_chemical_composition: dict[str, int],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Free-energy integration over composition between two stoichiometries.

    The starting composition is read off ``structure``; the target
    composition is given as ``output_chemical_composition`` (a dict of
    element-symbol → atom-count).
    """
    if not output_chemical_composition:
        raise ValueError(
            "composition_scaling requires "
            "`output_chemical_composition={'A': n_a, 'B': n_b, ...}` "
            "(target atom counts per element)"
        )
    return _run_one(
        mode="composition_scaling",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase="solid",
        temperature=temperature,
        pressure=pressure,
        builder_kwargs=dict(
            temperature=temperature,
            pressure=pressure,
            output_chemical_composition=output_chemical_composition,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
