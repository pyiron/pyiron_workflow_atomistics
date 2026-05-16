"""Physics-level input dataclasses for engine calculations.

These dataclasses describe *what* you want the engine to do in
physics-level terms (force tolerance, temperature, ensemble) — never in
engine-specific jargon (no EDIFFG, no LAMMPS units style). The engine is
responsible for translating these to its native parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CalcInputStatic:
    """A single-point energy/force evaluation. No tunable parameters."""

    pass


@dataclass(init=False)
class CalcInputMinimize:
    """Structural relaxation parameters.

    Attributes
    ----------
    force_convergence_tolerance
        Max allowed force component on any atom, in eV/Å. Default 1e-2.
    energy_convergence_tolerance
        Energy change between consecutive steps, in eV. Default 1e-5.
    max_iterations
        Hard cap on optimiser steps.
    cell_relaxation
        Which cell degrees of freedom to relax. ``"none"`` (atoms only,
        fixed cell), ``"volume"`` (cell volume only, shape and atoms
        fixed), ``"shape"`` (cell shape only, volume and atoms fixed), or
        ``"full"`` (cell + atoms). Default ``"none"``.

    Notes
    -----
    The legacy ``relax_cell: bool`` argument is still accepted but
    deprecated — ``relax_cell=True`` is an alias for
    ``cell_relaxation="full"`` and ``relax_cell=False`` for
    ``cell_relaxation="none"``. Specifying both raises ``ValueError``.
    """

    force_convergence_tolerance: float = 1e-2
    energy_convergence_tolerance: float = 1e-5
    max_iterations: int = 1_000_000
    cell_relaxation: Literal["none", "volume", "shape", "full"] = "none"

    def __init__(
        self,
        force_convergence_tolerance: float = 1e-2,
        energy_convergence_tolerance: float = 1e-5,
        max_iterations: int = 1_000_000,
        cell_relaxation: Literal["none", "volume", "shape", "full"] | None = None,
        *,
        relax_cell: bool | None = None,
    ) -> None:
        if cell_relaxation is not None and relax_cell is not None:
            raise ValueError(
                "Specify cell_relaxation OR relax_cell, not both. "
                "relax_cell is deprecated; prefer cell_relaxation."
            )
        if relax_cell is not None:
            import warnings as _warnings

            _warnings.warn(
                "relax_cell is deprecated; use cell_relaxation='full' or "
                "cell_relaxation='none' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cell_relaxation = "full" if relax_cell else "none"
        if cell_relaxation is None:
            cell_relaxation = "none"
        object.__setattr__(self, "force_convergence_tolerance", force_convergence_tolerance)
        object.__setattr__(self, "energy_convergence_tolerance", energy_convergence_tolerance)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "cell_relaxation", cell_relaxation)

    @property
    def relax_cell(self) -> bool:
        """Deprecated. Backwards-compat property: True iff cell_relaxation != 'none'."""
        return self.cell_relaxation != "none"


@dataclass
class CalcInputMD:
    """Molecular-dynamics parameters with selectable ensemble and thermostat.

    Attributes
    ----------
    mode
        Ensemble: ``"NVE"``, ``"NVT"``, or ``"NPT"``.
    thermostat
        Coupling algorithm. ``"nose-hoover"`` (deterministic), ``"langevin"``
        (stochastic), ``"berendsen"`` (weak-coupling), ``"andersen"`` (random
        collisions).
    temperature
        Target temperature in Kelvin.
    n_ionic_steps
        Number of MD timesteps to run.
    n_print
        Frequency of thermo output (steps).
    pressure
        Target pressure in Pascal (used by NPT mode only).
    time_step
        Integration timestep in **femtoseconds**.
    thermostat_time_constant
        Thermostat coupling timescale in femtoseconds.
    pressure_damping_timescale
        Barostat coupling timescale in femtoseconds.
    seed
        RNG seed for stochastic thermostats. ``None`` ⇒ non-deterministic.
    initial_temperature
        Temperature used to initialise velocities (defaults to ``temperature``).
    compressibility
        Isothermal compressibility in bar⁻¹. Required by NPT-Berendsen; ignored
        by other ensembles. Default 4.57e-5 (≈ liquid water at room T); for a
        metal, supply something like 1e-6.
    """

    mode: Literal["NVE", "NVT", "NPT"] = "NVT"
    thermostat: Literal["nose-hoover", "langevin", "berendsen", "andersen"] = "langevin"
    temperature: float = 300.0
    n_ionic_steps: int = 10_000
    n_print: int = 100
    pressure: float | None = None
    time_step: float = 1.0  # fs
    thermostat_time_constant: float = 100.0  # fs
    pressure_damping_timescale: float = 1000.0  # fs
    seed: int | None = None
    initial_temperature: float | None = None
    compressibility: float = 4.57e-5  # bar^-1, water-like default
