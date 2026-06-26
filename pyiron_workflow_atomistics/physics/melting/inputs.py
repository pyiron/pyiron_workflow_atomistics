from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeltingInput:
    """Parameters for an interface-method melting-point run (notebook defaults)."""

    element: str
    crystalstructure: str | None = None  # default: ASE reference state
    a: float | None = None
    n_atoms: int = 8000  # solid cell targets n_atoms/2
    temperature_left: float = 0.0
    temperature_right: float = 1000.0
    convergence_goal: float = 1.0  # K
    timestep_lst: list[float] = field(default_factory=lambda: [2.0, 2.0, 1.0])
    fit_range_lst: list[float] = field(default_factory=lambda: [0.05, 0.01, 0.01])
    nve_steps_lst: list[int] = field(default_factory=lambda: [25000, 20000, 50000])
    nvt_run_steps: int = 10000
    npt_run_steps: int = 50000
    strain_run_steps: int = 1000
    n_strain_points: int = 21
    ratio_boundary: float = 0.25
    boundary_value: float = 0.25
    delta_t_melt: float = 1000.0  # superheat for interface build
    seed: int | None = None
    # NPT thermostat must keep the cell isotropic/orthorhombic: "berendsen" for the
    # ASE engine, "nose-hoover" for the LAMMPS engine (fix npt ... iso).
    npt_thermostat: str = "berendsen"
