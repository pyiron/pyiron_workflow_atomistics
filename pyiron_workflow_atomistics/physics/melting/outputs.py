from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class MeltingIterationRecord:
    temperature_in: float
    temperature_next: float
    strains: list[float]
    ratios: list[float]
    pressures: list[float]
    temperatures: list[float]
    converged: bool
    center: float = 1.0  # fitted zero-pressure strain; next grid re-centres here


@dataclass
class MeltingResult:
    melting_temperature: float
    converged: bool
    n_iterations: int
    element: str
    crystalstructure: str
    n_atoms: int
    initial_guess: float
    iterations: list[MeltingIterationRecord] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class PhaseScreenRecord:
    """One candidate polymorph's cheap Step-1 screen result.

    ``crystalstructure`` is the *requested* phase; ``observed_phase`` is the
    dominant CNA class of the relaxed (0 K) cell — they differ when the potential
    spontaneously transforms the seeded phase. ``held`` is their agreement.
    ``t_guess`` is the Step-1 superheating estimate used to rank candidates.
    """

    crystalstructure: str
    observed_phase: str
    lattice_constant: float
    t_guess: float
    held: bool
    refined: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class MeltingScanResult:
    """Across-polymorph melting scan: screen all candidates, refine survivors.

    The melting point is ``max`` over refined polymorphs and ``selected_phase``
    is the polymorph achieving it (the potential's own pre-melt phase).
    """

    element: str
    melting_temperature: float
    selected_phase: str
    runner_up_phase: str | None
    delta_runner_up: float | None
    screened: list[PhaseScreenRecord] = field(default_factory=list)
    refined: list[MeltingResult] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["screened"] = [r.to_dict() for r in self.screened]
        out["refined"] = [r.to_dict() for r in self.refined]
        return out
