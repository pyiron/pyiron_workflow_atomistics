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
