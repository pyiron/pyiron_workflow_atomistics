"""Structured result of a calphy free-energy calculation.

Same shape regardless of which mode produced it; per-mode arrays
(temperature_array, free_energy_array, melting_temperature, ...) are
None when not applicable. Pickleable — holds plain Python + numpy
types only, no Phase references, no LAMMPS handles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class FreeEnergyOutput:
    """Result of one calphy free-energy calculation.

    Units
    -----
    ``free_energy`` / ``free_energy_error`` / ``einstein_free_energy``:
    eV/atom (calphy native).
    ``temperature``: K.
    ``pressure``: bar (calphy native — differs from
    :class:`pyiron_workflow_atomistics.engine.CalcInputMD.pressure`
    which is Pa; do not mix).
    """

    mode: Literal["fe", "ts", "tscale", "pscale",
                  "melting_temperature", "alchemy", "composition_scaling"]
    reference_phase: Literal["solid", "liquid", "both"]
    free_energy: float
    free_energy_error: float
    temperature: float
    pressure: float
    n_atoms: int
    elements: list[str]
    simfolder: str
    report: dict[str, Any]

    # mode-specific; None when not applicable
    temperature_array: np.ndarray | None = None
    free_energy_array: np.ndarray | None = None
    pressure_array: np.ndarray | None = None
    melting_temperature: float | None = None
    melting_temperature_error: float | None = None
    composition_path: list[dict[str, int]] | None = None
    einstein_free_energy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of the dataclass fields."""
        return asdict(self)
