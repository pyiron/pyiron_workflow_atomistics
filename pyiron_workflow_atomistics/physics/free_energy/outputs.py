"""Structured result of a free-energy calculation.

Same shape across calphy + phonopy harmonic + phonopy QHA + dynaphopy modes;
per-mode arrays are None when not applicable. Pickleable for plain-data
fields; phonopy/dynaphopy handle fields (carried in memory when
``keep_handles=True``) are excluded from ``to_dict()``.

Unit conventions
----------------
``free_energy`` / ``free_energy_error`` / ``free_energy_array`` / ``gibbs_free_energy_array``
/ ``einstein_free_energy``: eV/atom (calphy native).
``temperature`` / ``temperature_array``: K.
``pressure`` for calphy modes (fe, ts, tscale, pscale, melting_temperature,
alchemy, composition_scaling): bar (calphy native).
``pressure`` for ``qha``: GPa (phonopy.qha native).
``bulk_modulus_array``: GPa.
``thermal_expansion_array``: 1/K.
``volumes`` / ``equilibrium_volume_array``: Å³/atom.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np

_HANDLE_FIELDS = frozenset(
    {"phonopy_handle", "qha_handle", "dynaphopy_handle", "band_structure", "phonon_dos"}
)


@dataclass
class FreeEnergyOutput:
    """Result of one free-energy calculation across calphy / phonopy / dynaphopy modes."""

    mode: Literal[
        "fe",
        "ts",
        "tscale",
        "pscale",
        "melting_temperature",
        "alchemy",
        "composition_scaling",
        "harmonic",
        "qha",
        "anharmonic_dynaphopy",
        "anharmonic_dynaphopy_tdi",
    ]
    reference_phase: Literal["solid", "liquid", "both"]
    free_energy: float
    free_energy_error: float
    temperature: float
    pressure: float
    n_atoms: int
    elements: list[str]
    simfolder: str
    report: dict[str, Any]

    # existing optional fields
    temperature_array: np.ndarray | None = None
    free_energy_array: np.ndarray | None = None
    pressure_array: np.ndarray | None = None
    melting_temperature: float | None = None
    melting_temperature_error: float | None = None
    composition_path: list[dict[str, int]] | None = None
    einstein_free_energy: float | None = None

    # NEW — harmonic + dynaphopy (single T)
    entropy: float | None = None
    heat_capacity: float | None = None

    # NEW — harmonic + qha + dynaphopy TDI (arrays over temperature_array)
    entropy_array: np.ndarray | None = None
    heat_capacity_array: np.ndarray | None = None

    # NEW — qha specific
    volumes: np.ndarray | None = None
    free_energy_volume_array: np.ndarray | None = None
    equilibrium_volume_array: np.ndarray | None = None
    gibbs_free_energy_array: np.ndarray | None = None
    bulk_modulus_array: np.ndarray | None = None
    thermal_expansion_array: np.ndarray | None = None

    # NEW — dynaphopy specific
    harmonic_frequencies: np.ndarray | None = None
    renormalised_frequencies: np.ndarray | None = None
    linewidths: np.ndarray | None = None
    renormalised_frequencies_per_T: np.ndarray | None = None
    linewidths_per_T: np.ndarray | None = None
    q_mesh: tuple[int, int, int] | None = None

    # NEW — handles (always excluded from to_dict)
    phonopy_handle: Any | None = None
    qha_handle: Any | None = None
    dynaphopy_handle: Any | None = None
    band_structure: dict | None = None
    phonon_dos: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every non-handle field."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in _HANDLE_FIELDS
        }
