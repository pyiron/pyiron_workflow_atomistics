"""PhononOutput dataclass — the structured result of a phonon workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ase import Atoms


@dataclass
class PhononOutput:
    """Structured result of a phonon thermal-conductivity calculation.

    Required fields are always populated. Optional fields are populated only
    when the corresponding macro flag is on:
        mode_resolved=True       → q_points, frequencies, group_velocities,
                                   mode_kappa, gamma, gruneisen
        harmonic_observables=True → band_structure, dos, free_energy
        keep_handles=True        → fc2, fc3, phono3py
    """

    structure: Atoms
    fc2_supercell_matrix: np.ndarray  # (3, 3) int
    fc3_supercell_matrix: np.ndarray  # (3, 3) int
    temperatures: np.ndarray  # (n_T,) K
    kappa: np.ndarray  # (n_T, 3, 3) W/m·K
    converged: bool

    q_points: np.ndarray | None = None  # (n_q, 3) reduced
    frequencies: np.ndarray | None = None  # (n_q, n_band) THz
    group_velocities: np.ndarray | None = None  # (n_q, n_band, 3)
    mode_kappa: np.ndarray | None = None  # (n_T, n_q, n_band, 6) Voigt
    gamma: np.ndarray | None = None  # (n_T, n_q, n_band) linewidths
    gruneisen: np.ndarray | None = None  # (n_q, n_band)

    band_structure: dict | None = None
    dos: dict | None = None
    free_energy: dict | None = None

    fc2: np.ndarray | None = None
    fc3: np.ndarray | None = None
    phono3py: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every field (ASE/phono3py objects by reference)."""
        return asdict(self)
