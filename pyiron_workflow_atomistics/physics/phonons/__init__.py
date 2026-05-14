"""Phonon workflows.

v1 covers phono3py-based lattice thermal conductivity κ(T) on top of a
phonopy harmonic FC2 calculation. Polar-material non-analytic correction
(BORN + ε∞) and dynaphopy-based MD renormalisation are documented in the
design spec as v2 follow-ups.

Public API
----------
- :class:`PhononOutput` — structured result dataclass.
- :func:`calculate_phonon_thermal_conductivity` — the user-facing macro.
"""

from .anharmonic import calculate_phonon_thermal_conductivity
from .output import PhononOutput

__all__ = ["PhononOutput", "calculate_phonon_thermal_conductivity"]
