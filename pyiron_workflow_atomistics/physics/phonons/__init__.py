"""Phonon workflows.

v0.0.7 covers phono3py-based lattice thermal conductivity κ(T) on top of a
phonopy harmonic FC2 calculation.

v0.0.8 adds dynaphopy-based MD-trajectory anharmonic renormalisation, which
captures full anharmonicity at finite T from a Langevin NVT segment.

Polar-material non-analytic correction (BORN + ε∞) is documented as a v2
follow-up in both phono3py and dynaphopy specs.

Public API
----------
- :class:`PhononOutput` — structured result of the phono3py BTE workflow.
- :class:`MdPhononOutput` — structured result of the dynaphopy MD-projection workflow.
- :func:`calculate_phonon_thermal_conductivity` — phono3py macro (v0.0.7).
- :func:`calculate_phonon_md_renormalisation` — dynaphopy macro (v0.0.8).
"""

from .anharmonic import calculate_phonon_thermal_conductivity
from .md_renormalised import calculate_phonon_md_renormalisation
from .output import MdPhononOutput, PhononOutput

__all__ = [
    "PhononOutput",
    "MdPhononOutput",
    "calculate_phonon_thermal_conductivity",
    "calculate_phonon_md_renormalisation",
]
