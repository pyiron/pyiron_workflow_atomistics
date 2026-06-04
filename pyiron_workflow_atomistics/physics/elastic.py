"""Elastic constants via the Materials Project stress-strain method.

Computes the full 6x6 stiffness tensor and all derived constants listed at
https://docs.materialsproject.org/methodology/materials-methodology/elasticity
using pymatgen.analysis.elasticity (the same library Materials Project uses),
driven by any ASE-backed Engine.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

EV_PER_A3_TO_GPA = 160.21766208
# ASE get_stress is tension-positive (Cauchy), same as pymatgen elasticity.
# Validated by the EMT test (Cu must give C11 > 0); flip to -1.0 if inverted.
_ASE_STRESS_SIGN = 1.0


def voigt_stress_to_gpa(stress_voigt) -> np.ndarray:
    """ASE Voigt stress (eV/A^3, order [xx,yy,zz,yz,xz,xy]) -> 3x3 tensor in GPa."""
    s = np.asarray(stress_voigt, dtype=float) * _ASE_STRESS_SIGN * EV_PER_A3_TO_GPA
    xx, yy, zz, yz, xz, xy = s
    return np.array(
        [[xx, xy, xz],
         [xy, yy, yz],
         [xz, yz, zz]]
    )
