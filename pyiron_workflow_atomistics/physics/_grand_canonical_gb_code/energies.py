"""Grain-boundary energy formula.

Adapted from GRIP's ``Calculator.get_gb_energy`` (``core/calculator.py``).

Differences from the GRIP formula
---------------------------------
GRIP divides by a single ``area`` and uses the count of atoms inside a
z-mask around the GB plane. That formulation only works when the
calculator provides per-atom energy decomposition (LAMMPS-native) so the
energy is summed against the same masked atom set. Under a generic ASE
engine the full slab energy is summed against a partial atom count,
producing non-physical negative excess.

Instead we use the standard pyiron_workflow_atomistics convention
(matches ``physics.grain_boundary.get_GB_energy``):

    Egb = (E_total − N × E_coh) / (2 × area) × 16.021766    [J/m²]

where ``N`` is the **total** atom count in the slab and the factor of 2
accounts for the two equivalent GB planes in a periodic bicrystal.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# eV / Å² → J / m²
_EV_PER_A2_TO_J_PER_M2 = 16.021766

# Clamp value for unphysical negative GB energies (preserves GRIP's behaviour
# of producing a single, easily-spotted sentinel value rather than a NaN).
_NEGATIVE_CLAMP_J_PER_M2 = 100.0


def gb_energy(
    final_energy_ev: float,
    n_atoms: int,
    gb_area_a2: float,
    e_cohesive_ev: float,
) -> float:
    """Grain-boundary energy in J/m² for a periodic bicrystal.

        Egb = (E_total − N × E_coh) / (2 × area) × 16.021766

    The factor of 2 in the denominator accounts for the two equivalent
    GB planes produced when the bicrystal is stitched with periodic
    boundaries (one at the stitch line, one across the z-boundary).

    Negative results indicate an unphysical configuration; clamped to
    100 J/m² with a warning.
    """
    e_bulk = n_atoms * e_cohesive_ev
    e_excess = final_energy_ev - e_bulk
    e_gb = e_excess / (2.0 * gb_area_a2) * _EV_PER_A2_TO_J_PER_M2

    if e_gb < 0:
        logger.warning(
            "Computed negative GB energy (%.4f J/m²); clamping to %.1f J/m²",
            e_gb,
            _NEGATIVE_CLAMP_J_PER_M2,
        )
        return _NEGATIVE_CLAMP_J_PER_M2
    return e_gb
