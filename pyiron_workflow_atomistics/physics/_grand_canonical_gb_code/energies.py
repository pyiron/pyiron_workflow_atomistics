"""Grain-boundary energy formula.

Port of GRIP's ``Calculator.get_gb_energy`` (``core/calculator.py``), as a
pure function — no calculator state, no atoms, just numbers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# eV / Å² → J / m²
_EV_PER_A2_TO_J_PER_M2 = 16.021766

# Clamp value for unphysical negative GB energies (matches GRIP).
_NEGATIVE_CLAMP_J_PER_M2 = 100.0


def gb_energy(
    final_energy_ev: float,
    n_gb_atoms: int,
    gb_area_a2: float,
    e_cohesive_ev: float,
) -> float:
    """Grain-boundary energy in J/m².

        Egb = (E_total - n_gb_atoms × E_coh) / area × 16.021766

    Negative results indicate an unphysical configuration; clamped to
    100 J/m² with a warning (preserves upstream GRIP behaviour).
    """
    e_bulk = n_gb_atoms * e_cohesive_ev
    e_excess = final_energy_ev - e_bulk
    e_gb = e_excess / gb_area_a2 * _EV_PER_A2_TO_J_PER_M2

    if e_gb < 0:
        logger.warning(
            "Computed negative GB energy (%.4f J/m²); clamping to %.1f J/m²",
            e_gb,
            _NEGATIVE_CLAMP_J_PER_M2,
        )
        return _NEGATIVE_CLAMP_J_PER_M2
    return e_gb
