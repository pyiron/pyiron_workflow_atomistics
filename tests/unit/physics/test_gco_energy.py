"""Unit tests for the gb_energy() pure formula."""

from __future__ import annotations

import logging

import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies import (
    gb_energy,
)


def test_zero_excess_gives_zero_egb():
    # E_total exactly == N * E_coh ⇒ Egb = 0
    assert gb_energy(
        final_energy_ev=-48.312,
        n_atoms=10,
        gb_area_a2=100.0,
        e_cohesive_ev=-4.8312,
    ) == pytest.approx(0.0, abs=1e-9)


def test_positive_excess_converts_to_jpm2_with_two_gb_factor():
    # 1 eV excess over 1 Å² with two-GB factor ⇒ Egb = 16.021766 / 2 J/m²
    e = gb_energy(
        final_energy_ev=-47.312,  # 1 eV above 10*(-4.8312)
        n_atoms=10,
        gb_area_a2=1.0,
        e_cohesive_ev=-4.8312,
    )
    assert e == pytest.approx(16.021766 / 2.0, rel=1e-6)


def test_negative_egb_clamped_to_hundred(caplog):
    with caplog.at_level(
        logging.WARNING,
        logger="pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies",
    ):
        e = gb_energy(
            final_energy_ev=-49.0,  # below the bulk reference
            n_atoms=10,
            gb_area_a2=100.0,
            e_cohesive_ev=-4.8312,
        )
    assert e == 100.0
    assert "negative" in caplog.text.lower()


def test_realistic_scale():
    # Al GB target: 0.5 J/m² over 100 Å² with 50 atoms at E_coh=-3.59 eV/atom.
    # Egb = (E_total − N·E_coh) / (2·A) · 16.021766
    # ⇒ E_total − N·E_coh = 0.5 · 2 · 100 / 16.021766 ≈ 6.242 eV
    # ⇒ E_total = 50·(-3.59) + 6.242 = -173.258 eV
    e = gb_energy(
        final_energy_ev=-173.258,
        n_atoms=50,
        gb_area_a2=100.0,
        e_cohesive_ev=-3.59,
    )
    assert e == pytest.approx(0.5, rel=0.01)


def test_matches_get_GB_energy_convention():
    """gb_energy() must match physics.grain_boundary.get_GB_energy semantics.

    Both use the periodic-bicrystal convention: full atom count divided by
    twice the cross-sectional area.
    """
    e_pwa = gb_energy(
        final_energy_ev=-100.0,
        n_atoms=30,
        gb_area_a2=50.0,
        e_cohesive_ev=-3.5,
    )
    # Same formula, computed inline:
    expected = (-100.0 - 30 * (-3.5)) / (2.0 * 50.0) * 16.021766
    assert e_pwa == pytest.approx(expected, rel=1e-9)
