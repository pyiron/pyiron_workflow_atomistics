"""Unit tests for the gb_energy() pure formula."""

from __future__ import annotations

import logging

import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies import (
    gb_energy,
)


def test_zero_excess_gives_zero_egb():
    # E_total exactly == n * E_coh ⇒ Egb = 0
    assert gb_energy(
        final_energy_ev=-48.312,
        n_gb_atoms=10,
        gb_area_a2=100.0,
        e_cohesive_ev=-4.8312,
    ) == pytest.approx(0.0, abs=1e-9)


def test_positive_excess_converts_to_jpm2():
    # 1 eV excess over 1 Å² ⇒ Egb = 16.021766 J/m²
    e = gb_energy(
        final_energy_ev=-47.312,  # 1 eV above 10*(-4.8312)
        n_gb_atoms=10,
        gb_area_a2=1.0,
        e_cohesive_ev=-4.8312,
    )
    assert e == pytest.approx(16.021766, rel=1e-6)


def test_negative_egb_clamped_to_hundred(caplog):
    with caplog.at_level(
        logging.WARNING,
        logger="pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies",
    ):
        e = gb_energy(
            final_energy_ev=-49.0,  # below the bulk reference
            n_gb_atoms=10,
            gb_area_a2=100.0,
            e_cohesive_ev=-4.8312,
        )
    assert e == 100.0
    assert "negative" in caplog.text.lower()


def test_realistic_scale():
    # Cu GB target: 0.5 J/m² over 100 Å² with 50 atoms in GB region at E_coh=-3.59 eV/atom.
    # E_excess = 0.5 / 16.021766 * 100 ≈ 3.121 eV
    # E_total  = 50 * (-3.59) + 3.121 ≈ -176.38 eV
    e = gb_energy(
        final_energy_ev=-176.38,
        n_gb_atoms=50,
        gb_area_a2=100.0,
        e_cohesive_ev=-3.59,
    )
    assert e == pytest.approx(0.5, rel=0.01)
