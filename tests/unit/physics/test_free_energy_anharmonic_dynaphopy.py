"""Tests for pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy."""

from __future__ import annotations

import numpy as np
import pytest


def _einstein_free_energy_per_mode(omega_THz: float, T_K: float) -> float:
    """Closed-form F per mode for an Einstein oscillator. eV.

    F = ℏω/2 + k_B T ln(1 − exp(−ℏω / k_B T))
    """
    import scipy.constants as c

    omega_rad_s = omega_THz * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV
    if T_K == 0:
        return 0.5 * hbar_omega_eV
    kT_eV = c.Boltzmann * T_K / c.eV
    x = hbar_omega_eV / kT_eV
    return 0.5 * hbar_omega_eV + kT_eV * np.log1p(-np.exp(-x))


def test_free_energy_from_spectrum_matches_einstein_closed_form():
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        _free_energy_from_spectrum,
    )

    omega_THz = 5.0
    frequencies = np.full((1, 3), omega_THz)  # 1 q, 3 bands, all identical
    q_weights = np.array([1.0])
    F, S, Cv = _free_energy_from_spectrum.node_function(
        frequencies=frequencies,
        q_weights=q_weights,
        temperature=300.0,
        n_atoms_primitive=1,
    )

    expected_per_atom = 3 * _einstein_free_energy_per_mode(omega_THz, 300.0)
    assert F == pytest.approx(expected_per_atom, rel=1e-8)
    assert Cv > 0


def test_free_energy_from_spectrum_rejects_imaginary_modes():
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        _free_energy_from_spectrum,
    )

    frequencies = np.array([[5.0, -1.0, 3.0]])  # one imaginary
    q_weights = np.array([1.0])
    with pytest.raises(ValueError, match="imaginary modes"):
        _free_energy_from_spectrum.node_function(
            frequencies=frequencies,
            q_weights=q_weights,
            temperature=300.0,
            n_atoms_primitive=1,
        )
