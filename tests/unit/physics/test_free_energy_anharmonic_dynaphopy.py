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


def _einstein_entropy_per_mode(omega_THz: float, T_K: float) -> float:
    """Closed-form S per mode for an Einstein oscillator. eV/K.

    S = k_B [ x/(exp(x)−1) − ln(1 − exp(−x)) ]
    """
    import scipy.constants as c

    omega_rad_s = omega_THz * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV
    kT_eV = c.Boltzmann * T_K / c.eV
    x = hbar_omega_eV / kT_eV
    kB_eV_per_K = c.Boltzmann / c.eV
    return kB_eV_per_K * (x / np.expm1(x) - np.log1p(-np.exp(-x)))


def _einstein_cv_per_mode(omega_THz: float, T_K: float) -> float:
    """Closed-form C_v per mode for an Einstein oscillator. eV/K.

    C_v = k_B [ x^2 exp(x) / (exp(x)−1)^2 ]
    """
    import scipy.constants as c

    omega_rad_s = omega_THz * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV
    kT_eV = c.Boltzmann * T_K / c.eV
    x = hbar_omega_eV / kT_eV
    kB_eV_per_K = c.Boltzmann / c.eV
    return kB_eV_per_K * (x**2) * np.exp(x) / np.expm1(x) ** 2


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

    expected_F = 3 * _einstein_free_energy_per_mode(omega_THz, 300.0)
    expected_S = 3 * _einstein_entropy_per_mode(omega_THz, 300.0)
    expected_Cv = 3 * _einstein_cv_per_mode(omega_THz, 300.0)
    assert F == pytest.approx(expected_F, rel=1e-8)
    assert S == pytest.approx(expected_S, rel=1e-8)
    assert Cv == pytest.approx(expected_Cv, rel=1e-8)


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
