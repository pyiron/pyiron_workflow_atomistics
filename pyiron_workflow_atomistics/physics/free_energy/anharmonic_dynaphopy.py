"""Anharmonic free energy via dynaphopy MD-projection — single T and TDI over T."""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf


@pwf.as_function_node("free_energy_per_atom", "entropy_per_atom", "cv_per_atom")
def _free_energy_from_spectrum(
    frequencies: np.ndarray,         # (n_q, n_band) THz
    q_weights: np.ndarray,           # (n_q,), sums to 1
    temperature: float,              # K
    n_atoms_primitive: int,
) -> tuple[float, float, float]:
    """Harmonic free energy / entropy / Cv on a discrete (q, band) frequency grid.

    F = sum_q w_q * sum_b [ ℏω_qb/2 + k_B T ln(1 − exp(−ℏω_qb / k_B T)) ]

    Acoustic modes at Γ are zeroed upstream by dynaphopy's
    `_project_with_dynaphopy` (positions[:3]=0 at q==0); any ω ≤ 0 remaining
    is treated as imaginary and rejected.

    Units
    -----
    Frequencies in THz. Returned F / S / Cv per primitive-cell atom, in
    eV / (eV/K) / (eV/K) respectively.
    """
    import scipy.constants as c

    freqs = np.asarray(frequencies, dtype=float)
    weights = np.asarray(q_weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"q_weights must sum to 1, got {weights.sum()}")
    if freqs.ndim != 2 or freqs.shape[0] != weights.shape[0]:
        raise ValueError(
            f"frequencies shape {freqs.shape} incompatible with q_weights shape "
            f"{weights.shape}"
        )

    # Acoustic-at-Γ modes (already zeroed upstream) are dropped from the sum.
    is_acoustic_gamma = freqs == 0.0
    active = freqs[~is_acoustic_gamma]
    imag_mask = active < 0
    if imag_mask.any():
        raise ValueError(
            f"Spectrum has {int(imag_mask.sum())} imaginary modes; "
            "harmonic free energy is undefined for an unstable spectrum."
        )

    omega_rad_s = freqs * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV  # (n_q, n_band)

    if temperature <= 0:
        # T=0: only zero-point energy contributes; S and Cv are zero.
        F_modes = 0.5 * hbar_omega_eV
        F_modes = np.where(is_acoustic_gamma, 0.0, F_modes)
        F = float((weights[:, None] * F_modes).sum() / n_atoms_primitive)
        S = 0.0
        Cv = 0.0
    else:
        kT_eV = c.Boltzmann * temperature / c.eV
        x = hbar_omega_eV / kT_eV
        # ln(1 − exp(−x)) — stable; modes with x huge → ln(1) = 0.
        log_term = np.where(is_acoustic_gamma, 0.0, np.log1p(-np.exp(-x)))
        F_modes = 0.5 * hbar_omega_eV + kT_eV * log_term
        F_modes = np.where(is_acoustic_gamma, 0.0, F_modes)

        # Entropy per mode:  S = k_B [ x/(exp(x)−1) − ln(1 − exp(−x)) ]
        kB_eV_per_K = c.Boltzmann / c.eV
        with np.errstate(over="ignore", invalid="ignore"):
            S_modes = kB_eV_per_K * (
                np.where(is_acoustic_gamma, 0.0, x / np.expm1(x)) - log_term
            )

        # Cv per mode:  C_v = k_B (x^2 exp(x)) / (exp(x)−1)^2
        with np.errstate(over="ignore", invalid="ignore"):
            denom = np.expm1(x) ** 2
            num = (x**2) * np.exp(x)
            Cv_modes = kB_eV_per_K * np.where(
                is_acoustic_gamma | (denom == 0), 0.0, num / denom
            )

        F = float((weights[:, None] * F_modes).sum() / n_atoms_primitive)
        S = float((weights[:, None] * S_modes).sum() / n_atoms_primitive)
        Cv = float((weights[:, None] * Cv_modes).sum() / n_atoms_primitive)

    return F, S, Cv
