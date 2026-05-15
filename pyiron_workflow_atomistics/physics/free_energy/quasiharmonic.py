"""Quasiharmonic free energy via phonopy.qha — single user-facing entry point.

QHA recipe: EOS volume sweep + per-volume harmonic free energy → phonopy.qha.QHA
gives G(T,P), V*(T,P), B(T,P), α(T,P). Reuses `harmonic_free_energy` per volume.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, calculate
from pyiron_workflow_atomistics.physics.bulk import generate_structures
from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
    harmonic_free_energy,
)
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.phonons._compat import require_phonopy


def _check_qha_volume_range(
    V_T: np.ndarray,
    temperatures: np.ndarray,
    strain_range: tuple[float, float],
    volumes: np.ndarray,
) -> None:
    """Raise if `phonopy.qha` returned NaN V*(T) anywhere — indicates V grid too narrow."""
    nan_mask = ~np.isfinite(V_T)
    if nan_mask.any():
        bad_T = np.asarray(temperatures)[nan_mask].tolist()
        raise RuntimeError(
            f"QHA equilibrium volume undefined at T={bad_T} K — widen "
            f"`strain_range` or extend it to include positive strain "
            f"(current range: {strain_range}, current V grid: {volumes.tolist()} Å³/atom)."
        )


@pwf.as_function_node("qha_results")
def _fit_qha(
    energies,
    volumes,
    free_energy_per_T_V,
    entropy_per_T_V,
    cv_per_T_V,
    temperatures,
    pressure_GPa: float,
    eos_type: str,
) -> dict:
    """Fit phonopy.qha.QHA on the (V, T) grid and return derived thermodynamics."""
    require_phonopy()
    from phonopy.qha.core import QHA

    E = np.asarray(energies, dtype=float)
    V = np.asarray(volumes, dtype=float)
    T = np.asarray(temperatures, dtype=float)
    F_TV = np.asarray(free_energy_per_T_V, dtype=float)
    S_TV = np.asarray(entropy_per_T_V, dtype=float)
    Cv_TV = np.asarray(cv_per_T_V, dtype=float)

    # phonopy.qha truncates its outputs to length (len(T) - 1) because it needs
    # one extra temperature point as a finite-difference buffer for thermal
    # expansion. To return arrays aligned 1:1 with the input temperature grid,
    # append one extrapolated buffer point above T[-1] and slice the buffer off
    # the outputs (when phonopy retains it).
    if T.size >= 2:
        dT = T[-1] - T[-2]
        T_ext = np.concatenate([T, [T[-1] + dT]])
        F_ext = np.vstack([F_TV, 2.0 * F_TV[-1] - F_TV[-2]])
        S_ext = np.vstack([S_TV, 2.0 * S_TV[-1] - S_TV[-2]])
        Cv_ext = np.vstack([Cv_TV, 2.0 * Cv_TV[-1] - Cv_TV[-2]])
    else:
        T_ext, F_ext, S_ext, Cv_ext = T, F_TV, S_TV, Cv_TV

    qha = QHA(
        volumes=V,
        electronic_energies=E,
        temperatures=T_ext,
        fe_phonon=F_ext,
        cv=Cv_ext,
        entropy=S_ext,
        eos=eos_type,
        pressure=pressure_GPa,
    )
    qha.run()

    n_T = T.size
    V_T_raw = np.asarray(qha.volume_temperature)
    V_T = V_T_raw[:n_T] if V_T_raw.size >= n_T else V_T_raw
    _check_qha_volume_range(V_T, T, strain_range=(V.min(), V.max()), volumes=V)

    gibbs_raw = np.asarray(qha.get_gibbs_temperature())
    bulk_raw = np.asarray(qha.get_bulk_modulus_temperature())
    alpha_raw = np.asarray(qha.thermal_expansion)
    gibbs = gibbs_raw[:n_T] if gibbs_raw.size >= n_T else gibbs_raw
    bulk = bulk_raw[:n_T] if bulk_raw.size >= n_T else bulk_raw
    alpha = alpha_raw[:n_T] if alpha_raw.size >= n_T else alpha_raw

    return {
        "equilibrium_volume_array": V_T,
        "gibbs_free_energy_array": gibbs,
        "bulk_modulus_array": bulk,
        "thermal_expansion_array": alpha,
        "qha_handle": qha,
    }
