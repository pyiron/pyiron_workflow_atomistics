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
    from phonopy.qha import QHA

    E = np.asarray(energies, dtype=float)
    V = np.asarray(volumes, dtype=float)
    T = np.asarray(temperatures, dtype=float)
    F_TV = np.asarray(free_energy_per_T_V, dtype=float)
    S_TV = np.asarray(entropy_per_T_V, dtype=float)
    Cv_TV = np.asarray(cv_per_T_V, dtype=float)

    qha = QHA(
        volumes=V,
        electronic_energies=E,
        temperatures=T,
        free_energy=F_TV,
        cv=Cv_TV,
        entropy=S_TV,
        eos=eos_type,
        pressure=pressure_GPa,
    )
    qha.run()
    V_T = np.asarray(qha.get_volume_temperature())
    _check_qha_volume_range(V_T, T, strain_range=(V.min(), V.max()), volumes=V)

    return {
        "equilibrium_volume_array": V_T,
        "gibbs_free_energy_array": np.asarray(qha.get_gibbs_temperature()),
        "bulk_modulus_array": np.asarray(qha.get_bulk_modulus_temperature()),
        "thermal_expansion_array": np.asarray(qha.get_thermal_expansion()),
        "qha_handle": qha,
    }
