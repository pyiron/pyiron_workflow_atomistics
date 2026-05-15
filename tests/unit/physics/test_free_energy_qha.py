"""Tests for pyiron_workflow_atomistics.physics.free_energy.quasiharmonic."""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_qha_inputs(n_T=5, n_V=7):
    """Build a synthetic well-behaved QHA input grid (Vinet-like E(V), Einstein phonons)."""
    V0 = 16.0  # Å³/atom
    B0 = 70.0  # GPa, converted later
    Bp = 4.0
    volumes = V0 * np.linspace(0.95, 1.05, n_V)
    # Murnaghan-like E(V) parametrisation in eV/atom
    energies = -3.5 + 9 * V0 * B0 / 1602.176 / Bp / (Bp - 1) * (volumes / V0) ** (
        1 - Bp
    ) * ((volumes / V0) ** Bp - 1)
    # Per-volume Einstein-like F(T): F(T,V) = -3 k_B T ln(...) — keep simple.
    temperatures = np.linspace(0, 400, n_T)
    F_TV = np.zeros((n_T, n_V))
    S_TV = np.zeros((n_T, n_V))
    Cv_TV = np.zeros((n_T, n_V))
    for j, V in enumerate(volumes):
        omega_THz = 5.0 * (V0 / V) ** 1.5
        from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
            _free_energy_from_spectrum,
        )

        for i, T in enumerate(temperatures):
            F_TV[i, j], S_TV[i, j], Cv_TV[i, j] = (
                _free_energy_from_spectrum.node_function(
                    frequencies=np.full((1, 3), omega_THz),
                    q_weights=np.array([1.0]),
                    temperature=T,
                    n_atoms_primitive=1,
                )
            )
    return energies, volumes, temperatures, F_TV, S_TV, Cv_TV


def test_fit_qha_produces_finite_arrays():
    pytest.importorskip("phonopy.qha", reason="phonopy.qha not installed")
    from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import _fit_qha

    energies, volumes, T, F_TV, S_TV, Cv_TV = _synthetic_qha_inputs()
    result = _fit_qha.node_function(
        energies=energies,
        volumes=volumes,
        free_energy_per_T_V=F_TV,
        entropy_per_T_V=S_TV,
        cv_per_T_V=Cv_TV,
        temperatures=T,
        pressure_GPa=0.0,
        eos_type="vinet",
    )
    for key in (
        "equilibrium_volume_array",
        "gibbs_free_energy_array",
        "bulk_modulus_array",
        "thermal_expansion_array",
    ):
        arr = result[key]
        assert arr.shape == T.shape, f"{key} shape {arr.shape} != {T.shape}"
        assert np.all(np.isfinite(arr[1:])), f"{key} has NaNs at finite T"


def test_check_qha_volume_range_raises_on_nan_volume():
    from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
        _check_qha_volume_range,
    )

    V_T = np.array([16.0, 16.1, np.nan, np.nan])
    T = np.array([0.0, 100.0, 200.0, 300.0])
    volumes = np.array([15.5, 16.0, 16.5])
    with pytest.raises(RuntimeError, match="QHA equilibrium volume undefined"):
        _check_qha_volume_range(V_T, T, strain_range=(-0.03, 0.03), volumes=volumes)


@pytest.mark.slow
def test_quasiharmonic_free_energy_emt_al(tmp_path):
    pytest.importorskip("phonopy.qha", reason="phonopy.qha not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
        quasiharmonic_free_energy,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = quasiharmonic_free_energy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(0.0, 300.0),
        strain_range=(-0.03, 0.03),
        num_volumes=5,
        working_directory=str(tmp_path),
        subdir="qha",
    )
    out = wf.run()
    out = out["free_energy_output"] if isinstance(out, dict) else out

    assert out.mode == "qha"
    # Thermal expansion is positive on warming for Al/EMT
    assert out.equilibrium_volume_array[1] > out.equilibrium_volume_array[0]
    # Thermal expansion coefficient is finite and positive at 300 K
    assert np.isfinite(out.thermal_expansion_array[1])
    assert out.thermal_expansion_array[1] > 0
