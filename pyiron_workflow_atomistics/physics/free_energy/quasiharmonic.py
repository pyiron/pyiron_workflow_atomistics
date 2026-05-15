"""Quasiharmonic free energy via phonopy.qha — single user-facing entry point.

QHA recipe: EOS volume sweep + per-volume harmonic free energy → phonopy.qha.QHA
gives G(T,P), V*(T,P), B(T,P), α(T,P). Reuses `harmonic_free_energy` per volume.
"""

from __future__ import annotations

import os

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, calculate
from pyiron_workflow_atomistics.physics.bulk import generate_structures
from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
    _resolve_simfolder,
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


@pwf.as_function_node("energies_per_volume", "volumes")
def _static_energies_per_volume(strained_structures: list[Atoms], engine: Engine):
    """One-shot static energy per strained cell. Returns (energies, volumes/atom)."""
    energies: list[float] = []
    volumes: list[float] = []
    for i, s in enumerate(strained_structures):
        sub_engine = engine.with_working_directory(f"vol_E_{i:03d}")
        out = calculate.node_function(structure=s, engine=sub_engine)
        if not out.converged:
            raise RuntimeError(
                f"Static-energy calc failed for strained cell {i} "
                f"(volume {s.get_volume():.3f} Å³)."
            )
        energies.append(float(out.final_energy))
        volumes.append(float(s.get_volume()) / len(s))
    return np.asarray(energies), np.asarray(volumes)


@pwf.as_function_node(
    "free_energy_per_T_V", "entropy_per_T_V", "cv_per_T_V"
)
def _harmonic_grid_over_volumes(
    strained_structures: list[Atoms],
    engine: Engine,
    fc2_supercell_matrix,
    temperatures,
    displacement_distance: float,
    is_plusminus,
    working_directory: str,
):
    """Run harmonic_free_energy at each strained cell, stack F/S/Cv along V axis."""
    import scipy.constants as c

    # phonopy.qha expects fe_phonon in kJ/mol and entropy/cv in J/K/mol per
    # primitive unit cell. ``harmonic_free_energy`` reports those quantities
    # in eV / (eV/K) / atom per the FreeEnergyOutput spec, so convert here
    # at the boundary.
    ev_to_kj_mol = c.eV * c.Avogadro / 1000.0  # ≈ 96.485

    T_arr = np.asarray(temperatures)
    n_T = int(T_arr.size)
    n_V = len(strained_structures)
    F_TV = np.zeros((n_T, n_V))
    S_TV = np.zeros((n_T, n_V))
    Cv_TV = np.zeros((n_T, n_V))
    for j, s in enumerate(strained_structures):
        vol_dir = os.path.join(working_directory, f"vol_{j:03d}")
        sub_engine = engine.with_working_directory(vol_dir)
        sub_wf = harmonic_free_energy(
            structure=s,
            engine=sub_engine,
            fc2_supercell_matrix=fc2_supercell_matrix,
            temperatures=temperatures,
            displacement_distance=displacement_distance,
            is_plusminus=is_plusminus,
            working_directory=vol_dir,
            subdir="harmonic",
        )
        out = sub_wf.run()
        out = out["free_energy_output"] if isinstance(out, dict) else out
        F_TV[:, j] = np.asarray(out.free_energy_array) * ev_to_kj_mol
        S_TV[:, j] = np.asarray(out.entropy_array) * ev_to_kj_mol * 1000.0
        Cv_TV[:, j] = np.asarray(out.heat_capacity_array) * ev_to_kj_mol * 1000.0
    return F_TV, S_TV, Cv_TV


@pwf.as_function_node("free_energy_output")
def _pack_qha_output(
    structure: Atoms,
    qha_results: dict,
    volumes: np.ndarray,
    free_energy_per_T_V: np.ndarray,
    entropy_per_T_V: np.ndarray,
    cv_per_T_V: np.ndarray,
    temperatures,
    pressure_GPa: float,
    simfolder: str,
    keep_handles: bool,
) -> FreeEnergyOutput:
    """Pack phonopy.qha results plus the V/T grids into a FreeEnergyOutput."""
    T = np.asarray(temperatures, dtype=float)
    elements = list(dict.fromkeys(structure.get_chemical_symbols()))
    # entropy_array/heat_capacity_array are a representative slice at the
    # central reference volume; the full (n_T, n_V) grid lives in
    # free_energy_volume_array (the unconstrained F(T,V) phonon grid).
    mid = entropy_per_T_V.shape[1] // 2
    return FreeEnergyOutput(
        mode="qha",
        reference_phase="solid",
        free_energy=float(qha_results["gibbs_free_energy_array"][0]),
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=float(pressure_GPa),
        n_atoms=len(structure),
        elements=elements,
        simfolder=simfolder,
        report={
            "method": "qha",
            "n_volumes": int(np.asarray(volumes).size),
            "pressure_GPa": float(pressure_GPa),
        },
        temperature_array=T,
        free_energy_array=qha_results["gibbs_free_energy_array"],
        entropy_array=entropy_per_T_V[:, mid],
        heat_capacity_array=cv_per_T_V[:, mid],
        volumes=volumes,
        free_energy_volume_array=free_energy_per_T_V,
        equilibrium_volume_array=qha_results["equilibrium_volume_array"],
        gibbs_free_energy_array=qha_results["gibbs_free_energy_array"],
        bulk_modulus_array=qha_results["bulk_modulus_array"],
        thermal_expansion_array=qha_results["thermal_expansion_array"],
        qha_handle=qha_results["qha_handle"] if keep_handles else None,
    )


@pwf.api.as_macro_node("free_energy_output")
def quasiharmonic_free_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0),
    pressure: float = 0.0,
    strain_range: tuple[float, float] = (-0.03, 0.03),
    num_volumes: int = 7,
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    eos_type: str = "vinet",
    working_directory: str = ".",
    subdir: str = "quasiharmonic_free_energy",
    keep_handles: bool = False,
):
    """Gibbs free energy G(T,P), V*(T,P), B(T,P), α(T,P) via phonopy.qha.QHA.

    Pressure is in **GPa** (phonopy.qha native). At ``pressure=0.0`` the
    ``gibbs_free_energy_array`` field is the Helmholtz free energy F(T).

    The returned ``FreeEnergyOutput`` populates ``free_energy_array`` directly
    from ``gibbs_free_energy_array`` for compatibility with the calphy ``ts``
    mode shape — at finite pressure this is Gibbs, at zero pressure it is
    Helmholtz.

    See spec ``docs/design/specs/2026-05-15-free-energy-consolidation-design.md``.
    """
    wf.paths = _resolve_simfolder(
        engine=engine,
        working_directory=working_directory,
        subdir=subdir,
    )
    wf.strained_structures = generate_structures(
        base_structure=structure,
        axes=["iso"],
        strain_range=strain_range,
        num_points=num_volumes,
    )
    wf.static_E = _static_energies_per_volume(
        strained_structures=wf.strained_structures.outputs.structure_list,
        engine=wf.paths.outputs.sub_engine,
    )
    wf.harmonic_grid = _harmonic_grid_over_volumes(
        strained_structures=wf.strained_structures.outputs.structure_list,
        engine=engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperatures=temperatures,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        working_directory=wf.paths.outputs.simfolder,
    )
    wf.qha = _fit_qha(
        energies=wf.static_E.outputs.energies_per_volume,
        volumes=wf.static_E.outputs.volumes,
        free_energy_per_T_V=wf.harmonic_grid.outputs.free_energy_per_T_V,
        entropy_per_T_V=wf.harmonic_grid.outputs.entropy_per_T_V,
        cv_per_T_V=wf.harmonic_grid.outputs.cv_per_T_V,
        temperatures=temperatures,
        pressure_GPa=pressure,
        eos_type=eos_type,
    )
    wf.synthesis = _pack_qha_output(
        structure=structure,
        qha_results=wf.qha.outputs.qha_results,
        volumes=wf.static_E.outputs.volumes,
        free_energy_per_T_V=wf.harmonic_grid.outputs.free_energy_per_T_V,
        entropy_per_T_V=wf.harmonic_grid.outputs.entropy_per_T_V,
        cv_per_T_V=wf.harmonic_grid.outputs.cv_per_T_V,
        temperatures=temperatures,
        pressure_GPa=pressure,
        simfolder=wf.paths.outputs.simfolder,
        keep_handles=keep_handles,
    )
    return wf.synthesis.outputs.free_energy_output
