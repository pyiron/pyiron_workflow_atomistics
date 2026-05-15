"""Harmonic free energy via phonopy FC2 — single user-facing entry point.

Built on top of phonopy via thin wrappers around the FC2 helpers in
`physics.phonons.harmonic`. The κ(T) workflow continues to own those
helpers; we import them here without behavioural change.
"""

from __future__ import annotations

import os

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.phonons._compat import require_phonopy
from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
    _check_all_converged,
    _evaluate_supercells,
    _stack_forces,
)
from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _ase_to_phonopy,
    _compute_harmonic_observables,
    _generate_fc2_supercells,
    _normalise_supercell_matrix,
)


@pwf.as_function_node("simfolder", "sub_engine")
def _resolve_simfolder(
    engine: Engine,
    working_directory: str,
    subdir: str,
):
    """Resolve the absolute simfolder path AND create it on disk.

    Joins ``working_directory`` and ``subdir``, absolute-paths the result,
    mkdirs it (``exist_ok=True``), and wraps the engine into a child that
    runs inside that directory. The mkdir side effect is intentional: by
    the time downstream nodes execute, the simfolder is guaranteed to exist.

    Lives in a function-node so the path-join executes at run time with real
    string values, not the proxy UserInput objects a macro body sees during
    graph construction.
    """
    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)
    sub_engine = engine.with_working_directory(simfolder)
    return simfolder, sub_engine


@pwf.as_function_node("phonopy_view")
def _produce_fc2_view(
    structure: Atoms,
    fc2_supercell_matrix,
    fc2_engine_outputs: list,
    displacement_distance: float,
    is_plusminus,
):
    """Build a phonopy.Phonopy with FC2 fitted from supercell forces."""
    require_phonopy()
    import phonopy

    _check_all_converged(fc2_engine_outputs, label="FC2")
    sc = _normalise_supercell_matrix(fc2_supercell_matrix)

    unitcell = _ase_to_phonopy(structure)
    phonon = phonopy.Phonopy(
        unitcell=unitcell, supercell_matrix=sc, primitive_matrix="auto"
    )
    phonon.generate_displacements(
        distance=displacement_distance, is_plusminus=is_plusminus
    )
    forces = _stack_forces(fc2_engine_outputs)
    if forces.shape[0] != len(phonon.supercells_with_displacements):
        raise RuntimeError(
            f"FC2 force/supercell mismatch: {forces.shape[0]} forces vs "
            f"{len(phonon.supercells_with_displacements)} expected supercells. "
            "displacement kwargs likely drifted between generation and synthesis."
        )
    phonon.forces = forces
    phonon.produce_force_constants()
    return phonon


class _Phono3pyShim:
    """Minimal adapter exposing the four attributes _compute_harmonic_observables reads.

    `_compute_harmonic_observables` was originally written against a Phono3py
    object; for the harmonic-only path we hand it a Phonopy view with the
    same four attributes. Keeps the helper signature unchanged.
    """

    def __init__(self, phonopy_view) -> None:
        self.phonon_primitive = phonopy_view.primitive
        self.phonon_supercell_matrix = phonopy_view.supercell_matrix
        self.fc2 = phonopy_view.force_constants


@pwf.as_function_node("free_energy_output")
def _pack_harmonic_output(
    structure: Atoms,
    phonopy_view,
    temperatures,
    fc2_supercell_matrix,
    displacement_distance: float,
    simfolder: str,
    keep_handles: bool,
) -> FreeEnergyOutput:
    """Compute thermal properties from the FC2 view and pack into FreeEnergyOutput.

    Note: phonopy's ``ThermalProperties`` reports ``free_energy`` in kJ/mol
    and ``entropy``/``heat_capacity`` in J/K/mol per *primitive cell* (not
    per atom). The spec for ``FreeEnergyOutput`` documents eV / (eV/K) per
    **primitive-cell atom**, so we divide by both ``ev_to_kj_mol`` and the
    number of atoms in phonopy's primitive cell. For users who pass a
    conventional (non-primitive) cell, phonopy's ``primitive_matrix="auto"``
    decides the primitive cell — e.g. ``bulk("Al", "fcc", cubic=True)``
    (4-atom conventional) reduces to a 1-atom fcc primitive.

    Downstream callers that expect phonopy's native units per primitive
    cell (e.g. ``phonopy.qha.QHA`` in
    ``quasiharmonic._harmonic_grid_over_volumes``) must multiply back by
    ``c.eV * c.Avogadro / 1000`` (= ``EvTokJmol``) AND by
    ``n_atoms_primitive`` (stashed in ``report["n_atoms_primitive"]``).
    """
    import scipy.constants as c

    T = np.asarray(temperatures, dtype=float)
    band_structure, dos, free_energy_dict = _compute_harmonic_observables(
        ph3=_Phono3pyShim(phonopy_view), temperatures=T
    )
    # phonopy → eV / (eV/K) per primitive-cell atom.
    n_atoms_primitive = len(phonopy_view.primitive)
    ev_to_kj_mol = c.eV * c.Avogadro / 1000.0  # ≈ 96.485
    F = np.asarray(free_energy_dict["F"]) / (ev_to_kj_mol * n_atoms_primitive)
    S = np.asarray(free_energy_dict["S"]) / (ev_to_kj_mol * 1000.0 * n_atoms_primitive)
    Cv = np.asarray(free_energy_dict["Cv"]) / (
        ev_to_kj_mol * 1000.0 * n_atoms_primitive
    )

    elements = list(dict.fromkeys(structure.get_chemical_symbols()))
    return FreeEnergyOutput(
        mode="harmonic",
        reference_phase="solid",
        free_energy=float(F[0]),
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=0.0,
        n_atoms=len(structure),
        elements=elements,
        simfolder=simfolder,
        report={
            "method": "harmonic",
            "fc2_supercell_matrix": _normalise_supercell_matrix(
                fc2_supercell_matrix
            ).tolist(),
            "displacement_distance": float(displacement_distance),
            "n_atoms_primitive": int(n_atoms_primitive),
        },
        temperature_array=T,
        free_energy_array=F,
        entropy=float(S[0]),
        heat_capacity=float(Cv[0]),
        entropy_array=S,
        heat_capacity_array=Cv,
        phonopy_handle=phonopy_view if keep_handles else None,
        band_structure=band_structure if keep_handles else None,
        phonon_dos=dos if keep_handles else None,
    )


@pwf.api.as_macro_node("free_energy_output")
def harmonic_free_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0),
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    working_directory: str = ".",
    subdir: str = "harmonic_free_energy",
    keep_handles: bool = False,
):
    """Helmholtz free energy F(T), entropy S(T), heat capacity Cv(T) at fixed volume.

    Returns
    -------
    FreeEnergyOutput
        ``mode="harmonic"``, ``reference_phase="solid"``. The scalar
        ``free_energy`` is the value at the *lowest* T in ``temperatures``
        (typically T=0, i.e. zero-point energy). Curves are in
        ``temperature_array`` / ``free_energy_array`` / ``entropy_array`` /
        ``heat_capacity_array``.

    See spec: docs/design/specs/2026-05-15-free-energy-consolidation-design.md
    """
    wf.paths = _resolve_simfolder(
        engine=engine,
        working_directory=working_directory,
        subdir=subdir,
    )

    wf.fc2_supercells = _generate_fc2_supercells(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    wf.fc2_eval = _evaluate_supercells(
        supercells=wf.fc2_supercells.outputs.fc2_supercells,
        engine=wf.paths.outputs.sub_engine,
        prefix="fc2_disp_",
    )
    wf.fc2_view = _produce_fc2_view(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc2_engine_outputs=wf.fc2_eval.outputs.engine_outputs,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    wf.synthesis = _pack_harmonic_output(
        structure=structure,
        phonopy_view=wf.fc2_view.outputs.phonopy_view,
        temperatures=temperatures,
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=displacement_distance,
        simfolder=wf.paths.outputs.simfolder,
        keep_handles=keep_handles,
    )
    return wf.synthesis.outputs.free_energy_output
