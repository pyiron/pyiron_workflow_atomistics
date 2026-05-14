"""phono3py FC3 + lattice thermal conductivity workflow.

The single user-facing entry point is :func:`calculate_phonon_thermal_conductivity`.
Everything else in this module is a private node or helper.
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _build_phono3py,
    _phonopy_to_ase,
)


def _check_polar_unsupported(
    *,
    born_charges: np.ndarray | None,
    epsilon_inf: np.ndarray | None,
) -> None:
    """Raise NotImplementedError if the caller asked for polar-material support.

    v1 is metals/non-polar only. The follow-up that adds NAC is tracked under
    "NAC / BORN effective charges" in the design spec.
    """
    if born_charges is not None or epsilon_inf is not None:
        raise NotImplementedError(
            "Non-analytic correction (BORN + ε∞) is not supported in v1; "
            "see the 'NAC / BORN effective charges' follow-up at the end of "
            "docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md."
        )


@pwf.as_function_node("fc3_supercells")
def _generate_fc3_supercells(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
    displacement_distance: float = 0.03,
    is_plusminus: str | bool = "auto",
    cutoff_pair_distance: float | None = None,
    number_of_snapshots: int | None = None,
    random_seed: int | None = None,
) -> list[Atoms]:
    """FC3 displaced supercells via phono3py.generate_displacements.

    Finite-difference path is used when number_of_snapshots is None;
    random-displacement path (and symfc fitting) lands in Task 11.
    """
    if number_of_snapshots is not None:
        # Filled in by Task 11. Until then, refuse random mode loudly so a
        # user who passes the kwarg too early gets a clear message.
        raise NotImplementedError(
            "Random-displacement FC3 sampling is added in a later task; "
            "set number_of_snapshots=None for the FD path."
        )
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )
    fc3_supercells = [
        _phonopy_to_ase(s) for s in ph3.supercells_with_displacements
    ]
    return fc3_supercells


from pyiron_workflow_atomistics.engine import Engine, EngineOutput, calculate


@pwf.as_function_node("engine_outputs")
def _evaluate_supercells(
    supercells: list[Atoms],
    engine: Engine,
    prefix: str,
) -> list[EngineOutput]:
    """Loop ``calculate`` over a list of supercells, routing each to its own subdir.

    Mirrors ``physics.bulk.evaluate_structures`` — the canonical "fan out
    `calculate` over a list of structures" pattern in this codebase.
    """
    engine_outputs: list[EngineOutput] = []
    for i, supercell in enumerate(supercells):
        sub_engine = engine.with_working_directory(f"{prefix}{i:04d}")
        engine_outputs.append(
            calculate.node_function(structure=supercell, engine=sub_engine)
        )
    return engine_outputs


import warnings

from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _normalise_supercell_matrix,
)
from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput


def _check_all_converged(engine_outputs, label: str) -> None:
    """Raise RuntimeError listing failed supercell indices + working_directory."""
    failed = [
        (i, getattr(out.final_structure, "info", {}).get("working_directory", "<unknown>"))
        for i, out in enumerate(engine_outputs)
        if not out.converged
    ]
    if failed:
        details = ", ".join(f"{i} ({wd})" for i, wd in failed)
        raise RuntimeError(
            f"Force calc failed for {label} supercells: {details}"
        )


def _stack_forces(engine_outputs) -> np.ndarray:
    """(n_supercells, n_atoms, 3) — phono3py's expected forces layout."""
    return np.stack([np.asarray(o.final_forces) for o in engine_outputs], axis=0)


def _kappa_voigt_to_tensor(kappa_voigt: np.ndarray) -> np.ndarray:
    """Convert (n_T, 6) Voigt → (n_T, 3, 3) full tensor.

    phono3py returns κ as (n_T, 6) in (xx, yy, zz, yz, xz, xy) order.
    """
    n_T = kappa_voigt.shape[0]
    out = np.zeros((n_T, 3, 3))
    for t in range(n_T):
        xx, yy, zz, yz, xz, xy = kappa_voigt[t]
        out[t] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return out


@pwf.as_function_node("phonon_output")
def _run_phono3py_thermal_conductivity(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
    displacement_distance: float,
    is_plusminus: str | bool,
    cutoff_pair_distance: float | None,
    number_of_snapshots: int | None,
    random_seed: int | None,
    fc_calculator: str | None,
    fc2_engine_outputs: list,
    fc3_engine_outputs: list,
    temperatures: ArrayLike,
    q_mesh: ArrayLike,
    mode_resolved: bool,
    harmonic_observables: bool,
    keep_handles: bool,
) -> PhononOutput:
    """Synthesis node: rebuild Phono3py, attach forces, fit FCs, run BTE."""
    _check_all_converged(fc2_engine_outputs, label="FC2")
    _check_all_converged(fc3_engine_outputs, label="FC3")

    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    # Re-generate displacements identically so the dataset matches the forces.
    ph3.generate_fc2_displacements(
        distance=displacement_distance, is_plusminus=is_plusminus
    )
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )

    fc2_forces = _stack_forces(fc2_engine_outputs)
    fc3_forces = _stack_forces(fc3_engine_outputs)
    if fc2_forces.shape[0] != len(ph3.phonon_supercells_with_displacements):
        raise RuntimeError(
            f"FC2 force/supercell mismatch: {fc2_forces.shape[0]} forces vs "
            f"{len(ph3.phonon_supercells_with_displacements)} expected. "
            "Displacement kwargs likely drifted between generation and synthesis."
        )
    if fc3_forces.shape[0] != len(ph3.supercells_with_displacements):
        raise RuntimeError(
            f"FC3 force/supercell mismatch: {fc3_forces.shape[0]} forces vs "
            f"{len(ph3.supercells_with_displacements)} expected. "
            "Displacement kwargs likely drifted between generation and synthesis."
        )

    ph3.phonon_forces = fc2_forces
    ph3.forces = fc3_forces
    ph3.produce_fc2()
    ph3.produce_fc3(fc_calculator=fc_calculator)

    T = np.asarray(temperatures, dtype=float)
    mesh = np.asarray(q_mesh, dtype=int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ph3.mesh_numbers = mesh
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=T, write_kappa=False)
        converged = not any(
            "not converged" in str(w.message).lower() for w in caught
        )

    tc = ph3.thermal_conductivity
    kappa = _kappa_voigt_to_tensor(np.asarray(tc.kappa[0]))  # (n_T, 3, 3)

    return PhononOutput(
        structure=structure,
        fc2_supercell_matrix=_normalise_supercell_matrix(fc2_supercell_matrix),
        fc3_supercell_matrix=_normalise_supercell_matrix(fc3_supercell_matrix),
        temperatures=T,
        kappa=kappa,
        converged=converged,
    )
