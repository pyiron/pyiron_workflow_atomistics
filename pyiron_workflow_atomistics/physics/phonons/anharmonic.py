"""phono3py FC3 + lattice thermal conductivity workflow.

The single user-facing entry point is :func:`calculate_phonon_thermal_conductivity`.
Everything else in this module is a private node or helper.
"""

from __future__ import annotations

from typing import Literal

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


@pwf.as_function_node("fc3_supercell_matrix_out", "fc_calculator_out")
def _resolve_defaults(
    fc2_supercell_matrix,
    fc3_supercell_matrix,
    number_of_snapshots,
    fc_calculator,
    born_charges,
    epsilon_inf,
):
    """Runtime guard + default resolution for the macro.

    Checks polar kwargs (raises before any phono3py import), defaults
    fc3_supercell_matrix to fc2_supercell_matrix when not supplied, and
    auto-selects 'symfc' for random-displacement mode.
    """
    _check_polar_unsupported(born_charges=born_charges, epsilon_inf=epsilon_inf)
    if fc3_supercell_matrix is None:
        fc3_supercell_matrix = fc2_supercell_matrix
    if number_of_snapshots is not None and fc_calculator is None:
        fc_calculator = "symfc"
    return fc3_supercell_matrix, fc_calculator


@pwf.as_function_node("fc3_supercells")
def _generate_fc3_supercells(
    structure: Atoms,
    fc2_supercell_matrix,
    fc3_supercell_matrix,
    displacement_distance: float = 0.03,
    is_plusminus="auto",
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
    fc2_supercell_matrix,
    fc3_supercell_matrix,
    displacement_distance: float,
    is_plusminus,
    cutoff_pair_distance,
    number_of_snapshots,
    random_seed,
    fc_calculator,
    fc2_engine_outputs,
    fc3_engine_outputs,
    temperatures,
    q_mesh,
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

    n_fc2_forces = len(fc2_engine_outputs)
    n_fc2_expected = len(ph3.phonon_supercells_with_displacements)
    if n_fc2_forces != n_fc2_expected:
        raise RuntimeError(
            f"FC2 force/supercell mismatch: {n_fc2_forces} forces vs "
            f"{n_fc2_expected} expected. "
            "Displacement kwargs likely drifted between generation and synthesis."
        )
    n_fc3_forces = len(fc3_engine_outputs)
    n_fc3_expected = len(ph3.supercells_with_displacements)
    if n_fc3_forces != n_fc3_expected:
        raise RuntimeError(
            f"FC3 force/supercell mismatch: {n_fc3_forces} forces vs "
            f"{n_fc3_expected} expected. "
            "Displacement kwargs likely drifted between generation and synthesis."
        )

    fc2_forces = _stack_forces(fc2_engine_outputs)
    fc3_forces = _stack_forces(fc3_engine_outputs)

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


from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _generate_fc2_supercells,
)


@pwf.api.as_macro_node("phonon_output")
def calculate_phonon_thermal_conductivity(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    fc3_supercell_matrix=None,
    temperatures=(300.0,),
    q_mesh=(11, 11, 11),
    # phono3py.generate_displacements kwargs
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    cutoff_pair_distance: float | None = None,
    number_of_snapshots: int | None = None,
    random_seed: int | None = None,
    fc_calculator: str | None = None,
    # output tiers
    mode_resolved: bool = False,
    harmonic_observables: bool = False,
    keep_handles: bool = False,
    # polar-material kwargs (v1: must be None)
    born_charges=None,
    epsilon_inf=None,
):
    """Compute lattice thermal conductivity κ(T) via phono3py.

    Reuses the existing Engine Protocol — every supercell force evaluation
    goes through ``engine.calculate``. Returns a :class:`PhononOutput`.

    See spec: docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md
    """
    # Node 0: runtime polar-guard + default resolution.
    # (Cannot call _check_polar_unsupported directly in the macro body because
    # the macro body runs during __init__ with proxy UserInput objects, not
    # real values. This node runs it at execution time.)
    wf.defaults = _resolve_defaults(
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
        number_of_snapshots=number_of_snapshots,
        fc_calculator=fc_calculator,
        born_charges=born_charges,
        epsilon_inf=epsilon_inf,
    )

    wf.fc2_supercells = _generate_fc2_supercells(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    wf.fc3_supercells = _generate_fc3_supercells(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=wf.defaults.outputs.fc3_supercell_matrix_out,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
    )
    wf.fc2_eval = _evaluate_supercells(
        supercells=wf.fc2_supercells.outputs.fc2_supercells,
        engine=engine,
        prefix="fc2_disp_",
    )
    wf.fc3_eval = _evaluate_supercells(
        supercells=wf.fc3_supercells.outputs.fc3_supercells,
        engine=engine,
        prefix="fc3_disp_",
    )
    wf.synthesis = _run_phono3py_thermal_conductivity(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=wf.defaults.outputs.fc3_supercell_matrix_out,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
        fc_calculator=wf.defaults.outputs.fc_calculator_out,
        fc2_engine_outputs=wf.fc2_eval.outputs.engine_outputs,
        fc3_engine_outputs=wf.fc3_eval.outputs.engine_outputs,
        temperatures=temperatures,
        q_mesh=q_mesh,
        mode_resolved=mode_resolved,
        harmonic_observables=harmonic_observables,
        keep_handles=keep_handles,
    )

    return wf.synthesis.outputs.phonon_output
