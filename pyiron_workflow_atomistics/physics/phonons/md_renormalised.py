"""dynaphopy MD-trajectory anharmonic phonon renormalisation workflow.

The single user-facing entry point is
:func:`calculate_phonon_md_renormalisation`.

Built on top of dynaphopy via a thin wrapper that exposes its functionality
as pyiron_workflow function-nodes and macros. The upstream package's name
is the authoritative source for behaviour and bug reports; this file
routes inputs/outputs through the pyiron_workflow Engine Protocol.
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.engine import Engine
from pyiron_workflow_atomistics.physics.phonons._compat import require_phonopy
from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput


def _normalise_supercell_matrix(m: ArrayLike) -> np.ndarray:
    """Accept int / list[int] of length 3 / (3,3) ndarray; return (3,3) int.

    Local copy of harmonic.py's helper to avoid the import edge case during
    arg validation. The two are byte-identical.
    """
    arr = np.asarray(m)
    if arr.ndim == 0:
        return int(arr) * np.eye(3, dtype=int)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(
                f"supercell_matrix 1d shape must be (3,), got {arr.shape}"
            )
        return np.diag(arr.astype(int))
    if arr.ndim == 2:
        if arr.shape != (3, 3):
            raise ValueError(
                f"supercell_matrix 2d shape must be (3,3), got {arr.shape}"
            )
        return arr.astype(int)
    raise ValueError(
        f"supercell_matrix must be int / (3,) / (3,3); got {arr.shape}"
    )


def _auto_band_path(cell: np.ndarray, npoints: int) -> np.ndarray:
    """ASE-auto-derived high-symmetry band path for the given primitive cell."""
    from ase.dft.kpoints import bandpath as ase_bandpath

    bp = ase_bandpath(path=None, cell=cell, npoints=npoints)
    return np.asarray(bp.kpts)


@pwf.as_function_node(
    "resolved_fc2_supercell",
    "resolved_q_points",
    "resolved_seed",
    "fc2_source_tag",
    "fc2_array",
)
def _resolve_md_defaults(
    structure: Atoms,
    fc2_supercell_matrix,
    phono3py_output: PhononOutput | None,
    q_points,
    band_npoints: int,
    seed,
):
    """Execution-time arg resolver for the MD-renormalisation macro.

    Validates the four-case coupling table:
        - both None → ValueError
        - both supplied with mismatched supercells → ValueError
        - phono3py_output with fc2=None → ValueError (advise keep_handles=True)
        - else: derive resolved_fc2_supercell + (optionally) materialise fc2_array

    Auto-derives q_points from an ASE bandpath when q_points is None. Fills
    the seed via SeedSequence().entropy when seed is None.
    """
    # ── FC2 source coupling ────────────────────────────────────────────
    if fc2_supercell_matrix is None and phono3py_output is None:
        raise ValueError(
            "Must supply fc2_supercell_matrix or phono3py_output (got neither). "
            "See docs/design/specs/2026-05-15-dynaphopy-md-renormalisation-design.md."
        )

    if phono3py_output is not None:
        if phono3py_output.fc2 is None:
            raise ValueError(
                "phono3py_output.fc2 is None; re-run "
                "calculate_phonon_thermal_conductivity with keep_handles=True "
                "to enable FC2 reuse, or pass fc2_supercell_matrix instead to "
                "recompute FC2 in this macro."
            )
        upstream_sc = _normalise_supercell_matrix(
            phono3py_output.fc2_supercell_matrix
        )
        if fc2_supercell_matrix is not None:
            user_sc = _normalise_supercell_matrix(fc2_supercell_matrix)
            if not np.array_equal(user_sc, upstream_sc):
                raise ValueError(
                    f"fc2_supercell_matrix={user_sc.tolist()} disagrees with "
                    f"phono3py_output.fc2_supercell_matrix={upstream_sc.tolist()}; "
                    "supercell matrices must match if both are supplied."
                )
        resolved_fc2_supercell = upstream_sc
        fc2_source_tag = "reuse"
        fc2_array = np.asarray(phono3py_output.fc2)
    else:
        resolved_fc2_supercell = _normalise_supercell_matrix(fc2_supercell_matrix)
        fc2_source_tag = "recompute"
        fc2_array = None

    # ── q-point selection ─────────────────────────────────────────────
    if q_points is None:
        resolved_q_points = _auto_band_path(
            cell=np.asarray(structure.cell), npoints=band_npoints
        )
    else:
        resolved_q_points = np.atleast_2d(np.asarray(q_points, dtype=float))
        if resolved_q_points.shape[-1] != 3:
            raise ValueError(
                f"q_points must be (n, 3) or (3,); got shape {resolved_q_points.shape}"
            )

    # ── seed plumbing ─────────────────────────────────────────────────
    if seed is None:
        resolved_seed = int(np.random.SeedSequence().entropy % (2**32))
    else:
        resolved_seed = int(seed)

    return (
        resolved_fc2_supercell,
        resolved_q_points,
        resolved_seed,
        fc2_source_tag,
        fc2_array,
    )


@pwf.as_function_node("fc2_array")
def _compute_fc2_from_scratch(
    structure: Atoms,
    engine: Engine,
    resolved_fc2_supercell,
) -> np.ndarray:
    """Run FC2 displacements, evaluate forces via the engine, fit FC2 via phonopy.

    Reuses the v0.0.7 building blocks (`_generate_fc2_supercells`,
    `_evaluate_supercells`) and feeds the resulting forces into a
    phonopy.Phonopy view that owns the FC2 fit.
    """
    require_phonopy()
    from phonopy import Phonopy

    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
    )
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _ase_to_phonopy,
        _generate_fc2_supercells,
    )

    # Generate displaced supercells (FD, deterministic).
    fc2_supercells = _generate_fc2_supercells.node_function(
        structure=structure,
        fc2_supercell_matrix=resolved_fc2_supercell,
    )
    # Evaluate forces on each supercell.
    fc2_engine_outputs = _evaluate_supercells.node_function(
        supercells=fc2_supercells,
        engine=engine,
        prefix="fc2_disp_",
    )
    if not all(o.converged for o in fc2_engine_outputs):
        failed = [i for i, o in enumerate(fc2_engine_outputs) if not o.converged]
        raise RuntimeError(
            f"FC2 force calc failed for supercells {failed}; check engine logs."
        )

    # Build a phonopy view, attach forces, fit FC2.
    unitcell = _ase_to_phonopy(structure)
    phonon = Phonopy(unitcell=unitcell, supercell_matrix=resolved_fc2_supercell)
    phonon.generate_displacements()
    forces = np.stack(
        [np.asarray(o.final_forces) for o in fc2_engine_outputs], axis=0
    )
    if forces.shape[0] != len(phonon.supercells_with_displacements):
        raise RuntimeError(
            f"FC2 force/supercell count mismatch: {forces.shape[0]} forces vs "
            f"{len(phonon.supercells_with_displacements)} expected supercells."
        )
    phonon.forces = forces
    phonon.produce_force_constants()
    fc2_array = np.asarray(phonon.force_constants)
    return fc2_array


@pwf.as_function_node("trajectory_pack")
def _run_nvt_trajectory(
    structure: Atoms,
    engine: Engine,
    resolved_fc2_supercell,
    temperature: float,
    equilibration_steps: int,
    production_steps: int,
    time_step: float,
    thermostat_time_constant: float,
    seed: int,
) -> dict:
    """Run Langevin NVT MD on a supercell built from `structure`.

    Discards `equilibration_steps`, records the next `production_steps`
    into a trajectory pack of plain ndarrays + scalars suitable for
    dynaphopy's `Dynamics` constructor downstream.
    """
    from ase import units

    # Build the supercell to actually run MD on. dynaphopy projects the
    # supercell trajectory onto modes via the FC2 supercell, so we MUST
    # run MD at the FC2 supercell size.
    from ase.build import make_supercell
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    supercell_atoms = make_supercell(structure, resolved_fc2_supercell)
    # Attach the engine's calculator. We deliberately don't go through
    # engine.calculate here because MD wants step-by-step ASE control.
    if hasattr(engine, "calculator"):
        supercell_atoms.calc = engine.calculator
    else:
        raise RuntimeError(
            "Engine must expose a `calculator` attribute for MD trajectory "
            "generation. Current MD path supports ASEEngine only; pass an "
            "ASEEngine instance."
        )

    rng = np.random.default_rng(seed)  # noqa: F841 — kept for parity with future expanded seeding
    MaxwellBoltzmannDistribution(
        supercell_atoms,
        temperature_K=temperature,
        rng=np.random.RandomState(seed),
    )

    dt = time_step * units.fs
    ttime = thermostat_time_constant * units.fs
    dyn = Langevin(
        supercell_atoms,
        timestep=dt,
        temperature_K=temperature,
        friction=1.0 / ttime,
        rng=np.random.RandomState(seed),
    )

    # Equilibration — discarded.
    if equilibration_steps > 0:
        dyn.run(equilibration_steps)

    # Production — recorded.
    positions = np.zeros((production_steps, len(supercell_atoms), 3))
    velocities = np.zeros_like(positions)
    times = np.zeros(production_steps)
    instantaneous_T = np.zeros(production_steps)

    step_counter = {"i": 0}

    def record_step():
        i = step_counter["i"]
        # `if i < production_steps` (rather than an early-return) so pyiron_workflow's
        # AST output-parser doesn't see two `return` statements in the enclosing node.
        if i < production_steps:
            positions[i] = supercell_atoms.get_positions()
            velocities[i] = supercell_atoms.get_velocities()
            times[i] = i * time_step  # fs
            instantaneous_T[i] = supercell_atoms.get_temperature()
            step_counter["i"] += 1

    dyn.attach(record_step, interval=1)
    dyn.run(production_steps)

    pack = {
        "positions": positions,
        "velocities": velocities,
        "time": times,
        "supercell": np.asarray(supercell_atoms.cell),
        "n_md_steps": production_steps,
        "time_step_fs": float(time_step),
        "md_temperature_mean": float(instantaneous_T.mean()),
        "md_temperature_std": float(instantaneous_T.std()),
    }
    return pack
