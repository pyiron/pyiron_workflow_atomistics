"""dynaphopy MD-trajectory anharmonic phonon renormalisation workflow.

The single user-facing entry point is
:func:`calculate_phonon_md_renormalisation`.

Built on top of dynaphopy via a thin wrapper that exposes its functionality
as pyiron_workflow function-nodes and macros. The upstream package's name
is the authoritative source for behaviour and bug reports; this file
routes inputs/outputs through the pyiron_workflow Engine Protocol.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.engine import Engine
from pyiron_workflow_atomistics.physics.phonons._compat import (
    require_dynaphopy,
    require_phonopy,
)
from pyiron_workflow_atomistics.physics.phonons.output import (
    MdPhononOutput,
    PhononOutput,
)


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


def _multiplier_to_cell_vectors(
    primitive_cell: ArrayLike, multiplier: ArrayLike
) -> np.ndarray:
    """Convert an integer supercell multiplier into the 3x3 cell-vectors matrix.

    Two APIs in this workflow want different ``supercell`` semantics:

    - ``phonopy.Phonopy(supercell_matrix=...)`` and
      ``dynaphopy.interface.phonopy_link.ForceConstants(supercell=...)`` expect
      the **integer multiplier** (e.g. ``2*np.eye(3)``).
    - ``dynaphopy.dynamics.Dynamics(supercell=...)`` expects the **cell-vectors
      matrix** of the simulation cell.

    This converter makes the two-form duality explicit so the contract on
    ``trajectory_pack['supercell']`` is named rather than implicit-via-ASE.
    Uses the ASE convention ``supercell_cell = multiplier @ primitive_cell``
    (rows are lattice vectors).
    """
    P = _normalise_supercell_matrix(multiplier).astype(float)
    return P @ np.asarray(primitive_cell, dtype=float)


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
        "supercell": _multiplier_to_cell_vectors(structure.cell, resolved_fc2_supercell),
        "n_md_steps": production_steps,
        "time_step_fs": float(time_step),
        "md_temperature_mean": float(instantaneous_T.mean()),
        "md_temperature_std": float(instantaneous_T.std()),
    }
    return pack


def _build_phonopy_view(structure: Atoms, fc2_array: np.ndarray, supercell_matrix):
    """Build a phonopy.Phonopy with the supplied FC2 attached.

    Helper used by _project_with_dynaphopy. Kept separate so future v2
    extensions (NAC, custom path) can extend it without touching the
    projection logic.
    """
    require_phonopy()
    from phonopy import Phonopy

    from pyiron_workflow_atomistics.physics.phonons.harmonic import _ase_to_phonopy

    unitcell = _ase_to_phonopy(structure)
    phonon = Phonopy(unitcell=unitcell, supercell_matrix=supercell_matrix)
    phonon.force_constants = np.asarray(fc2_array)
    return phonon


def _harmonic_frequencies_at(phonon, q_points: np.ndarray) -> np.ndarray:
    """Evaluate the harmonic phonon frequencies at a list of q-points.

    Returns (n_q, n_band) in THz.
    """
    n_band = 3 * len(phonon.primitive)
    out = np.zeros((len(q_points), n_band))
    for i, q in enumerate(q_points):
        freqs = phonon.get_frequencies(q)
        out[i] = np.asarray(freqs)
    return out


def _ase_to_dynaphopy_structure(structure: Atoms, fc2_array, fc2_supercell_matrix):
    """Build a dynaphopy.atoms.Structure from an ASE Atoms with FC2 attached.

    The dynaphopy Structure carries the primitive cell along with the FC2
    and the (3x3 integer multiplier) supercell used during the FC2 fit.
    dynaphopy's Phonopy plumbing (``get_phonon``) passes that supercell
    array directly to ``phonopy.Phonopy(supercell_matrix=...)`` which
    expects an integer multiplier — NOT the supercell cell vectors.
    """
    require_dynaphopy()
    from dynaphopy.atoms import Structure as DynaphopyStructure
    from dynaphopy.interface.phonopy_link import ForceConstants

    cell = np.asarray(structure.cell)
    sc = np.asarray(fc2_supercell_matrix)
    # dynaphopy's Structure._supercell_matrix is a 1d int vector (diagonal-only).
    if sc.ndim == 2:
        sc_diag = np.diag(sc).astype(int)
        sc_int_matrix = sc.astype(int)
    else:
        sc_diag = sc.astype(int)
        sc_int_matrix = np.diag(sc_diag)

    dyn_structure = DynaphopyStructure(
        cell=cell,
        scaled_positions=np.asarray(structure.get_scaled_positions()),
        atomic_elements=list(structure.get_chemical_symbols()),
        primitive_matrix=np.eye(3),
    )
    dyn_structure.set_supercell_matrix(sc_diag)
    dyn_structure.set_force_constants(
        ForceConstants(np.asarray(fc2_array), supercell=sc_int_matrix)
    )
    return dyn_structure


@pwf.as_function_node("md_phonon_output")
def _project_with_dynaphopy(
    structure: Atoms,
    fc2_array: np.ndarray,
    resolved_fc2_supercell,
    trajectory_pack: dict,
    resolved_q_points: np.ndarray,
    temperature: float,
    power_spectra: bool,
    keep_handles: bool,
) -> MdPhononOutput:
    """Build dynaphopy.Quasiparticle, fit each q-point, pack into MdPhononOutput."""
    require_dynaphopy()
    from dynaphopy import Quasiparticle
    from dynaphopy.analysis.fitting import phonon_fitting_analysis
    from dynaphopy.dynamics import Dynamics

    phonon = _build_phonopy_view(structure, fc2_array, resolved_fc2_supercell)

    # Harmonic reference frequencies (always populated for comparison).
    harmonic_frequencies = _harmonic_frequencies_at(phonon, resolved_q_points)

    # Build a dynaphopy primitive Structure carrying the FC2 + supercell info.
    dyn_structure = _ase_to_dynaphopy_structure(
        structure, fc2_array, resolved_fc2_supercell
    )

    # Sanity check: trajectory_pack must have been generated against the same
    # supercell we're projecting with, otherwise the mode projection is wrong.
    expected_cell_vectors = _multiplier_to_cell_vectors(
        structure.cell, resolved_fc2_supercell
    )
    actual_cell_vectors = np.asarray(trajectory_pack["supercell"])
    if not np.allclose(actual_cell_vectors, expected_cell_vectors, atol=1e-6):
        raise ValueError(
            f"trajectory_pack['supercell'] ({actual_cell_vectors.tolist()}) "
            "does not match the cell vectors derived from structure.cell @ "
            f"resolved_fc2_supercell ({expected_cell_vectors.tolist()}). The "
            "MD trajectory was generated against a different supercell than "
            "the one being used for FC2 projection."
        )

    # Build dynaphopy Dynamics from the trajectory pack. dynaphopy expects:
    #   trajectory : complex ndarray (n_steps, n_atoms, 3) — Angstrom
    #   velocity   : Angstrom/ps consistent with `time` in ps.
    #                Pass `None` so dynaphopy reconstructs velocity from
    #                `np.gradient(positions, time_step)` — this is unit-clean
    #                and matches the convention used by dynaphopy's own
    #                example workflows. Passing ASE's `get_velocities()`
    #                values directly would feed dynaphopy values in
    #                Angstrom/AU_t (1 AU_t ~= 10.18 fs), biasing the
    #                power-spectrum peaks by a constant factor across all
    #                bands. `_run_nvt_trajectory` still records velocities
    #                in the trajectory_pack for diagnostics and future use.
    #   time       : ndarray (n_steps,) in picoseconds
    #   supercell  : 3x3 cell vectors of the MD simulation cell
    dynamics = Dynamics(
        structure=dyn_structure,
        trajectory=np.asarray(trajectory_pack["positions"], dtype=complex),
        velocity=None,
        time=np.asarray(trajectory_pack["time"]) * 1e-3,  # fs → ps
        supercell=actual_cell_vectors,
    )

    qp = Quasiparticle(dynamics)
    qp.set_temperature(temperature)
    qp.parameters.silent = True

    n_q = len(resolved_q_points)
    n_band = harmonic_frequencies.shape[1]
    renormalised = np.full((n_q, n_band), np.nan)
    linewidths = np.full((n_q, n_band), np.nan)
    failed: list[tuple[int, int]] = []

    spectra_blocks: list[np.ndarray] = []
    freq_grid: np.ndarray | None = None

    for iq, q in enumerate(resolved_q_points):
        q_arr = np.asarray(q, dtype=float)
        qp.set_reduced_q_vector(q_arr)
        try:
            ps = qp.get_power_spectrum_phonon()
            ps_frequencies = np.asarray(qp.parameters.frequency_range)
            data = phonon_fitting_analysis(
                np.asarray(ps),
                ps_frequencies,
                harmonic_frequencies=qp.get_frequencies(),
                thermal_expansion_shift=None,
                show_plots=False,
                use_degeneracy=qp.parameters.use_symmetry,
                show_occupancy=False,
            )
            positions = np.asarray(data["positions"], dtype=float)
            widths = np.asarray(data["widths"], dtype=float)
            # At Gamma (q == 0), the three acoustic bands are pinned to 0 by
            # translation invariance. dynaphopy clamps them explicitly in
            # ``get_commensurate_points_data``; mirror that idiom here so
            # callers get the physically correct values rather than fit noise
            # around zero.
            if np.allclose(q_arr, 0.0):
                positions[:3] = 0.0
                widths[:3] = 0.0
            renormalised[iq] = positions[:n_band]
            linewidths[iq] = widths[:n_band]

            if power_spectra:
                if freq_grid is None:
                    freq_grid = ps_frequencies
                spectra_blocks.append(np.asarray(ps).T)  # (n_band, n_freq)
        except Exception:  # noqa: BLE001 — dynaphopy may raise various fit errors
            failed.append((iq, -1))
            if power_spectra:
                spectra_blocks.append(np.full((n_band, 1), np.nan))

    converged = not failed
    if failed:
        n_total = n_q * n_band
        warnings.warn(
            f"Lorentzian fit failed for {len(failed)} of {n_total} (q, band) "
            "pairs; corresponding entries are NaN. Set power_spectra=True and "
            "inspect the raw spectra if you need to debug.",
            stacklevel=2,
        )

    if power_spectra and spectra_blocks:
        power_spectra_array = np.stack(spectra_blocks, axis=0)
    else:
        power_spectra_array = None

    out = MdPhononOutput(
        structure=structure,
        fc2_supercell_matrix=_normalise_supercell_matrix(resolved_fc2_supercell),
        temperature=float(temperature),
        q_points=np.asarray(resolved_q_points),
        harmonic_frequencies=harmonic_frequencies,
        renormalised_frequencies=renormalised,
        linewidths=linewidths,
        converged=converged,
        n_md_steps=int(trajectory_pack["n_md_steps"]),
        time_step_fs=float(trajectory_pack["time_step_fs"]),
        md_temperature_mean=float(trajectory_pack["md_temperature_mean"]),
        md_temperature_std=float(trajectory_pack["md_temperature_std"]),
        power_spectra=power_spectra_array,
        frequency_grid=freq_grid if power_spectra else None,
        quasiparticle=qp if keep_handles else None,
        dynamics=dynamics if keep_handles else None,
        phonopy=phonon if keep_handles else None,
    )

    # Auto-warn for bad MD on first run.
    healthy, issues = out.check_md_health()
    if not healthy:
        warnings.warn(
            "MD diagnostics indicate potential issues:\n  - "
            + "\n  - ".join(issues)
            + "\nThe renormalised frequencies / linewidths may be unreliable. "
            "See MdPhononOutput.check_md_health() for the heuristics.",
            stacklevel=2,
        )

    return out


@pwf.as_function_node("fc2_array")
def _select_or_compute_fc2(
    structure: Atoms,
    engine: Engine,
    resolved_fc2_supercell,
    fc2_source_tag: str,
    fc2_array_reused,
) -> np.ndarray:
    """Pick FC2 between reuse (from phono3py_output) and recomputation.

    pyiron_workflow macros can't branch on proxy `UserInput` values at graph
    construction time, so the if/elif lives here in a function-node that runs
    at execution time.
    """
    if fc2_source_tag == "reuse":
        if fc2_array_reused is None:
            raise RuntimeError(
                "Internal error: fc2_source_tag='reuse' but fc2_array_reused is None"
            )
        fc2_array = np.asarray(fc2_array_reused)
    elif fc2_source_tag == "recompute":
        fc2_array = _compute_fc2_from_scratch.node_function(
            structure=structure,
            engine=engine,
            resolved_fc2_supercell=resolved_fc2_supercell,
        )
    else:
        raise ValueError(f"Unknown fc2_source_tag: {fc2_source_tag!r}")
    return fc2_array


@pwf.api.as_macro_node("md_phonon_output")
def calculate_phonon_md_renormalisation(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix=None,
    temperature: float = 300.0,
    # MD plumbing
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,
    thermostat_time_constant: float = 100.0,
    seed=None,
    # q-point selection
    q_points=None,
    band_npoints: int = 30,
    # FC2 source — optional re-use from phono3py
    phono3py_output: PhononOutput | None = None,
    # output tiers
    power_spectra: bool = False,
    keep_handles: bool = False,
):
    """Compute anharmonic phonon renormalisation at finite T via dynaphopy.

    Reuses the existing Engine Protocol — FC2 force evaluations go through
    ``engine.calculate``; MD trajectory generation borrows the engine's
    calculator directly (per the spec's documented carve-out for step-by-step
    MD control).

    See spec: docs/design/specs/2026-05-15-dynaphopy-md-renormalisation-design.md
    """
    # Node 0: runtime arg resolution (proxy-safe).
    wf.defaults = _resolve_md_defaults(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        phono3py_output=phono3py_output,
        q_points=q_points,
        band_npoints=band_npoints,
        seed=seed,
    )

    # Node 1: FC2 source — recompute or reuse.
    wf.fc2 = _select_or_compute_fc2(
        structure=structure,
        engine=engine,
        resolved_fc2_supercell=wf.defaults.outputs.resolved_fc2_supercell,
        fc2_source_tag=wf.defaults.outputs.fc2_source_tag,
        fc2_array_reused=wf.defaults.outputs.fc2_array,
    )

    # Node 2: MD trajectory.
    wf.trajectory = _run_nvt_trajectory(
        structure=structure,
        engine=engine,
        resolved_fc2_supercell=wf.defaults.outputs.resolved_fc2_supercell,
        temperature=temperature,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        time_step=time_step,
        thermostat_time_constant=thermostat_time_constant,
        seed=wf.defaults.outputs.resolved_seed,
    )

    # Node 3: dynaphopy projection synthesis.
    wf.projection = _project_with_dynaphopy(
        structure=structure,
        fc2_array=wf.fc2.outputs.fc2_array,
        resolved_fc2_supercell=wf.defaults.outputs.resolved_fc2_supercell,
        trajectory_pack=wf.trajectory.outputs.trajectory_pack,
        resolved_q_points=wf.defaults.outputs.resolved_q_points,
        temperature=temperature,
        power_spectra=power_spectra,
        keep_handles=keep_handles,
    )

    return wf.projection.outputs.md_phonon_output
