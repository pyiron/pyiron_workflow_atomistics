# dynaphopy MD-trajectory anharmonic phonon renormalisation

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-15 |
| Repo | `pyiron/pyiron_workflow_atomistics` |
| Scope | dynaphopy: NVT MD → velocity-ACF mode projection → renormalised frequencies + linewidths. Optional reuse of FC2 from an existing `PhononOutput`. |
| Out of scope (v1) | NVE / NPT ensembles, multi-temperature MD per call, NAC for polar materials (same v2 follow-up as phono3py), validation against published dispersion (validation notebook only). |
| Predecessor spec | `docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md` (v0.0.7 phono3py + harmonic side-products). The dataclass + lazy-import patterns + macro-graph shape are deliberately mirrored here. |

## Problem

The v0.0.7 phonons subpackage covers harmonic FC2 (phonopy) and 3rd-order perturbative κ(T) (phono3py). Both rely on perturbation theory around the 0 K crystal — fine for moderate anharmonicity, but for systems near a structural transition, soft modes, or high T relative to the Debye temperature, the perturbative expansion misses physics that's clearly visible in MD: mode softening, large lifetimes from non-Lorentzian peaks, full anharmonicity at finite T.

dynaphopy fills that gap: it takes a finite-T MD trajectory plus the harmonic FC2, projects the velocity autocorrelation onto the harmonic eigenmodes, fits Lorentzians to the per-mode power spectra, and returns renormalised frequencies and linewidths at the chosen q-points. The renormalisation captures the full anharmonic effect at the trajectory's temperature with no perturbative truncation.

What needs design is (a) how the user expresses the workflow without bolting on a parallel vocabulary for MD parameters, (b) how dynaphopy's MD-projection results fit into a structured output dataclass alongside the existing `PhononOutput`, (c) how FC2 is sourced (recompute vs reuse from a prior phono3py run), and (d) handholding for non-experts whose first MD run may be subtly broken (drift, sampling, ensemble mistake).

## Approach

A new `pyiron_workflow_atomistics.physics.phonons.md_renormalised` module with one user-facing macro `calculate_phonon_md_renormalisation` that returns a structured `MdPhononOutput` dataclass. Internally it builds a 4- or 5-node macro graph: arg resolution → optional FC2 (re)computation → NVT MD trajectory → dynaphopy mode-projection → packaged output.

`dynaphopy` ships as a **superset optional extra**: `pip install pyiron_workflow_atomistics[phonons-md]` pulls in `[phonons]` plus `dynaphopy`. The base install is unaffected; users who only want κ keep the smaller `[phonons]` group.

Six deliberate API decisions, each motivated below:

- **`q_points=None` auto-derives a high-symmetry band path via `ase.dft.kpoints.bandpath`.** Matches what the harmonic side already does in `_compute_harmonic_observables` so the renormalised dispersion plots cleanly on top of the harmonic dispersion. Non-experts get a meaningful visualisation by default; power users override the kwarg. The macro never silently expands a phono3py mesh of hundreds of q-points (which would be 10×+ slower).
- **NVT-Langevin only in v1, with explicit thermalisation + production phases.** Finite-T phonon renormalisation expects a thermalised constant-T trajectory. NVE drifts; NPT introduces cell fluctuations that distort the projection. Exposing the choice via `CalcInputMD` would let users pick combinations that produce noise. v2 can add NVE for systems where Langevin stochasticity contaminates lifetimes.
- **FC2 source is either-or: pass `fc2_supercell_matrix` to recompute, OR `phono3py_output` to reuse.** Either-or beats forcing one upstream pattern. The macro validates at entry that exactly one is supplied (or both with matching supercell matrices) and short-circuits FC2 computation when reuse is possible.
- **Dedicated `MdPhononOutput` dataclass, not an extension of `PhononOutput`.** The two workflows produce different physical quantities (perturbative RTA κ vs MD-projected renormalised frequencies). Sharing a dataclass would invite confusion about which fields apply to which workflow. Keeping them separate is documentation by structure.
- **`harmonic_frequencies` always populated alongside `renormalised_frequencies`.** Without the harmonic reference, a user can't tell whether the renormalisation is meaningful softening or fitting noise. Free to compute from the phonopy FC2 view that's already built; populated unconditionally.
- **`md_temperature_mean` + `md_temperature_std` always populated + a `check_md_health()` method + auto-warning at macro completion.** Non-experts need to know when their first run is subtly broken. Statistics block is cheap; the health check encodes two ironclad heuristics (drift tolerance, Langevin σ_T rule of thumb); the auto-warn surfaces issues on the first run without the user having to know to look.

## Module-header convention

Each phonon-workflow module starts with a docstring naming the upstream package it wraps. This is added to all three workflow files (the dynaphopy one as new, the phonopy and phono3py ones as updates):

```python
"""<one-line purpose>.

Built on top of <phonopy | phono3py | dynaphopy> via a thin wrapper that exposes
its functionality as pyiron_workflow function-nodes and macros. The upstream
package's name is the authoritative source for behaviour and bug reports;
this file routes inputs/outputs through the pyiron_workflow Engine Protocol.
"""
```

The convention is documented in this spec so future contributors keep it consistent across the subpackage.

## Components

```
pyiron_workflow_atomistics/physics/phonons/
├── __init__.py          # add MdPhononOutput, calculate_phonon_md_renormalisation
├── _compat.py           # add require_dynaphopy()
├── output.py            # add MdPhononOutput dataclass + check_md_health()
├── harmonic.py          # update header per convention; no functional change
├── anharmonic.py        # update header per convention; no functional change
└── md_renormalised.py   # NEW
```

### `MdPhononOutput` (in `output.py`)

```python
@dataclass
class MdPhononOutput:
    """Structured result of a dynaphopy MD-trajectory mode-projection workflow.

    Required fields are always populated. Optional fields are populated only
    when the corresponding macro flag is on:
        power_spectra=True  → power_spectra, frequency_grid
        keep_handles=True   → quasiparticle, dynamics, phonopy

    MD health diagnostics
    ---------------------
    The two fields below let you sanity-check the NVT segment that drove the
    projection. Anharmonic renormalisation results are only as good as the
    underlying trajectory; if the diagnostics look bad, treat the
    ``renormalised_frequencies`` and ``linewidths`` as suspect.

    md_temperature_mean : float
        Time-averaged kinetic temperature over the production segment, in K.
        Healthy: within ~3% of the requested ``temperature``. Drift larger
        than that means the thermostat coupling time is too long, the
        production segment is too short to equilibrate, or the chosen
        integrator is leaking energy. Rerun with adjusted
        ``thermostat_time_constant`` or longer ``equilibration_steps``.

    md_temperature_std : float
        Std-dev of the instantaneous kinetic temperature over the production
        segment, in K. For a Langevin NVT, the expected fluctuation scales
        as ``T * sqrt(2 / (3 * N))`` where N is atom count — e.g. for
        N=32 atoms at T=300 K, σ_T ≈ 43 K. Values dramatically larger or
        smaller than this rule of thumb indicate sampling or coupling
        problems.

    Call ``out.check_md_health()`` to get a structured pass/fail summary.
    """

    structure: Atoms
    fc2_supercell_matrix: np.ndarray              # (3, 3) int
    temperature: float                            # K
    q_points: np.ndarray                          # (n_q, 3) reduced — actually used
    harmonic_frequencies: np.ndarray              # (n_q, n_band) THz — pre-renormalisation
    renormalised_frequencies: np.ndarray          # (n_q, n_band) THz — fitted
    linewidths: np.ndarray                        # (n_q, n_band) THz FWHM
    converged: bool

    n_md_steps: int
    time_step_fs: float
    md_temperature_mean: float
    md_temperature_std: float

    power_spectra: np.ndarray | None = None       # (n_q, n_band, n_freq_bins)
    frequency_grid: np.ndarray | None = None      # (n_freq_bins,) THz

    quasiparticle: Any | None = None              # dynaphopy.Quasiparticle
    dynamics: Any | None = None                   # dynaphopy.Dynamics
    phonopy: Any | None = None                    # phonopy.Phonopy view used

    def to_dict(self) -> dict[str, Any]: ...

    def check_md_health(self, drift_tolerance: float = 0.03) -> tuple[bool, list[str]]:
        """Sanity-check the MD segment.

        Returns (is_healthy, warnings). ``is_healthy`` is True iff no warnings
        fired. Checks: (a) ``md_temperature_mean`` within ``drift_tolerance``
        of ``temperature``; (b) ``md_temperature_std`` within [0.5×, 2×] of
        the Langevin expectation ``T·sqrt(2/(3·N))``.
        """
```

The synthesis node calls `out.check_md_health()` just before returning and emits a `warnings.warn(...)` if the result is unhealthy, so non-experts see the diagnostic on first run.

### Macro signature (in `md_renormalised.py`)

```python
@pwf.api.as_macro_node("md_phonon_output")
def calculate_phonon_md_renormalisation(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix: ArrayLike | None = None,
    temperature: float = 300.0,
    # MD plumbing
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,                       # fs
    thermostat_time_constant: float = 100.0,      # fs
    seed: int | None = None,
    # q-point selection
    q_points: ArrayLike | None = None,            # None → auto-derived band path
    band_npoints: int = 30,
    # FC2 source — optional re-use from phono3py
    phono3py_output: PhononOutput | None = None,
    # output tiers
    power_spectra: bool = False,
    keep_handles: bool = False,
) -> MdPhononOutput: ...
```

### Argument coupling table (validated at macro entry by `_resolve_md_defaults`)

| Case | `fc2_supercell_matrix` | `phono3py_output` | Behaviour |
|---|---|---|---|
| Recompute FC2 | required | None | New `_compute_fc2_from_scratch` step runs ahead of MD |
| Reuse FC2 | optional | required (with `keep_handles=True`) | FC2 lifted from `phono3py_output.fc2`; supercell from `phono3py_output.fc2_supercell_matrix` |
| Both supplied | provided | provided | Both supercell matrices must match; raise `ValueError` if not |
| Neither | None | None | Raise `ValueError("Must supply fc2_supercell_matrix or phono3py_output")` |

### When to override the `q_points=None` default

The auto-derived band path is the right default for non-experts seeing renormalised dispersion. Pass explicit `q_points` when:

1. **Reproducing published values.** Direct comparison with a paper requires the exact q-points the reference used; ASE's auto-path will only coincide with the reference path for the most standard lattices and will be off by a permutation or extra segment in many real cases.
2. **Targeting specific physics.** Soft modes at zone-boundary q (e.g. M-point in perovskites near a structural transition), Kohn anomalies near 2k_F in metals, or q's coupling to a specific electronic feature. The auto-path won't preferentially sample these.
3. **Fast smoke / development runs.** `q_points=[[0, 0, 0]]` cuts a ~1-min default to ~2 s; useful when iterating on MD parameters.
4. **Non-standard or low-symmetry cells.** ASE's `bandpath(path=None, cell=...)` classifies the lattice via spglib and picks a canonical path. For unconventional primitive cells, lower-symmetry derivatives (alloys with broken symmetry, defected supercells), or cross-validation against an existing `phono3py_output.q_points`, you typically want to supply the q-point list explicitly.

### Macro graph

```
                            structure ─┐
                                       │
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
              _resolve_md_defaults         (phono3py_output if any)
                       │
                       │ (resolved_fc2_supercell, resolved_q_points,
                       │  resolved_seed, fc2_source_tag,
                       │  fc2_array if reused else None)
                       ▼
       ┌───────────────────────────┐    (skipped if phono3py_output gave FC2)
       │  _compute_fc2_from_scratch│
       │  (FC2 displacements +     │
       │   evaluate + phonopy fit)  │
       └───────────────────────────┘
                       │ fc2_array (n_sc, n_sc, 3, 3)
                       ▼
              _run_nvt_trajectory   ──── engine.with_working_directory("md_run/")
                       │ trajectory_pack: dict
                       │   {positions, velocities, time, supercell,
                       │    md_temperature_mean, md_temperature_std,
                       │    n_md_steps, time_step_fs}
                       ▼
              _project_with_dynaphopy
                       │ (builds Dynamics, Quasiparticle, fits each q in
                       │  resolved_q_points, packages MdPhononOutput,
                       │  auto-warns via check_md_health)
                       ▼
                  MdPhononOutput
```

**No Dynaphopy or Phonopy object crosses node boundaries.** Same principle as the phono3py macro: trajectory data are passed as plain ndarrays + a small dict of scalars; `Dynamics` and `Quasiparticle` are constructed inside `_project_with_dynaphopy` and only escape via `keep_handles=True`.

### Node responsibilities

- `_resolve_md_defaults(structure, fc2_supercell_matrix, phono3py_output, q_points, band_npoints, seed) → (resolved_fc2_supercell, resolved_q_points, resolved_seed, fc2_source_tag, fc2_array)` — runs at execution time (same proxy-arg constraint as `_resolve_defaults` in `anharmonic.py`). Validates the argument coupling table, auto-derives the band path when `q_points is None`, materialises `fc2_array` from `phono3py_output.fc2` if reusing, fills `seed` via `np.random.SeedSequence().entropy` when `None`.
- `_compute_fc2_from_scratch(structure, engine, resolved_fc2_supercell) → fc2_array` — runs only when `fc2_source_tag == "recompute"`. Reuses the existing `_generate_fc2_supercells` + `_evaluate_supercells` + a phonopy `produce_force_constants()` call. Returns the FC2 array.
- `_run_nvt_trajectory(structure, engine, resolved_fc2_supercell, temperature, equilibration_steps, production_steps, time_step, thermostat_time_constant, seed) → trajectory_pack: dict` — builds a `CalcInputMD(mode="NVT", thermostat="langevin", ...)`, runs equilibration discarded + production retained, captures positions/velocities/time/supercell ndarrays and the running mean/std of kinetic temperature. Returns a small dict.
- `_project_with_dynaphopy(structure, fc2_array, resolved_fc2_supercell, trajectory_pack, resolved_q_points, temperature, power_spectra, keep_handles) → MdPhononOutput` — synthesis. Calls `require_dynaphopy()`. Builds a `phonopy.Phonopy` with the FC2 attached. Builds `dynaphopy.dynamics.Dynamics` from the trajectory arrays. Builds `Quasiparticle(dynamics)`, sets temperature + force constants + q-points, drives the per-q Lorentzian fits, collects renormalised frequencies + linewidths into ndarrays, NaN-fills failed fits, packs into `MdPhononOutput`, calls `out.check_md_health()` and `warnings.warn(...)` if unhealthy. Returns the output.

## Error handling

Five named failure modes, each actionable:

1. **Missing `[phonons-md]` extra** — `_require_dynaphopy()` raises `ImportError("dynaphopy is required for this workflow. Install with: pip install pyiron_workflow_atomistics[phonons-md]")`. The new extras group is `phonons-md = ["phonopy", "phono3py", "symfc", "dynaphopy"]` — a superset of `[phonons]`. Users running only κ workflows keep the smaller install.
2. **Argument coupling violations** — caught in `_resolve_md_defaults` per the four-case table. Raised before any expensive step: `ValueError("Must supply fc2_supercell_matrix or phono3py_output (got neither)")`, or `ValueError(f"fc2_supercell_matrix={user_value} disagrees with phono3py_output.fc2_supercell_matrix={output_value}")`, or `ValueError("phono3py_output.fc2 is None; re-run calculate_phonon_thermal_conductivity with keep_handles=True to enable FC2 reuse")`.
3. **MD non-convergence as a soft signal** — `check_md_health()` auto-warn from § 2. Does NOT raise; the run completed and the numbers are returned, but the user sees `UserWarning("MD diagnostics indicate potential issues: ⟨T⟩ drift 7.2% exceeds tolerance 3%: ...")` immediately. We don't raise because there's no bright-line failure; the user may have a loose tolerance deliberately or want to inspect partial results.
4. **Per-q-point Lorentzian fit non-convergence** — dynaphopy reports per-fit success status. If any fit fails:
   - `converged=False` on the output
   - `renormalised_frequencies[q, band]` and `linewidths[q, band]` set to `NaN` (plotting code naturally drops those points)
   - `warnings.warn(f"Lorentzian fit failed for {n_failed} of {n_total} (q, band) pairs; corresponding entries are NaN")`
   Power users with `power_spectra=True` get the raw spectra and can refit.
5. **Inconsistent FC2 / missing FC2 in reused output** — covered in mode 2.

What we deliberately don't do: no retries on failed Lorentzian fits, no automatic equilibration extension based on energy drift, no silent trajectory truncation to skip bad frames. Each of those would mask user-facing problems.

## Testing

Three tiers.

**Tier 1 — Cheap unit tests (no dynaphopy / phono3py / MD needed; always run):**

- `test_md_phonon_output_dataclass_shape` — required fields are required (no default), optional fields default `None`, `to_dict()` round-trips, `check_md_health()` returns `(bool, list[str])`.
- `test_check_md_health_passes_on_clean_run` — construct `MdPhononOutput` with `md_temperature_mean=300.0, md_temperature_std=43.0, temperature=300.0`; assert health check returns `(True, [])`.
- `test_check_md_health_flags_temperature_drift` — same construction but `md_temperature_mean=270.0` (10% drift); assert `(False, [<message naming "drift">])`.
- `test_check_md_health_flags_anomalous_sigma` — `md_temperature_std=200.0`; assert σ_T flag fires.
- `test_resolve_md_defaults_validates_argument_coupling` — three sub-cases: both `None` → `ValueError`; inconsistent supercells → `ValueError`; `phono3py_output.fc2 is None` → `ValueError` naming `keep_handles=True`.
- `test_require_dynaphopy_missing_actionable` — patch `sys.modules["dynaphopy"] = None`; assert `_compat.require_dynaphopy()` raises `ImportError` naming `pip install pyiron_workflow_atomistics[phonons-md]`.

**Tier 2 — Integration tests (gated on `pytest.importorskip("dynaphopy")` per test):**

- `test_calculate_phonon_md_renormalisation_emt_smoke` — Cu FCC primitive (1 atom, 3 bands), `fc2_supercell_matrix=2*np.eye(3)`, `temperature=300`, `q_points=[[0,0,0]]` (Γ only for runtime), `equilibration_steps=500`, `production_steps=2000`, `time_step=1.0`, ASEEngine + EMT. Asserts `converged is True`, `renormalised_frequencies.shape == (1, 3)` (one q-point × 3 bands for the 1-atom primitive), all values finite and positive, `md_temperature_mean` within 10% of 300 K.
- `test_fc2_reuse_path_skips_fc2_recompute` — first run `calculate_phonon_thermal_conductivity` with `keep_handles=True`, then call the dynaphopy macro with `phono3py_output=<that output>`. Assert (a) no `fc2_disp_*` directories in the dynaphopy run's working_directory, (b) macro completes, (c) `out.fc2_supercell_matrix` matches the upstream value.
- `test_fc2_recompute_path_creates_fc2_subdirs` — no `phono3py_output`, explicit `fc2_supercell_matrix`. Assert FC2 displacement subdirs exist on disk, macro completes.
- `test_power_spectra_off_by_default` — `out.power_spectra is None`, `out.frequency_grid is None`.
- `test_power_spectra_on_populates_arrays` — `power_spectra=True`; assert shapes match `(n_q, n_band, n_freq_bins)`.
- `test_keep_handles_returns_quasiparticle_and_dynamics` — `keep_handles=True`; assert handles populated and respond to expected attribute access.
- `test_auto_warn_fires_on_bad_md` — monkey-patch the MD step to inject a degenerate trajectory (positions barely moving → ⟨T⟩≈0 or chaotic → σ_T huge); assert `warnings.warn` fires during macro execution.

**Tier 3 — Determinism:**

- `test_md_seed_determinism` — call the macro twice with the same `seed=42`; assert byte-identical `renormalised_frequencies`. Validates the seed plumbing.
- `test_band_path_auto_derivation_deterministic` — call `_resolve_md_defaults` twice with `q_points=None` and identical cell; assert identical `q_points` array.

**CI wiring**

In `.ci_support/environment.yml`:
- Add `- dynaphopy` to the conda dependency list.

In `pyproject.toml`:
- Add `phonons-md = ["phonopy", "phono3py", "symfc", "dynaphopy"]` to `[project.optional-dependencies]`. The existing `phonons` group is unchanged.

**What we don't test:**

- Numerical accuracy against published renormalised dispersion — belongs in a validation notebook, not pytest.
- Multiple-temperature runs in a single macro call — out of v1 scope.
- Comparison with phono3py's mode-resolved γ — different physics (RTA perturbation vs full MD); no analytic equivalence to test against.

## Public API change

`pyiron_workflow_atomistics/physics/phonons/__init__.py` exports two new symbols:

```python
from .output import MdPhononOutput, PhononOutput
from .anharmonic import calculate_phonon_thermal_conductivity
from .harmonic import compute_phonopy_harmonic
from .md_renormalised import calculate_phonon_md_renormalisation

__all__ = [
    "PhononOutput",
    "MdPhononOutput",
    "calculate_phonon_thermal_conductivity",
    "calculate_phonon_md_renormalisation",
]
```

Consumers import per-topic, matching the project convention:

```python
from pyiron_workflow_atomistics.physics.phonons import (
    calculate_phonon_md_renormalisation,
    MdPhononOutput,
)
```

No existing symbols move or break — purely additive.

## Versioning + release

Minor bump rather than patch: `0.0.7 → 0.0.8`, via versioneer tag `pyiron_workflow_atomistics-0.0.8`. CHANGELOG gets a new `[0.0.8]` entry describing the additions + the `[phonons-md]` extra. Release sequence identical to v0.0.7 (PR → tag → GitHub Release → shared `pyproject-release.yml` auto-publishes).

## Follow-ups (explicitly v2 or later)

- **NVE / NPT ensembles**, controlled via the existing `CalcInputMD`. Useful when Langevin stochasticity contaminates lifetimes (long-tailed Lorentzians) or for explicit volume-fluctuation studies.
- **Multi-T MD per call** — currently a single `temperature` per macro call. v2 could accept `temperatures=[300, 500, 700]` and either run one long trajectory at each T or one trajectory at each T concurrently via `engine.with_working_directory`.
- **NAC support** — same hooks as the phono3py macro (`born_charges`, `epsilon_inf`); raises `NotImplementedError` until v2.
- **dispersion-path convenience helper** — `band_path_to_q_points(path_string, cell)` so users who want a non-default path don't have to construct it themselves.
- **Phonopy-side caching** — currently if FC2 is recomputed in the macro, the resulting FC2 array is dropped. v2 could expose `cached_fc2` on `MdPhononOutput` so downstream calls reuse it without re-running phonopy.
