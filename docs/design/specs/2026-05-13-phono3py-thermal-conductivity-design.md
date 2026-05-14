# Phonon thermal conductivity via phono3py

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-13 |
| Repo | `pyiron/pyiron_workflow_atomistics` |
| Scope | phono3py (FC2 + FC3 + κ); harmonic phonopy outputs are exposed as side-products since FC2 is computed anyway |
| Out of scope (v1) | dynaphopy (post-MD anharmonic renormalisation), non-analytic correction (BORN + ε∞) for polar materials, Z*/ε∞ DFPT workflows |

## Problem

`pyiron_workflow_atomistics` has clean physics workflows for total-energy-derived quantities (`bulk.eos_volume_scan`, `point_defect.get_vacancy_formation_energy`, `surface`, `grain_boundary`) but no phonon properties. The most-requested next physics output is lattice thermal conductivity κ(T), which requires anharmonic (3rd-order) force constants and is the headline output of `phono3py`.

The natural integration point is unambiguous: every existing physics macro fans out by looping `engine.calculate` over a list of perturbed structures. phono3py's workflow is exactly that — generate displaced supercells, compute forces on each, fit force constants, solve the Boltzmann transport equation. No new engine code is needed; the same `Engine` Protocol that powers EOS already powers phonons.

What does need design is (a) how the user *expresses* the workflow without learning a parallel mini-vocabulary for displacement samplers, (b) the dataclass shape the macro returns so callers don't have to reach into a live `Phono3py` object, and (c) the dependency boundary so users who don't need phonons aren't forced to install phono3py.

## Approach

A new `pyiron_workflow_atomistics.physics.phonons` subpackage with one user-facing macro, `calculate_phonon_thermal_conductivity`, that returns a structured `PhononOutput` dataclass. Internally it builds a macro graph: two parallel displacement-generation + force-evaluation fan-outs (FC2 and FC3), each reusing the existing `calculate` node verbatim, joined by a synthesis node that produces FC2/FC3 force constants and runs phono3py's BTE solver.

`phonopy`, `phono3py`, and `symfc` ship as an **optional extra** (`pip install pyiron_workflow_atomistics[phonons]`), behind lazy imports. Tier-1 unit tests (3 cases) run without the extra; tier-2 integration tests are gated on `pytest.importorskip("phono3py")` and are part of the CI matrix when the extra installs cleanly.

Four deliberate API decisions, each motivated below:

- **Expose phono3py's own kwargs, not a parallel sampler abstraction.** The user already chose FD+random as both being v1-supported behind a single macro signature. Rather than wrap that in a `DisplacementSampler` Protocol (which a phonopy user has never heard of), the macro takes phono3py's documented parameters (`displacement_distance`, `is_plusminus`, `cutoff_pair_distance`, `number_of_snapshots`, `random_seed`, `fc_calculator`) verbatim. The two modes are dispatched by a one-line rule (`number_of_snapshots is not None` ⇒ random; defaults `fc_calculator="symfc"` if user didn't override). YAGNI on the Protocol — if/when a third sampler (ALAMODE, hiPhive) lands, the same kwargs can stay as a sugar overload above a then-introduced Protocol.
- **Pass forces, not Phono3py objects, across node boundaries.** `phono3py.Phono3py` holds a `Symmetry` C-extension handle and an interaction tensor that don't round-trip cleanly through pickle. Every node that needs a Phono3py instance reconstructs one from the same construction kwargs the previous node used — deterministic for FD, seed-deterministic for random. The only cross-edge payload is `list[Atoms]` (generated supercells) and `list[np.ndarray]` (computed forces, extracted inside the synthesis node from the `EngineOutput` list).
- **Random-mode seed is resolved once at macro entry.** Random-displacement determinism between the generation node and the synthesis node depends on both nodes drawing the *same* random sample. If `number_of_snapshots is not None and random_seed is None`, the macro draws a seed with `np.random.SeedSequence().entropy` once at macro entry and threads that resolved seed through both `_generate_fc3_supercells` and `_run_phono3py_thermal_conductivity`. The resolved seed is also stored on `PhononOutput.phono3py` (when `keep_handles=True`) for reproducibility. Without this, the supercell-count guard in § 4 would miss the failure: counts match, but positions differ — silent corruption.
- **`PhononOutput` has three opt-in tiers of optional fields.** The required fields (`structure`, `fc2_supercell_matrix`, `fc3_supercell_matrix`, `temperatures`, `kappa`, `converged`) are always populated. Optional groups gated by macro flags: `mode_resolved=True` populates `q_points`, `frequencies`, `group_velocities`, `mode_kappa`, `gamma`, `gruneisen`; `harmonic_observables=True` populates `band_structure`, `dos`, `free_energy`; `keep_handles=True` populates `fc2`, `fc3`, `phono3py`. All default `False` to keep the cheapest path actually cheap.

## Components

```
pyiron_workflow_atomistics/physics/phonons/
├── __init__.py          # public re-exports
├── output.py            # PhononOutput dataclass
├── harmonic.py          # phonopy FC2 helpers + band/DOS/F(T) nodes
└── anharmonic.py        # phono3py FC3 + κ(T); the user-facing macro
```

`samplers.py` is **not** created in v1 — the sampler-as-Protocol abstraction is deferred per the approach above. Lazy-import shims (`_require_phonopy`, `_require_phono3py`, `_require_symfc`) live one each in `harmonic.py` and `anharmonic.py`.

### `PhononOutput` (in `output.py`)

```python
@dataclass
class PhononOutput:
    # required
    structure: Atoms                              # primitive cell used for the run
    fc2_supercell_matrix: np.ndarray              # (3, 3)
    fc3_supercell_matrix: np.ndarray              # (3, 3)
    temperatures: np.ndarray                      # (n_T,) K
    kappa: np.ndarray                             # (n_T, 3, 3) tensor, W/m·K
    converged: bool                               # κ-solver convergence flag

    # mode-resolved (populated iff mode_resolved=True)
    q_points: np.ndarray | None = None            # (n_q, 3) reduced coords
    frequencies: np.ndarray | None = None         # (n_q, n_band) THz
    group_velocities: np.ndarray | None = None    # (n_q, n_band, 3) Å·THz
    mode_kappa: np.ndarray | None = None          # (n_T, n_q, n_band, 6) Voigt
    gamma: np.ndarray | None = None               # (n_T, n_q, n_band) linewidths
    gruneisen: np.ndarray | None = None           # (n_q, n_band) mode γ

    # harmonic side-products (populated iff harmonic_observables=True)
    band_structure: dict | None = None            # {"path": list[str], "q": ndarray,
                                                  #  "frequencies": ndarray}
    dos: dict | None = None                       # {"frequencies": ndarray, "dos": ndarray}
    free_energy: dict | None = None               # {"temperatures": ndarray,
                                                  #  "F": ndarray, "S": ndarray, "Cv": ndarray}

    # escape hatch (populated iff keep_handles=True)
    fc2: np.ndarray | None = None                 # (n_sc, n_sc, 3, 3)
    fc3: np.ndarray | None = None                 # (n_sc, n_sc, n_sc, 3, 3, 3)
    phono3py: Any | None = None                   # the live Phono3py instance

    def to_dict(self) -> dict[str, Any]: ...
```

### Macro graph

`calculate_phonon_thermal_conductivity` (in `anharmonic.py`) is a `@pwf.api.as_macro_node(...)` whose body wires together:

```
                       structure ─┐
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
   _generate_fc2_supercells              _generate_fc3_supercells
              │                                       │
              ▼                                       ▼
       fc2_atoms_list                          fc3_atoms_list
              │                                       │
              ▼                                       ▼
   _evaluate_supercells                  _evaluate_supercells
   (engine, "fc2_disp_")                 (engine, "fc3_disp_")
              │                                       │
              ▼                                       ▼
      fc2_engine_outputs                       fc3_engine_outputs
              │                                       │
              └──────────────┬────────────────────────┘
                             ▼
              _run_phono3py_thermal_conductivity
                             │
                             ▼
                       PhononOutput
```

### Node responsibilities

- `_generate_fc2_supercells(structure, fc2_supercell_matrix, displacement_distance, is_plusminus) -> list[Atoms]` — constructs a `Phono3py` object, calls `generate_displacements(distance=..., is_plusminus=...)`, returns `phono3py_obj.supercells_with_displacements` converted to ASE `Atoms`. The Phono3py instance is **not** returned; it is reconstructed identically downstream from the same kwargs.
- `_generate_fc3_supercells(structure, fc2_supercell_matrix, fc3_supercell_matrix, displacement_distance, is_plusminus, cutoff_pair_distance, number_of_snapshots, random_seed) -> list[Atoms]` — same idea for FC3. phono3py uses one `Phono3py` instance for both, but splitting generation into two nodes lets the two force fan-outs run in parallel.
- `_evaluate_supercells(supercells, engine, prefix) -> list[EngineOutput]` — copy of the existing `bulk.evaluate_structures` pattern: loops `calculate.node_function(supercell, engine.with_working_directory(f"{prefix}{i:04d}"))`. The only node in the phonon graph that touches the engine.
- `_run_phono3py_thermal_conductivity(structure, fc2_supercell_matrix, fc3_supercell_matrix, displacement_distance, is_plusminus, cutoff_pair_distance, number_of_snapshots, random_seed, fc_calculator, fc2_engine_outputs, fc3_engine_outputs, temperatures, q_mesh, mode_resolved, harmonic_observables, keep_handles) -> PhononOutput` — synthesis. Extracts `out.final_forces` from each `EngineOutput` (after asserting `out.converged is True` for all of them — see § 4). Rebuilds a `Phono3py` from the kwargs, attaches both force sets, calls `produce_fc2()` + `produce_fc3(fc_calculator=...)`, then `run_thermal_conductivity(temperatures=..., mesh=q_mesh)`. Optionally runs `run_band_structure()` / `run_total_dos()` / `run_thermal_properties()` on a phonopy view if `harmonic_observables=True`. Bundles into `PhononOutput`. No separate `_extract_forces` node — there is only one downstream consumer, unlike the `bulk._extract_energies` pattern which fans out to both EOS and per-atom normalisation.

### Macro signature

```python
@pwf.api.as_macro_node("phonon_output")
def calculate_phonon_thermal_conductivity(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike | None = None,   # defaults to fc2_supercell_matrix
    temperatures: ArrayLike = (300.0,),
    q_mesh: ArrayLike = (11, 11, 11),
    # phono3py.generate_displacements kwargs, passed through verbatim
    displacement_distance: float = 0.03,
    is_plusminus: bool | Literal["auto"] = "auto",
    cutoff_pair_distance: float | None = None,
    number_of_snapshots: int | None = None,          # presence switches to random mode
    random_seed: int | None = None,
    fc_calculator: str | None = None,                # auto-set to "symfc" if random
    # output tiers
    mode_resolved: bool = False,
    harmonic_observables: bool = False,
    keep_handles: bool = False,
    # polar-material kwargs, v1 raises NotImplementedError if non-None
    born_charges: np.ndarray | None = None,
    epsilon_inf: np.ndarray | None = None,
    # plumbing
    parent_working_directory: str = "phono3py",
) -> PhononOutput: ...
```

`fc2_supercell_matrix` and `fc3_supercell_matrix` accept `int`, `list[int]` of length 3, or `(3, 3)` ndarray — normalised internally to a `(3, 3)` int matrix in the same style as `phonopy`'s constructor.

## Error handling

Five well-named failure modes, each with an actionable message:

1. **Missing extras** — `_require_phono3py()` / `_require_phonopy()` / `_require_symfc()` raise `ImportError` whose message contains the exact `pip install pyiron_workflow_atomistics[phonons]` command. `_require_symfc()` fires only when `fc_calculator == "symfc"`; default-FD users don't pay for it.
2. **Non-converged supercell force calc** — before the synthesis node fits force constants, it scans `fc2_engine_outputs` and `fc3_engine_outputs` and raises `RuntimeError(f"Force calc failed for {prefix} supercells: {failed_indices}")` listing both the supercell indices and their `working_directory` paths. Silent drop would corrupt the fit asymmetrically.
3. **Polar material kwargs in v1** — passing `born_charges` or `epsilon_inf` raises `NotImplementedError("Non-analytic correction (BORN + ε∞) is not supported in v1; see the 'NAC / BORN effective charges' follow-up at the end of this spec.")` without ever importing phono3py. No auto-detection of "is this polar?" — that's brittle.
4. **κ-solver non-convergence** — `phono3py.run_thermal_conductivity` doesn't raise on non-convergence; it warns. The synthesis node uses `warnings.catch_warnings(record=True)`, scans for phono3py's documented non-convergence messages, and sets `PhononOutput.converged = False`. κ values still returned (often useful as a sanity check) but the flag tells the caller they're untrusted.
5. **Supercell/force count mismatch** — the rebuild-from-kwargs trick has exactly one silent-corruption failure mode: if displacement kwargs drift between generation and synthesis, the rebuilt Phono3py has a different number of supercells than the forces list. The synthesis node asserts this loudly: `RuntimeError(f"FC2 force/supercell mismatch: {n_forces} forces vs {n_supercells} expected. ...")`. Same check for FC3.

What we don't do: no retries (force calcs are expensive and the user owns the engine config); no partial-result salvaging (a missing FC3 supercell can't be zero-filled — biases the fit); no swallowing of phono3py warnings other than the κ-convergence one.

## Testing

Three tiers.

**Tier 1 — Cheap unit tests (no phono3py needed; always run):**

- `test_missing_phono3py_raises_actionable` — patch `sys.modules["phono3py"] = None`, assert `_require_phono3py()` raises with the exact `pip install` line.
- `test_born_charges_kwarg_raises_not_implemented` — calling the macro with `born_charges=...` raises `NotImplementedError` without importing phono3py.
- `test_phonon_output_dataclass_shape` — `PhononOutput` is a dataclass, required fields non-Optional, all optionals default `None`, `to_dict()` round-trips.

**Tier 2 — Integration tests (gated on `pytest.importorskip("phono3py")`):**

- `test_calculate_thermal_conductivity_emt_smoke` — Cu FCC primitive, `fc2_supercell_matrix = fc3_supercell_matrix = 2*np.eye(3)`, `temperatures=[300]`, `q_mesh=(5,5,5)`, ASEEngine + EMT. Asserts `converged is True`, `kappa.shape == (1, 3, 3)`, diagonal terms positive and within 2× of a hard-coded EMT-Cu reference value (sanity floor, not benchmark). Budget: ~60s.
- `test_random_displacement_mode_runs` — same setup with `number_of_snapshots=20, random_seed=0, fc_calculator="symfc"`. Asserts `converged is True` and `kappa` is finite. Additionally gated on `pytest.importorskip("symfc")`.
- `test_optional_tiers_off_by_default` — runs the smoke test with all toggles off, asserts the corresponding `PhononOutput` fields are `None`. Re-runs with each toggle on, asserts populated and shaped as documented.
- `test_engine_with_working_directory_is_used` — `tmp_path`-rooted engine; asserts `tmp_path/fc2_disp_0000/` and `tmp_path/fc3_disp_0000/` exist on disk.
- `test_supercell_force_mismatch_raises` — monkeypatch `_evaluate_supercells` to return one fewer force array than supercells; assert the explicit `RuntimeError` from § 4.
- `test_force_calc_nonconvergence_raises` — engine fixture returns `converged=False` for supercell index 3; assert `RuntimeError` lists "3" in failed indices.

**Tier 3 — Determinism (Tier 1 if mockable, else Tier 2):**

- `test_fd_generation_is_deterministic` — `_generate_fc2_supercells` twice with same kwargs returns identical Atoms.
- `test_random_generation_is_seed_deterministic` — same with `number_of_snapshots=10, random_seed=42`.

**What we don't test:** numerical accuracy against a DFT reference (validation notebook territory, not pytest), and per-property numerical agreement with the phono3py CLI (would couple to a phono3py release).

## CI wiring

`pyproject.toml`:

```toml
[project.optional-dependencies]
phonons = ["phonopy", "phono3py", "symfc"]
test = ["pytest", "nbformat", "nbclient"]
```

CI install line becomes `pip install -e ".[test,phonons]"`. Wheels exist for phono3py on cpython 3.9–3.12 / linux+macOS; if a future Python lacks a wheel we add a matrix exclude rather than a runtime skip, so "tests pass" continues to mean "all tiers ran".

## Public API change

`pyiron_workflow_atomistics/physics/phonons/__init__.py` exports:

```python
from .output import PhononOutput
from .anharmonic import calculate_phonon_thermal_conductivity
from .harmonic import compute_phonopy_harmonic  # advanced users only
```

Consumers import per-topic, matching the project convention:

```python
from pyiron_workflow_atomistics.physics.phonons import (
    calculate_phonon_thermal_conductivity,
    PhononOutput,
)
```

No existing symbols move or break — purely additive.

## Versioning + release

Patch bump via versioneer tag (next tag: `pyiron_workflow_atomistics-0.0.7`). `CHANGELOG.md` gets a single section describing the addition + the `[phonons]` extra. Release sequence identical to the engine-conformance suite spec (PR → tag → shared `pyproject-release.yml` auto-publishes).

## Follow-ups (explicitly v2 or later)

- **NAC / BORN effective charges** for polar materials. Accept user-supplied `born_charges` (N, 3, 3) and `epsilon_inf` (3, 3); wire into `Phonopy(nac_params=...)`. No DFPT — user computes externally.
- **Z*/ε∞ DFPT workflow.** Requires engine-side hooks that don't exist; only meaningful for VASP/QE backends. Separate spec.
- **dynaphopy** post-MD anharmonic renormalisation. Lives at `physics/phonons/md_renormalised.py`. Reuses the harmonic FC2 step here, runs MD via `CalcInputMD` + the existing engine, projects velocity ACF onto modes. Separate spec.
- **`DisplacementSampler` Protocol** for ALAMODE / hiPhive integration. Add only when a concrete third sampler is requested; keep current kwargs as a sugar overload.
- **Two-supercell asymmetry optimization** (FC2 supercell larger than FC3). Already supported by the API surface — the macro accepts independent matrices. Just hasn't been profiled yet.
