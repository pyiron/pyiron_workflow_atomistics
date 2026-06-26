# Melting point via the interface (coexistence) method — design

**Date:** 2026-06-27
**Branch:** `feat/melting-point-interface-method`
**Status:** Design (awaiting review)
**Author:** Han (with Claude)

---

## 1. Goal

Add an engine-agnostic **melting-point** workflow to `pyiron_workflow_atomistics`
implementing the full **interface / solid–liquid coexistence method**, as a new
`physics/melting/` subpackage. It must run on **any conformant `Engine`** —
verified on the ASE engine (EMT/Al, then a GRACE foundation model) **and** on a
new **`LammpsEngine`** (full coexistence run with `pair_style grace`).

The source material is `/ptmp/hmai/meltingpoint.ipynb`, a partial port (Step 1
only) of `pyiron_atomistics.thermodynamics.interfacemethod`. This work ports the
**complete** algorithm (Steps 1 + 2) onto the modern engine abstraction and
converts the notebook into a runnable script.

### Deliverables
1. `physics/melting/` subpackage (full method: nodes + Python drivers).
2. A conformant `engine/lammps.py` (`LammpsEngine`) running in-process LAMMPS.
3. `scripts/meltingpoint.py` — the notebook as a runnable script (ASE/EMT demo).
4. Unit tests (fast, no MD) + integration smoke tests (EMT/Al, slow marker).
5. A verification run: EMT/Al (Step 1 + ≥1 coexistence iteration), then GRACE
   (Step 1), then a LAMMPS coexistence run with `pair_style grace`.

---

## 2. The methodology (what we are implementing)

The interface method brackets the melting temperature `T_m` by holding a
half-solid / half-liquid cell in coexistence and extrapolating the measured
temperature to zero pressure.

**Step 1 — initial guess (the notebook).** Build a bulk crystal (~`N/2` atoms),
relax it, then **bisect temperature**: at each `T` run NPT MD on the *bulk solid*
and classify *solid vs molten* from the dominant CNA fraction (stays crystalline
⇒ raise `T`; melts ⇒ lower `T`). Converges to a rough `T0` (≈914 K for the
notebook's EAM Al).

**Step 2 — coexistence refinement (`calc_temp_iteration`, looped).** At the
current `T`:
1. **NPT-relax the solid** at `T`.
2. **Build a solid–liquid interface**: freeze the lower half (`z < 0.5`), melt the
   upper half at `T + ΔT_melt` (NPT, z-only barostat), then recool the upper half
   to `T` against the frozen solid → a coexistence cell.
3. **Strain scan** along `z`: for each strain in a grid around the equilibrium
   c-length, **NVT-equilibrate** then **NVE**-run; record per-frame temperature
   and pressure.
4. **Solid fraction per strain** via CNA + a 1-D **kernel-density estimate** of
   the crystalline atoms' `z`-positions (fraction of the box that is solid).
5. **`ratio_selection`**: keep the contiguous strain window where the solid
   fraction sits near 0.5 (true coexistence).
6. **Void rejection**: drop strains whose **Voronoi** max-volume indicates holes.
7. **Predict `T_m`**: fit `P(strain)` and `T(strain)` (and `T(P)`); the new
   estimate is `T` extrapolated to `P = 0`.
8. **Convergence loop**: repeat with progressively finer schedules
   (`timestep_lst`, `fit_range_lst`, `nve_steps_lst`) until
   `|ΔT_next| ≤ convergence_goal`.

Reference functions (verbatim algorithm) live in
`pyiron_atomistics.thermodynamics.interfacemethod`; this design re-expresses them
on `EngineOutput` instead of pyiron-atomistics LAMMPS jobs.

---

## 3. Architecture

### 3.1 Principles
- **Engine-agnostic**: the algorithm only ever calls
  `calculate(structure, engine)` (the `engine/protocol.py` node) with
  `CalcInputMD` / `CalcInputMinimize`. It never imports ASE or LAMMPS. ⇒ "supports
  both" by construction.
- **Adaptive loops in Python drivers (Approach A)**: pure operations are
  `@pwf.as_function_node`s; the bisection and convergence *loops* live inside
  `@pwf.as_function_node` driver functions that iterate, calling
  `calculate.node_function(...)` per step — exactly the
  `physics/free_energy/quasiharmonic.py::_static_energies_per_volume` idiom. A
  thin top-level `@pwf.as_macro_node` wires the stages.
- **Per-step isolation** via `engine.with_working_directory(f"...{i:03d}")`
  (or the `subengine` node inside macros).
- **Per-step calc parameters** by producing an engine variant carrying a fresh
  `CalcInputMD`/`CalcInputMinimize` (helper `with_calc_input`, below).

### 3.2 Why the trajectory suffices (verified 2026-06-27)
`EngineOutput` already carries `structures` (per-frame snapshots), `energies`,
and `stresses` (per-frame Voigt). Empirically (ASE engine, EMT/Al NVT):
snapshots **preserve momenta**, so `snapshot.get_temperature()` recovers per-frame
temperature, and per-frame Voigt stress gives the virial pressure. The analysis
computes the **total pressure = virial + kinetic ideal-gas term**
(`P = -tr(σ)/3 + ρ·k_B·T`); omitting the kinetic term would bias the `T(P→0)`
extrapolation. **No change to `EngineOutput` or the conformance contract is
required.** (The `LammpsEngine` must likewise populate `structures` with
velocities and per-frame `stresses`.)

### 3.3 NVE velocity initialisation
NVE has no thermostat, so kinetic energy equipartitions ⇒ the equilibrated
temperature is ~½ the initial. To land at the target `T`, NVE/strain runs set
`CalcInputMD.initial_temperature = 2·T` (the notebook's `half_velocity` trick).
Verified: a 900 K MaxwellBoltzmann init equilibrates near ~450 K without it.

### 3.4 Module layout
```
pyiron_workflow_atomistics/physics/melting/
  __init__.py          # public API exports
  inputs.py            # MeltingInput dataclass
  outputs.py           # MeltingResult dataclass (+ MeltingIterationRecord)
  structures.py        # coexistence supercell, freeze_half, strain_cell_along_z, unfreeze
  analysis.py          # cna_fractions, classify_solid, solid_fraction_kde,
                       #   voronoi_max_mean / holes_mask,
                       #   temperatures_from_trajectory, pressures_from_trajectory
  md_steps.py          # with_calc_input, relax_structure, npt_relax_solid,
                       #   build_solid_liquid_interface, strain_scan_nvt_nve
  fitting.py           # ratio_selection, predict_melting_point
  initial_guess.py     # estimate_melting_temperature (Step 1 bisection driver)
  coexistence.py       # coexistence_iteration + refine_melting_point (Step 2 loop)
  study.py             # calculate_melting_point (top-level macro/driver)

pyiron_workflow_atomistics/engine/lammps.py   # LammpsEngine (conformant)

scripts/meltingpoint.py                        # notebook → script (EMT/Al Step-1 demo)

tests/unit/physics/melting/                    # fast unit tests
tests/unit/engine/test_lammps_conformance.py   # LammpsEngine vs conformance suite
tests/integration/test_melting_emt.py          # EMT/Al smoke (slow marker)
```

### 3.5 Node & driver inventory

**structures.py** (`@pwf.as_function_node`)
- `create_coexistence_supercell(element, crystalstructure, a=None, n_atoms=8000) -> Atoms`
  — `get_bulk(cubic=True)` then pick the `i×i×i` repeat whose atom count is
  closest to `n_atoms/2` (notebook logic).
- `freeze_half(structure, axis=2, fraction=0.5) -> Atoms` — `FixAtoms` on
  `scaled_pos[axis] < fraction`. **New** (not in `structure/defects.py`).
- `unfreeze(structure) -> Atoms` — clear constraints.
- `strain_cell_along_z(structure, strain) -> Atoms` — scale `cell[2,2]` by
  `strain`, `scale_atoms=True`.

**analysis.py** (`@pwf.as_function_node`)
- `cna_fractions(structure, ovito_compatibility=True) -> dict` — wraps
  `structuretoolkit.analyse.get_adaptive_cna_descriptors`.
- `classify_solid(structure, key_max, distribution_half) -> bool`.
- `analyse_reference_structure(structure) -> (key_max, n_atoms, distribution_half)`.
- `solid_fraction_kde(structure, crystalstructure, threshold=0.1) -> float`
  — CNA per-atom (`mode="str"`) + sklearn `KernelDensity` on crystalline `z`.
- `voronoi_max_mean(structure) -> (max, mean)` and
  `holes_mask(maxes, means, factor=2.0) -> list[bool]` — structuretoolkit Voronoi.
- `temperatures_from_trajectory(engine_output, last_n=20) -> list[float]`.
- `pressures_from_trajectory(engine_output, last_n=20) -> list[float]` — virial
  (Voigt) + kinetic term, returned in GPa.

**md_steps.py**
- `with_calc_input(engine, calc_input) -> Engine` (`@pwf.as_function_node`) —
  `dataclasses.replace(engine, EngineInput=calc_input)`.
- `npt_relax_solid(structure, engine, temperature, n_steps, timestep, seed) ->
  EngineOutput` driver: builds `CalcInputMD(mode="NPT", pressure=0, ...)`,
  `initial_temperature=2T`, runs `calculate.node_function`.
- `build_solid_liquid_interface(structure, engine, t_solid, t_liquid, ...) ->
  Atoms` driver: `freeze_half` → NPT-melt upper half at `t_liquid` → NPT-recool to
  `t_solid` → `unfreeze`.
- `strain_scan_nvt_nve(structure, engine, temperature, strains, nvt_steps,
  nve_steps, timestep, seed) -> list[dict]` driver: per strain, NVT then NVE;
  returns `{strain, mean_T, mean_P, solid_fraction, voronoi_max/mean}`.

**fitting.py** (`@pwf.as_function_node`)
- `ratio_selection(records, ratio_boundary=0.25) -> (selected_records, sl_flag)`.
- `predict_melting_point(selected_records, boundary_value=0.25) ->
  (t_next, t_mean, t_left, t_right)` — `np.polyfit` of P,T vs strain; `T(P=0)`.

**initial_guess.py**
- `estimate_melting_temperature(structure, engine, t_left=0, t_right=1000,
  strain_run_steps=1000, timestep=2.0, seed, crystalstructure) ->
  (t_guess, structure)` (`@pwf.as_function_node`) — Step-1 bisection driver
  (faithful port of notebook cells 38–48).

**coexistence.py**
- `coexistence_iteration(structure, engine, temperature, schedule, params) ->
  MeltingIterationRecord` driver — one `calc_temp_iteration`.
- `refine_melting_point(structure, engine, t_guess, input) -> MeltingResult`
  driver — the convergence loop over schedules.

**study.py**
- `calculate_melting_point(wf, engine, input: MeltingInput) -> MeltingResult`
  (`@pwf.as_macro_node`) — `create_coexistence_supercell` → relax/minimize →
  `analyse_reference_structure` → `estimate_melting_temperature` →
  `refine_melting_point`.

### 3.6 inputs / outputs (mirroring `free_energy` style: plain `@dataclass`, `to_dict()`)
```python
@dataclass
class MeltingInput:
    element: str
    crystalstructure: str | None = None     # default: ASE reference state
    a: float | None = None
    n_atoms: int = 8000
    temperature_left: float = 0.0
    temperature_right: float = 1000.0
    convergence_goal: float = 1.0            # K
    timestep_lst: list[float] = (2.0, 2.0, 1.0)        # fs
    fit_range_lst: list[float] = (0.05, 0.01, 0.01)
    nve_steps_lst: list[int] = (25000, 20000, 50000)
    nvt_run_steps: int = 10000
    npt_run_steps: int = 50000
    strain_run_steps: int = 1000
    n_strain_points: int = 21
    ratio_boundary: float = 0.25
    boundary_value: float = 0.25
    delta_t_melt: float = 1000.0             # superheat for interface build
    seed: int | None = None

@dataclass
class MeltingIterationRecord:
    temperature_in: float
    temperature_next: float
    strains: list[float]; ratios: list[float]
    pressures: list[float]; temperatures: list[float]
    converged: bool

@dataclass
class MeltingResult:
    melting_temperature: float
    converged: bool
    n_iterations: int
    element: str; crystalstructure: str; n_atoms: int
    initial_guess: float
    iterations: list[MeltingIterationRecord]
    report: dict[str, Any]
    def to_dict(self) -> dict[str, Any]: ...
```

---

## 4. The `LammpsEngine`

A new `engine/lammps.py::LammpsEngine` conforming to the `Engine` Protocol so the
*same* melting algorithm runs on LAMMPS.

- **Backend**: in-process LAMMPS via the `lammps` Python module
  (`/ptmp/hmai/lammps/python` + `liblammps.so`), reusing the notebook's
  `atomistics`-style command templating (NPT/NVT/NVE via jinja templates), or
  `pyiron_workflow_lammps` (pulled by the `free-energy` extra) if it offers a
  clean structure-in / results-out call. Final choice made in the plan.
- **Inputs**: `EngineInput ∈ {CalcInputStatic, CalcInputMinimize, CalcInputMD}` —
  same dataclasses as ASE; the engine translates to LAMMPS commands (units metal,
  `fix npt/nvt/nve`, `velocity create`, `box/relax`).
- **Potential**: a `LammpsPotential(pair_style, pair_coeff, potential_file)`
  (reuse `physics/free_energy/inputs.py::LammpsPotential`); for GRACE,
  `pair_style grace` + the exported model dir.
- **Output**: builds `EngineOutput` with `final_structure`, `final_energy`,
  `converged`, `final_forces/stress`, and trajectory `structures` (**with
  velocities**), `energies`, `stresses` so the coexistence analysis works
  unchanged.
- **Contract methods**: `working_directory`, `get_calculate_fn(structure)`,
  `with_working_directory(subdir)`; must be a pickleable `@dataclass`.
- **Validation**: a `tests/unit/engine/test_lammps_conformance.py` inheriting
  `pyiron_workflow_atomistics.testing.EngineConformanceTests`, plus a short MD
  smoke test (skipped if `lammps` import / LAMMPS build is unavailable).

---

## 5. Testing strategy

**Fast unit tests** (`tests/unit/physics/melting/`, no MD):
- `structures`: `create_coexistence_supercell` atom-count selection; `freeze_half`
  fixes exactly the lower-half indices; `strain_cell_along_z` scales only `c`.
- `analysis`: `cna_fractions` on a perfect fcc cell (≈100 % FCC); `solid_fraction_kde`
  on a synthetic half-fcc/half-random cell ≈ 0.5; `holes_mask` flags an injected
  void; `temperatures_/pressures_from_trajectory` against a hand-built
  `EngineOutput` with known momenta/stress (closed-form T and P).
- `fitting`: `predict_melting_point` on synthetic linear `P(strain)`,`T(strain)`
  with a known `T(P=0)`; `ratio_selection` picks the right contiguous window.

**Integration smoke** (`tests/integration/`, `@pytest.mark.slow`, EMT/Al, tiny
cell ~250 atoms, short MD): Step-1 bisection returns a plausible `T`; one
coexistence iteration runs end-to-end and returns a finite `T_next`.

**Engine conformance**: `LammpsEngine` passes `EngineConformanceTests`
(skip-guarded on LAMMPS availability).

---

## 6. Verification plan (this session)

1. **EMT/Al** (clean uv venv): run `estimate_melting_temperature` (cheap), then a
   scaled-down `coexistence_iteration`; report `T0` and `T_next` (EMT-Al melts
   ~well below the DFT 933 K; the number need only be self-consistent and finite).
2. **GRACE**: point the ASE engine at a cached GRACE-FM
   (`/ptmp/hmai/grace_cache`, e.g. `GRACE-1L-OAM`) and run Step 1 on a real
   element. (Install grace/tf into the venv, else run in the `grace` conda env
   with the package pip-installed.)
3. **LAMMPS + `pair_style grace`**: run a (scaled-down) full coexistence iteration
   through `LammpsEngine` using `/ptmp/hmai/lammps/build` + an exported GRACE
   model, confirming the engine-agnostic algorithm runs unchanged on LAMMPS.

Every run logs to `/ptmp` (not `/tmp`, which is node-local on Raven) and reports
full output paths.

---

## 7. Environment

- Fresh clone: `/ptmp/hmai/pwa_melting` (branch `feat/melting-point-interface-method`).
- Isolated `uv` venv `.venv` (Python 3.11): `uv pip install -e ".[test]"` +
  `structuretoolkit` + `pyscal3`. **Built & verified** 2026-06-27 (engine API +
  structuretoolkit import clean; ASE 3.28.0 / numpy 1.26.4 / pandas 3.0.3).
- GRACE step may need `tensorflow` + `grace` (TF_USE_LEGACY_KERAS=1) added to the
  venv, or fall back to the `grace` conda env.
- LAMMPS step uses the existing GRACE-enabled build at `/ptmp/hmai/lammps`.

---

## 8. Scope / non-goals
- No new public `EngineOutput` fields or conformance-contract changes.
- No attempt to reproduce the notebook's exact numbers (different engine,
  thermostats, cell sizes); correctness = self-consistent coexistence + sane
  fits, plus the analysis primitives matching closed-form unit tests.
- Production-scale (8000-atom, 50k-step) runs are *enabled* but not exhaustively
  run here; verification uses scaled-down sizes for turnaround.

---

## 9. Open questions for review
1. `LammpsEngine` backend: hand-rolled `lammps`-module templating (closest to the
   notebook) vs wrapping `pyiron_workflow_lammps`? (Plan will pick; leaning
   hand-rolled for control over velocity/stress trajectory capture.)
2. Place CNA/Voronoi as **general** `analysis/` quantities (reusable) vs keep them
   melting-local? (Leaning: keep melting-local now; promote later.)
3. GRACE for the ASE step — install into the uv venv, or run that step in the
   `grace` conda env with the package installed there?
