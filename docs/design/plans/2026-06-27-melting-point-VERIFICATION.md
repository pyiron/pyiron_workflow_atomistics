# Melting-point module — verification report (2026-06-27)

Branch `feat/melting-point-interface-method` (pwa_melting) + `feat/velocity-capture-for-melting` (pwl_dev).

## Summary

The full **interface/coexistence melting-point method** is implemented as an
engine-agnostic `physics/melting/` subpackage and verified end-to-end on the ASE
engine with both **EMT** and a real **GRACE foundation model**. The LAMMPS
(`pair_style grace`) path is wired and unit-verified, but blocked at runtime by a
GRACE-evaluator/model-build incompatibility in the LAMMPS side (not the module).

## Tests

- **Fast unit suite** (`tests/unit/analysis`, `tests/unit/physics/melting`): **20 passed**.
  Covers trajectory T/P (virial+kinetic, Voigt-6 & 3×3), CNA descriptors, Voronoi
  void detection, coexistence structures, KDE solid fraction, ratio-selection +
  T(P→0) extrapolation, dataclasses.
- **Integration (EMT/Al, `-m slow`)**: NPT-Berendsen metal smoke; MD step drivers
  (npt solid, interface build, strain scan); Step-1 bisection; one coexistence
  iteration; **full `calculate_melting_point` end-to-end** (build → relax → Step 1
  → Step 2 → `MeltingResult`, 201 s).
- **LAMMPS velocity-capture** (`pwl_dev`): unit test confirms trajectory frames
  carry per-atom velocities.

## Engine verification

| Engine | Calculator | Result |
|---|---|---|
| ASE | EMT (Al) | ✅ full method runs end-to-end; finite Tm + per-iteration provenance |
| ASE | **GRACE-1L-OAM** (`grace_fm`) | ✅ **Step-1 estimate = 1006 K for Al** (GPU node; DFT Al melts 933 K — a ~1006 K superheating Step-1 estimate is physically reasonable) |
| LAMMPS | `pair_style grace`, GRACE-1L-OAM | ⚠️ engine wiring correct, model loads, step-0 forces computed, velocity capture works — **GRACE evaluator crashes at step 1** |

### LAMMPS diagnosis
`in.lmp` is generated correctly (`pair_style grace`, `pair_coeff * * <model> Al`,
`vx vy vz` in the dump, NPT `fix npt ... iso`). `lmp` loads the saved_model, maps
Al→ACE species, builds neighbor lists, computes step-0 forces (1 dump frame), then
the GRACE evaluator aborts at the first dynamics step — on **both CPU and A100**.
This is a `pair_style grace` model/build pairing issue in the LAMMPS-GRACE stack,
independent of the melting module: the engine-agnostic algorithm is identical to
the EMT/GRACE-ASE paths that succeed. Building a `LammpsEngine` with
`command=<grace lmp>`, `path_to_model=<exported GRACE model>`,
`input_script_pair_style="grace"` and calling `coexistence_iteration(...,
npt_thermostat="nose-hoover")` will run once pointed at a `pair_style
grace`-compatible model/build (e.g. a `grace/fs` export, as in the calphy
campaigns). The per-session verification driver/SLURM scripts were kept local
(not committed upstream) to avoid hard-coded paths in the library.

## Bugs found & fixed during verification

1. **pyiron_workflow single-return rule** — `@pwf.as_function_node` bodies may have
   exactly one `return` and no nested function with a return. Rewrote
   `solid_fraction_kde`, `ratio_selection` (single-return) and moved the Step-1
   `heated` closure to a module-level `_heated_solid` helper.
2. **Bisection runaway** — undersampled MD never melts, so the "both solid" branch
   expanded temperature without bound (infinite loop / blow-up). Added
   `max_iterations` + `t_ceiling` guards to `estimate_melting_temperature`.
3. **Engine-agnostic NPT thermostat** — the method needs *isotropic/orthorhombic*
   NPT (CNA/Voronoi require it). ASE's isotropic NPT is `berendsen`; LAMMPS's is
   `nose-hoover` (`fix npt ... iso`), and `LammpsEngine` rejects non-nose-hoover
   NPT. ASE `nose-hoover` NPT goes full-triclinic and breaks pyscal Voronoi. Added
   a configurable `npt_thermostat` (default `berendsen`; pass `nose-hoover` for LAMMPS).

## Adversarial code review (multi-agent workflow)

A 4-dimension adversarial review (physics fidelity, numerics/units, pyiron_workflow
API, robustness) with per-finding skeptic verification: **34 candidates → 24
confirmed**. Fixed:

- **CRITICAL** NVE strain step re-seeded velocities at 2·T on an already-warm
  (NVT-equilibrated) config → kinetic T ≈ 1.5·T → runaway melting point. Now 1·T.
- **MAJOR** convergence loop stopped after `len(schedules)` iterations → rewrote to
  iterate until `|ΔT| ≤ convergence_goal`, holding the finest schedule (+ safety cap).
- **MAJOR** strain grid never re-centred → now re-centres on the fitted zero-pressure
  strain each iteration (`_next_center`).
- **MAJOR** `solid_fraction_kde` dropped the notebook's coordinate wrap → added
  `atoms.wrap()` before the KDE.
- **MINOR** `ratio_selection` selected by ratio value (duplicate ratios broke
  contiguity) → now selects by index.
- **MINOR** bisection inverted case (left molten/right solid) silently treated as
  both-molten → now an explicit branch.
- **MINOR** `create_coexistence_supercell` crashed for hcp (`cubic=True`) → hcp uses
  `orthorhombic=True`.
- **NIT** trajectory T/P silently returned 0 with no momenta → now raises.
- **NIT** removed dead `with_calc_input`.
- **LAMMPS (CRITICAL)** velocities attached in pyiron units (Å/fs) → `get_temperature()`
  ~103.6× low → convert by `/units.fs` in the patch.
- **LAMMPS (MAJOR)** `seed=None` written literally as "None" → guarded default.

Documented follow-ups (not blocking the ASE path):
- **LAMMPS stress convention** — `EngineOutput.stresses` is eV/Å³ virial (ASE) vs GPa
  total-pressure tensor (LAMMPS); `pressures_from_trajectory` assumes the ASE
  convention, so LAMMPS coexistence pressure needs the conventions reconciled.
- **n_print vs record_interval** — the ASE MD engine records every step (per the
  conformance test); for production-scale runs set `record_interval=n_print` to bound
  memory (engine + conformance-test change).
- Minor fidelity: solid-fraction gating/diamond detector, eager supercell build.

## Reproduce

```bash
# fast unit suite
/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis tests/unit/physics/melting -q
# EMT integration (slow)
/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration -q -m slow
# Step-1 estimate (EMT) — or --full for the complete method
/ptmp/hmai/pwa_melting/.venv/bin/python scripts/meltingpoint.py --element Al --a 4.05
# GRACE (ASE): build an ASEEngine with tensorpotential.calculator.grace_fm(<model>)
#   and call estimate_melting_temperature / calculate_melting_point on a GPU node.
# LAMMPS: build a LammpsEngine (command=<grace lmp>, path_to_model=<model>,
#   input_script_pair_style="grace") and call coexistence_iteration(npt_thermostat="nose-hoover").
```

## Output locations
- Code: `/ptmp/hmai/pwa_melting/pyiron_workflow_atomistics/{analysis,physics/melting}/`
- LAMMPS patch: `/ptmp/hmai/pwl_dev/pyiron_workflow_lammps/lammps.py`
- Verification runs: `/ptmp/hmai/pwa_melting/_verify_runs/`
  - `grace_ase_step1/summary.json` → GRACE-ASE Step-1 = 1006 K
  - `lammps_grace_coex/` → LAMMPS in.lmp + logs (step-1 crash)
  - `slurm_28365529.out/.err` → GPU job log
