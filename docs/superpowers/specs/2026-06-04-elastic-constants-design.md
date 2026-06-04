# Elastic-constants module for `pyiron_workflow_atomistics`

**Date:** 2026-06-04
**Branch:** `feat/elastic-constants`
**Author:** Han (with Claude)

## 1. Goal

Add a module `pyiron_workflow_atomistics/physics/elastic.py` that computes the full
elastic stiffness tensor and **all derived elastic constants listed in the Materials
Project elasticity methodology**
(<https://docs.materialsproject.org/methodology/materials-methodology/elasticity>),
using the existing `Engine` abstraction so any ASE calculator (EMT, GRACE, …) can drive it.

Verify the implementation two ways:
1. **Self-consistency / sanity:** EMT on cheap cubic metals inside the pytest suite.
2. **Physical correctness:** GRACE-2L-SMAX-large (foundation MLIP, via the ASE
   `grace_fm` calculator) on **Al, Cu, Si, Fe, Ni, W**, cross-checked against
   **Materials Project DFT** elastic tensors pulled live with `mp-api`.

## 2. Method (Materials Project convention)

MP computes elasticity by the **stress–strain** method (de Jong et al., *Sci. Data* 2,
150009, 2015), which is exactly what `pymatgen.analysis.elasticity` implements. We reuse
pymatgen so our conventions match MP by construction:

1. **Relax** the input structure (cell + ions) to a near-zero-stress reference.
2. **Deform:** apply the MP-standard independent deformations via
   `pymatgen...strain.DeformedStructureSet` — normal strains `[-0.01, -0.005, 0.005, 0.01]`
   and shear strains `[-0.06, -0.03, 0.03, 0.06]` → **24 deformed cells** (6 modes × 4 magnitudes).
3. **Re-relax ions at fixed cell** (`CalcInputMinimize(relax_cell=False)`, MP's ISIF=2
   analogue) for each deformed cell and read the **stress**.
4. **Fit** the 6×6 stiffness tensor by least squares with
   `ElasticTensor.from_independent_strains(strains, stresses, eq_stress=<reference stress>)`,
   then symmetrize and rotate to **IEEE** orientation (`convert_to_ieee`).
5. **Derive** all constants from the tensor.

### Constants emitted (the full MP list)
- **Elastic tensor** `C_ij` (6×6, GPa) — raw fit and IEEE-rotated.
- **Compliance tensor** `S_ij` (GPa⁻¹).
- **Bulk modulus** `K`: Voigt `K_V`, Reuss `K_R`, Voigt–Reuss–Hill `K_VRH`.
- **Shear modulus** `G`: Voigt `G_V`, Reuss `G_R`, VRH `G_VRH`.
- **Young's modulus** `E` (from VRH).
- **Poisson ratio** `ν` (homogeneous).
- **Universal anisotropy index** `A_U`.
- **Mechanical stability** flag (Born stability criteria via pymatgen).

## 3. Unit & sign convention (the one real risk)

ASE returns stress in **eV/Å³**, Voigt order `[xx, yy, zz, yz, xz, xy]`; pymatgen elastic
math works in **GPa**. Conversion: `1 eV/Å³ = 160.21766208 GPa`.

The **sign** of the ASE→pymatgen stress map determines whether moduli come out positive.
This is handled by a single, explicitly documented constant and is **validated empirically**:
the EMT pytest must produce `C11 > 0` and bulk modulus within ~15 % of the known EMT/literature
value for Cu and Al, and the GRACE run must match MP DFT in sign and rough magnitude. A sign
error cannot pass either gate.

## 4. Components

All nodes follow existing conventions (`@pwf.as_function_node`, `@pwf.as_macro_node`),
mirroring `physics/bulk.py` and `physics/surface.py`.

### `physics/elastic.py`
**Conversion / engine helpers**
- `_ase_voigt_stress_to_gpa(stress_voigt) -> np.ndarray` — eV/Å³ Voigt → GPa, with the
  documented sign convention.
- `_with_calc_input(engine, calc_input) -> Engine` — `dataclasses.replace` the engine's
  `EngineInput` so the macro can switch between full-relax and fixed-cell-relax modes while
  the user passes a single engine (carrying the calculator + working dir).

**Function nodes**
- `generate_mp_deformations(structure, norm_strains=..., shear_strains=...)` →
  `(deformed_structures: list[Atoms], strains: list[np.ndarray])`. Wraps
  `DeformedStructureSet` via `AseAtomsAdaptor`; returns the strain (Green–Lagrange or
  deformation-gradient strain, matching pymatgen) alongside each cell so the fit stays paired.
- `extract_stresses_gpa(engine_outputs)` → `list[np.ndarray]` (GPa stress tensors).
- `fit_elastic_tensor(strains, stresses, structure, eq_stress=None)` →
  `ElasticTensor` (IEEE-rotated, symmetrized).
- `elastic_constants_summary(elastic_tensor, structure)` → `dict` with every constant in §2
  (built from `ElasticTensor.property_dict` + tensors + Born-stability check).

**Macro node**
- `calculate_elastic_constants(wf, structure, engine, relax_initial=True, norm_strains=...,
  shear_strains=..., fmax=1e-3, max_steps=300)` →
  `(relaxed_structure, elastic_tensor, elastic_tensor_ieee, compliance_tensor,
    elastic_constants)`.
  Wires: optional initial full relax (`relax_cell=True`) → `generate_mp_deformations` →
  per-cell fixed-cell relax via `evaluate_structures`-style loop with a fixed-cell engine →
  `extract_stresses_gpa` → `fit_elastic_tensor` → `elastic_constants_summary`.
  Each deformation runs in its own subdir (`subengine`, like the existing modules).

### `tests/unit/physics/test_elastic_workflows.py`
- Fast EMT smoke/correctness test on **Cu fcc** (and Al): run the macro with a coarse strain
  set, assert the stiffness is positive-definite, cubic symmetry holds approximately
  (`C11>C12`, `C44>0`), and `K_VRH` is within tolerance of the known EMT value. Marked `slow`
  if needed, following the existing `test_bulk_workflows.py` style.
- Pure-unit tests for `_ase_voigt_stress_to_gpa` (units + sign) and
  `elastic_constants_summary` (given a known analytic `ElasticTensor`, check derived K/G/E/ν/A_U).

### `verification/elastic_grace_vs_mp.py` (script, not shipped in the package)
Standalone validation harness (lives under `verification/` in the repo, runnable on Raven):
- For each of **Al, Cu, Si, Fe, Ni, W**: build the conventional bulk structure (or pull the
  MP relaxed structure), run `calculate_elastic_constants` with
  `grace_fm("GRACE-2L-SMAX-large")` (`GRACE_CACHE=/ptmp/hmai/grace_cache`), and pull the MP
  elastic doc via `MPRester(MP_API_KEY).materials.elasticity.search(...)`.
- Emit a comparison table (GRACE vs MP: `K_VRH`, `G_VRH`, `E`, `ν`, `A_U`, and `C11/C12/C44`
  for cubics) as CSV + markdown, with per-material % error. Materials are independent →
  run in parallel (one subagent / process per material).

## 5. Data flow

```
structure ─▶ [initial full relax] ─▶ relaxed_structure
                                         │
              DeformedStructureSet (MP strains)
                                         ▼
                 24 deformed cells ─▶ [fixed-cell ion relax, per cell] ─▶ stresses (GPa)
                                         ▼
        ElasticTensor.from_independent_strains(strains, stresses, eq_stress)
                                         ▼
      IEEE-rotated C_ij ─▶ S_ij, K/G (V,R,VRH), E, ν, A_U, Born stability  ─▶ dict
```

## 6. Error handling
- Non-converged ionic relaxations: surface a warning and still record the stress
  (matches MP, which tolerates loose convergence), but flag the material in the summary.
- Magnetic materials (Fe, Ni): GRACE-SMAX carries its own magnetic treatment; we pass the
  structure through unchanged and rely on the calculator. The MP reference is the
  spin-polarized DFT elastic doc.
- Missing MP elastic data for a chosen material: skip with a logged note, don't fail the run.
- Singular/ill-conditioned strain set: pymatgen raises; we catch and report which strain
  modes were degenerate.

## 7. Out of scope (YAGNI)
- Temperature-dependent / finite-T elastic constants.
- Non-cubic-specific reporting beyond what pymatgen's generic tensor already gives
  (the method is fully general; we just don't add per-symmetry pretty-printers).
- LAMMPS engine path (ASE engine only, per the task).

## 8. Verification gates (definition of done)
1. `pytest tests/unit/physics/test_elastic_workflows.py` passes (EMT Cu/Al positive-definite,
   correct sign, K within tolerance).
2. `verification/elastic_grace_vs_mp.py` runs on all six materials and produces the
   GRACE-vs-MP table; GRACE-2L-SMAX-large reproduces MP DFT moduli in sign and to within a
   physically reasonable margin (report actual % errors — no hidden pass/fail fudging).
