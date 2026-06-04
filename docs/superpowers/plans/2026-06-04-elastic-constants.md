# Elastic-constants module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pyiron_workflow_atomistics/physics/elastic.py` computing the full elastic stiffness tensor and every derived constant in the Materials Project elasticity methodology, drivable by any ASE-backed `Engine`; verify with EMT (pytest) and GRACE-2L-SMAX-large vs MP DFT.

**Architecture:** pyiron function-nodes + one `calculate_elastic_constants` macro. The numerics reuse `pymatgen.analysis.elasticity` (the same library MP uses): MP-standard deformations → fixed-cell ion relax per cell → least-squares fit of `C_ij` → derived moduli. Follows the exact conventions in `physics/bulk.py`.

**Tech Stack:** Python 3.11, `pyiron_workflow`, `ase==3.28`, `pymatgen` (analysis.elasticity), `numpy`, `tensorpotential`/`grace_fm` (verification only), `mp-api` (verification only).

**Environment:** venv at `/ptmp/hmai/pwa_elastic/.venv`. Run python as `/ptmp/hmai/pwa_elastic/.venv/bin/python` and pytest as `/ptmp/hmai/pwa_elastic/.venv/bin/pytest`. Repo root: `/ptmp/hmai/pwa_elastic`. Branch: `feat/elastic-constants`. Set `export TF_CPP_MIN_LOG_LEVEL=3` to silence TF logs when GRACE is imported.

**Key API facts (verified against the installed packages):**
- `EngineOutput` (engine/protocol.py) fields: `final_structure`, `final_energy`, `converged`, `final_forces`, `final_stress` (3×3), `final_stress_voigt` (6, order `[xx,yy,zz,yz,xz,xy]`), `final_volume`, `final_magmoms`, plus relaxation trajectory lists.
- `ASEEngine(EngineInput=<CalcInputStatic|CalcInputMinimize|CalcInputMD>, calculator=<ase Calculator>, working_directory=..., ...)` is a dataclass; `engine.with_working_directory(subdir)` returns a copy in a subdir.
- `CalcInputMinimize(force_convergence_tolerance=1e-2, energy_convergence_tolerance=1e-5, max_iterations=1_000_000, relax_cell=False)`. `relax_cell=False` ⇒ ions relax at fixed cell (MP ISIF=2 analogue); `relax_cell=True` ⇒ full cell+ion relax.
- `calculate.node_function(structure=..., engine=...) -> EngineOutput` (single-point/relax per the engine's `EngineInput`).
- `evaluate_structures(structures, engine, parent_working_directory=".")` (physics/bulk.py) loops `engine.with_working_directory(f"strain_{i:03d}")` and returns a list of `EngineOutput` in input order.
- pymatgen: `DeformedStructureSet(structure, norm_strains=(-0.01,-0.005,0.005,0.01), shear_strains=(-0.06,-0.03,0.03,0.06), symmetry=False)` exposes `.deformations` (list of `Deformation`) and `.deformed_structures`. `Deformation.green_lagrange_strain` → `Strain`.
- `ElasticTensor.from_independent_strains(strains, stresses, eq_stress=None, vasp=False)`; `ElasticTensor.from_voigt(6x6)`; `et.calculate_stress(strain) -> Stress`; properties: `voigt`, `compliance_tensor`, `k_voigt/k_reuss/k_vrh`, `g_voigt/g_reuss/g_vrh`, `y_mod`, `homogeneous_poisson`, `universal_anisotropy`, `property_dict`, `convert_to_ieee(structure)`.
- `AseAtomsAdaptor.get_structure(atoms)` / `.get_atoms(structure)` convert ASE↔pymatgen.
- Unit conversion: `1 eV/Å³ = 160.21766208 GPa`. ASE `get_stress` is tension-positive (Cauchy). Sign into pymatgen is validated by the EMT test (Cu must give `C11>0`); if the test shows inverted signs, set `_ASE_STRESS_SIGN = -1.0`.

---

### Task 1: Module scaffold + stress unit/sign helper

**Files:**
- Create: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/physics/test_elastic.py
import numpy as np
from pyiron_workflow_atomistics.physics.elastic import voigt_stress_to_gpa, EV_PER_A3_TO_GPA


def test_ev_per_a3_to_gpa_constant():
    assert abs(EV_PER_A3_TO_GPA - 160.21766208) < 1e-6


def test_voigt_stress_to_gpa_shape_and_units():
    # 1 eV/A^3 hydrostatic in Voigt -> 3x3 GPa tensor
    voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    tensor = voigt_stress_to_gpa(voigt)
    assert tensor.shape == (3, 3)
    np.testing.assert_allclose(np.diag(tensor), [160.21766208] * 3, rtol=1e-6)
    # off-diagonal shear placement: voigt[3]=yz, [4]=xz, [5]=xy
    voigt2 = np.array([0.0, 0.0, 0.0, 2.0, 3.0, 4.0])
    t2 = voigt_stress_to_gpa(voigt2)
    np.testing.assert_allclose(t2[1, 2] / EV_PER_A3_TO_GPA, 2.0)  # yz
    np.testing.assert_allclose(t2[0, 2] / EV_PER_A3_TO_GPA, 3.0)  # xz
    np.testing.assert_allclose(t2[0, 1] / EV_PER_A3_TO_GPA, 4.0)  # xy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py -q`
Expected: FAIL (ModuleNotFoundError / ImportError: cannot import name).

- [ ] **Step 3: Write minimal implementation**

```python
# pyiron_workflow_atomistics/physics/elastic.py
"""Elastic constants via the Materials Project stress-strain method.

Computes the full 6x6 stiffness tensor and all derived constants listed at
https://docs.materialsproject.org/methodology/materials-methodology/elasticity
using pymatgen.analysis.elasticity (the same library Materials Project uses),
driven by any ASE-backed Engine.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

EV_PER_A3_TO_GPA = 160.21766208
# ASE get_stress is tension-positive (Cauchy), same as pymatgen elasticity.
# Validated by the EMT test (Cu must give C11 > 0); flip to -1.0 if inverted.
_ASE_STRESS_SIGN = 1.0


def voigt_stress_to_gpa(stress_voigt) -> np.ndarray:
    """ASE Voigt stress (eV/A^3, order [xx,yy,zz,yz,xz,xy]) -> 3x3 tensor in GPa."""
    s = np.asarray(stress_voigt, dtype=float) * _ASE_STRESS_SIGN * EV_PER_A3_TO_GPA
    xx, yy, zz, yz, xz, xy = s
    return np.array(
        [[xx, xy, xz],
         [xy, yy, yz],
         [xz, yz, zz]]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): stress unit/sign helper"
```

---

### Task 2: `_with_calc_input` engine mode-swap helper

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test**

```python
def test_with_calc_input_swaps_engine_mode():
    from ase.calculators.emt import EMT
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.elastic import with_calc_input

    base = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=".")
    relaxed = with_calc_input(base, CalcInputMinimize(relax_cell=False, force_convergence_tolerance=1e-3))
    assert isinstance(relaxed.EngineInput, CalcInputMinimize)
    assert relaxed.EngineInput.relax_cell is False
    assert relaxed.EngineInput.force_convergence_tolerance == 1e-3
    # original is untouched (immutability via dataclasses.replace)
    assert isinstance(base.EngineInput, CalcInputStatic)
    # calculator is preserved
    assert relaxed.calculator is base.calculator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_with_calc_input_swaps_engine_mode -q`
Expected: FAIL (ImportError: cannot import name 'with_calc_input').

- [ ] **Step 3: Write minimal implementation** (append to elastic.py)

```python
def with_calc_input(engine, calc_input):
    """Return a copy of a dataclass engine with its EngineInput replaced.

    Lets the elastic macro switch a single user-supplied engine between
    full-relax and fixed-cell-relax modes without the user wiring two engines.
    """
    return dataclasses.replace(engine, EngineInput=calc_input)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_with_calc_input_swaps_engine_mode -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): engine mode-swap helper"
```

---

### Task 3: `generate_mp_deformations` node

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generate_mp_deformations_count_and_pairing():
    from ase.build import bulk
    from pyiron_workflow_atomistics.physics.elastic import generate_mp_deformations

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = generate_mp_deformations.node_function(atoms)
    structs = out["deformed_structures"]
    strains = out["strains"]
    # 6 strain modes x 4 magnitudes = 24
    assert len(structs) == 24
    assert len(strains) == 24
    assert all(isinstance(s, Atoms) for s in structs)
    # strains are 3x3 arrays
    assert np.asarray(strains[0]).shape == (3, 3)
    # deformed cells actually differ from the reference cell
    ref = atoms.cell.array
    assert any(not np.allclose(s.cell.array, ref) for s in structs)
```
Add `from ase import Atoms` import already present at top of test module; add `from ase.build import bulk` locally as shown.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_generate_mp_deformations_count_and_pairing -q`
Expected: FAIL (cannot import name 'generate_mp_deformations').

- [ ] **Step 3: Write minimal implementation** (append to elastic.py)

```python
@pwf.as_function_node("deformed_structures", "strains")
def generate_mp_deformations(
    structure: Atoms,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
):
    """MP-standard deformation set (6 modes x 4 magnitudes = 24 cells).

    Returns the deformed ASE structures and their Green-Lagrange strain
    tensors (3x3), paired in order for the downstream fit.
    """
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg = AseAtomsAdaptor.get_structure(structure)
    dss = DeformedStructureSet(
        pmg,
        norm_strains=list(norm_strains),
        shear_strains=list(shear_strains),
    )
    deformed_structures = [AseAtomsAdaptor.get_atoms(s) for s in dss.deformed_structures]
    strains = [np.asarray(d.green_lagrange_strain) for d in dss.deformations]
    return deformed_structures, strains
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_generate_mp_deformations_count_and_pairing -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): MP deformation generator node"
```

---

### Task 4: `extract_stresses_gpa` node

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_stresses_gpa_from_engine_outputs():
    from types import SimpleNamespace
    from pyiron_workflow_atomistics.physics.elastic import extract_stresses_gpa, EV_PER_A3_TO_GPA

    o1 = SimpleNamespace(final_stress_voigt=np.array([1.0, 0, 0, 0, 0, 0]))
    o2 = SimpleNamespace(final_stress_voigt=np.array([0, 0, 0, 0, 0, 0.5]))
    out = extract_stresses_gpa.node_function([o1, o2])
    stresses = out  # node returns single output "stresses"
    assert len(stresses) == 2
    np.testing.assert_allclose(stresses[0][0, 0], 1.0 * EV_PER_A3_TO_GPA)
    np.testing.assert_allclose(stresses[1][0, 1], 0.5 * EV_PER_A3_TO_GPA)
```
(Note: a single-output `@as_function_node("stresses")` returns the bare value from `.node_function`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_extract_stresses_gpa_from_engine_outputs -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation** (append to elastic.py)

```python
@pwf.as_function_node("stresses")
def extract_stresses_gpa(engine_outputs):
    """3x3 stress tensors in GPa from a list of EngineOutput (input order)."""
    return [voigt_stress_to_gpa(o.final_stress_voigt) for o in engine_outputs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_extract_stresses_gpa_from_engine_outputs -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): stress extraction node"
```

---

### Task 5: `fit_elastic_tensor` node (recovers a known tensor — validates pairing + sign)

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fit_elastic_tensor_recovers_known_cubic():
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet
    from pyiron_workflow_atomistics.physics.elastic import fit_elastic_tensor

    # Known cubic stiffness (GPa)
    C11, C12, C44 = 200.0, 130.0, 100.0
    voigt = np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44],
    ])
    C_true = ElasticTensor.from_voigt(voigt)

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    pmg = AseAtomsAdaptor.get_structure(atoms)
    dss = DeformedStructureSet(pmg)
    strains = [np.asarray(d.green_lagrange_strain) for d in dss.deformations]
    # synthesize stresses (GPa) from the known tensor for each strain
    stresses = [np.asarray(C_true.calculate_stress(s)) for s in strains]

    out = fit_elastic_tensor.node_function(strains=strains, stresses=stresses, structure=atoms)
    C_fit = out  # single output "elastic_tensor"
    np.testing.assert_allclose(C_fit.voigt, voigt, atol=1.0)  # within 1 GPa
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_fit_elastic_tensor_recovers_known_cubic -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation** (append to elastic.py)

```python
@pwf.as_function_node("elastic_tensor")
def fit_elastic_tensor(strains, stresses, structure: Atoms, eq_stress=None):
    """Least-squares fit of the 6x6 stiffness tensor (GPa), MP convention.

    strains: list of 3x3 strain tensors (Green-Lagrange).
    stresses: list of 3x3 stress tensors in GPa (same order).
    structure: the reference (relaxed) ASE structure, used for symmetry.
    eq_stress: 3x3 reference stress in GPa (defaults to zero).
    Returns a pymatgen ElasticTensor, symmetrized.
    """
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity.strain import Strain
    from pymatgen.analysis.elasticity.stress import Stress

    pmg_strains = [Strain(s) for s in strains]
    pmg_stresses = [Stress(s) for s in stresses]
    eq = None if eq_stress is None else Stress(eq_stress)
    et = ElasticTensor.from_independent_strains(
        pmg_strains, pmg_stresses, eq_stress=eq, vasp=False
    )
    return et.voigt_symmetrized.fit_to_structure(
        __import__("pymatgen.io.ase", fromlist=["AseAtomsAdaptor"]).AseAtomsAdaptor.get_structure(structure)
    ) if hasattr(et, "fit_to_structure") else et
```
Note: keep the implementation simple — if `fit_to_structure` chaining is awkward, return `ElasticTensor(et.voigt_symmetrized)`. The test only requires recovering the Voigt matrix to 1 GPa, so the minimal body is:
```python
    et = ElasticTensor.from_independent_strains(pmg_strains, pmg_stresses, eq_stress=eq, vasp=False)
    return ElasticTensor(et.voigt_symmetrized)
```
Use this minimal body.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_fit_elastic_tensor_recovers_known_cubic -q`
Expected: PASS. If `C_fit.voigt` comes back with the wrong sign (negative diagonal), set `_ASE_STRESS_SIGN = -1.0` is NOT relevant here (synthetic stresses) — instead the fit/pairing is wrong; debug strain↔stress ordering.

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): tensor fit node"
```

---

### Task 6: `elastic_constants_summary` node (all MP constants)

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test**

```python
def test_elastic_constants_summary_known_cubic():
    from ase.build import bulk
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pyiron_workflow_atomistics.physics.elastic import elastic_constants_summary

    C11, C12, C44 = 200.0, 130.0, 100.0
    voigt = np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44],
    ])
    et = ElasticTensor.from_voigt(voigt)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    d = elastic_constants_summary.node_function(et, atoms)

    # Cubic Voigt bulk modulus K_V = (C11 + 2 C12)/3
    np.testing.assert_allclose(d["K_VRH"], (C11 + 2 * C12) / 3.0, rtol=1e-6)
    assert "K_Voigt" in d and "K_Reuss" in d
    assert "G_Voigt" in d and "G_Reuss" in d and "G_VRH" in d
    assert "youngs_modulus" in d and "poisson_ratio" in d
    assert "universal_anisotropy" in d
    assert d["mechanically_stable"] is True
    # full tensors present
    assert np.asarray(d["elastic_tensor_voigt"]).shape == (6, 6)
    assert np.asarray(d["compliance_tensor_voigt"]).shape == (6, 6)
    assert np.asarray(d["elastic_tensor_ieee"]).shape == (6, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_elastic_constants_summary_known_cubic -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation** (append to elastic.py)

```python
@pwf.as_function_node("elastic_constants")
def elastic_constants_summary(elastic_tensor, structure: Atoms) -> dict:
    """Every elastic constant in the MP elasticity methodology, as a flat dict.

    Includes full stiffness (raw + IEEE) and compliance tensors, bulk and shear
    moduli (Voigt/Reuss/Hill), Young's modulus, Poisson ratio, universal
    anisotropy, and a mechanical-stability flag (Born criteria, cubic + general).
    Moduli in GPa.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    et = elastic_tensor
    pmg = AseAtomsAdaptor.get_structure(structure)
    try:
        et_ieee = et.convert_to_ieee(pmg)
    except Exception:
        et_ieee = et

    C = np.asarray(et.voigt)
    # General Born stability: stiffness matrix positive-definite
    eigvals = np.linalg.eigvalsh(C)
    mech_stable = bool(np.all(eigvals > 0))

    d = {
        "K_Voigt": float(et.k_voigt),
        "K_Reuss": float(et.k_reuss),
        "K_VRH": float(et.k_vrh),
        "G_Voigt": float(et.g_voigt),
        "G_Reuss": float(et.g_reuss),
        "G_VRH": float(et.g_vrh),
        "youngs_modulus": float(et.y_mod) / 1e9 if et.y_mod > 1e6 else float(et.y_mod),
        "poisson_ratio": float(et.homogeneous_poisson),
        "universal_anisotropy": float(et.universal_anisotropy),
        "mechanically_stable": mech_stable,
        "stiffness_eigenvalues": eigvals.tolist(),
        "elastic_tensor_voigt": C.tolist(),
        "elastic_tensor_ieee": np.asarray(et_ieee.voigt).tolist(),
        "compliance_tensor_voigt": np.asarray(et.compliance_tensor.voigt).tolist(),
    }
    return d
```
Note on `youngs_modulus`: pymatgen's `y_mod` returns Pa-scale numbers in some versions and GPa in others; the guard normalizes to GPa. The subagent must verify `d["youngs_modulus"]` is ~ 9KG/(3K+G) in GPa for the test tensor (≈ 130 GPa here) and adjust the guard if needed. Add an assertion in the test:
```python
    K, G = d["K_VRH"], d["G_VRH"]
    expected_E = 9 * K * G / (3 * K + G)
    np.testing.assert_allclose(d["youngs_modulus"], expected_E, rtol=1e-3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py::test_elastic_constants_summary_known_cubic -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): MP constants summary node"
```

---

### Task 7: `calculate_elastic_constants` macro + EMT end-to-end test

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/elastic.py`
- Test: `tests/unit/physics/test_elastic_workflows.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/physics/test_elastic_workflows.py
import numpy as np
import pytest


@pytest.mark.slow
def test_calculate_elastic_constants_emt_cu(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = calculate_elastic_constants(
        structure=structure,
        engine=engine,
        relax_initial=True,
    )
    out = wf.run()
    d = out["elastic_constants"]
    C = np.asarray(d["elastic_tensor_voigt"])
    # Physically sane cubic metal: positive-definite, C11>C12, C44>0, sign correct
    assert d["mechanically_stable"] is True, f"C eigenvalues {d['stiffness_eigenvalues']}"
    assert C[0, 0] > 0 and C[0, 0] > C[0, 1]
    assert C[3, 3] > 0
    assert d["K_VRH"] > 0 and d["G_VRH"] > 0
    # EMT Cu bulk modulus is ~ 130-180 GPa range
    assert 80 < d["K_VRH"] < 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic_workflows.py -q`
Expected: FAIL (cannot import name 'calculate_elastic_constants').

- [ ] **Step 3: Write minimal implementation** (append to elastic.py)

```python
@pwf.as_function_node("eq_stress")
def _reference_stress_gpa(engine_output):
    """Reference (relaxed) stress in GPa as a 3x3 tensor, for eq_stress."""
    return voigt_stress_to_gpa(engine_output.final_stress_voigt)


@pwf.as_macro_node(
    "relaxed_structure",
    "elastic_tensor",
    "elastic_constants",
)
def calculate_elastic_constants(
    wf,
    structure: Atoms,
    engine,
    relax_initial: bool = True,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
    fmax: float = 1e-3,
    max_iterations: int = 300,
):
    """Full MP-style elastic-constants workflow.

    relax_initial: full cell+ion relax of the input before deforming.
    Deformations are MP-standard; each deformed cell has its ions relaxed at
    fixed cell. Returns the relaxed reference structure, the fitted pymatgen
    ElasticTensor, and a flat dict of all MP elastic constants.
    """
    from pyiron_workflow_atomistics.engine import CalcInputMinimize
    from pyiron_workflow_atomistics.physics.bulk import evaluate_structures

    fixed_cell = CalcInputMinimize(
        relax_cell=False,
        force_convergence_tolerance=fmax,
        max_iterations=max_iterations,
    )

    if relax_initial:
        full_relax = CalcInputMinimize(
            relax_cell=True,
            force_convergence_tolerance=fmax,
            max_iterations=max_iterations,
        )
        wf.relax_engine = with_calc_input_node(engine, full_relax)
        wf.relax = calculate(structure=structure, engine=wf.relax_engine)
        ref_structure = wf.relax.outputs.engine_output.final_structure
        wf.eq_stress = _reference_stress_gpa(wf.relax.outputs.engine_output)
        eq_stress = wf.eq_stress
    else:
        ref_structure = structure
        eq_stress = None

    wf.deform = generate_mp_deformations(
        ref_structure, norm_strains=norm_strains, shear_strains=shear_strains
    )
    wf.deform_engine = with_calc_input_node(engine, fixed_cell)
    wf.evals = evaluate_structures(
        structures=wf.deform.outputs.deformed_structures,
        engine=wf.deform_engine,
    )
    wf.stresses = extract_stresses_gpa(wf.evals.outputs.engine_output_lst)
    wf.fit = fit_elastic_tensor(
        strains=wf.deform.outputs.strains,
        stresses=wf.stresses,
        structure=ref_structure,
        eq_stress=eq_stress,
    )
    wf.summary = elastic_constants_summary(wf.fit, ref_structure)

    return ref_structure, wf.fit, wf.summary
```
Add a node wrapper for `with_calc_input` so it can be used inside the macro graph:
```python
@pwf.as_function_node("engine")
def with_calc_input_node(engine, calc_input):
    return with_calc_input(engine, calc_input)
```
(Place `with_calc_input_node` above the macro. The plain `with_calc_input` function from Task 2 stays for direct use/tests.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic_workflows.py -q`
Expected: PASS. **If `mechanically_stable` is False with all-negative eigenvalues (inverted sign), set `_ASE_STRESS_SIGN = -1.0` in elastic.py and re-run.** This is the empirical sign-validation gate from the spec.

- [ ] **Step 5: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add pyiron_workflow_atomistics/physics/elastic.py tests/unit/physics/test_elastic_workflows.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "feat(elastic): calculate_elastic_constants macro + EMT e2e test"
```

---

### Task 8: Public import smoke-test (per-topic convention)

**Convention:** `physics/__init__.py` intentionally re-exports nothing — the public API is the per-topic module path `from pyiron_workflow_atomistics.physics.elastic import ...`. **Do NOT add re-exports to `physics/__init__.py`.** This task just locks the per-topic import surface and runs the full suite.

**Files:**
- Test: `tests/unit/physics/test_elastic.py`

- [ ] **Step 1: Write the failing test** (will already pass if Tasks 1-7 are done; it guards the public surface)

```python
def test_public_import_surface():
    from pyiron_workflow_atomistics.physics.elastic import (
        calculate_elastic_constants,
        generate_mp_deformations,
        extract_stresses_gpa,
        fit_elastic_tensor,
        elastic_constants_summary,
        voigt_stress_to_gpa,
        with_calc_input,
    )
    assert all(callable(x) for x in (
        calculate_elastic_constants, generate_mp_deformations,
        extract_stresses_gpa, fit_elastic_tensor,
        elastic_constants_summary, voigt_stress_to_gpa, with_calc_input,
    ))
```

- [ ] **Step 2: Run the whole elastic suite**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics/test_elastic.py tests/unit/physics/test_elastic_workflows.py -q`
Expected: all PASS (the slow EMT e2e test included).

- [ ] **Step 3: Confirm nothing else in the repo broke**

Run: `cd /ptmp/hmai/pwa_elastic && .venv/bin/pytest tests/unit/physics -q`
Expected: existing physics tests still PASS (no regressions from the new module).

- [ ] **Step 4: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add tests/unit/physics/test_elastic.py
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "test(elastic): lock public per-topic import surface"
```

---

### Task 9: GRACE-2L-SMAX-large vs Materials Project verification script

**Files:**
- Create: `verification/elastic_grace_vs_mp.py`
- Create: `verification/README.md`

This is a standalone script (not part of the package, not pytest). It is the spec's physical-correctness gate. Materials are independent and may be run in parallel.

- [ ] **Step 1: Write the script**

```python
# verification/elastic_grace_vs_mp.py
"""GRACE-2L-SMAX-large elastic constants vs Materials Project DFT.

Usage:
    GRACE_CACHE=/ptmp/hmai/grace_cache MP_API_KEY=... \
      /ptmp/hmai/pwa_elastic/.venv/bin/python verification/elastic_grace_vs_mp.py \
      --material Cu --mp-id mp-30 --out-dir results

Runs one material; the driver loop / parallel dispatch runs all six.
"""
import argparse
import csv
import json
import os

import numpy as np

MATERIALS = {
    # symbol: (ase bulk args, crystalstructure, lattice a, MP id)
    "Cu": dict(name="Cu", crystalstructure="fcc", a=3.615, mp_id="mp-30"),
    "Al": dict(name="Al", crystalstructure="fcc", a=4.05, mp_id="mp-134"),
    "Si": dict(name="Si", crystalstructure="diamond", a=5.43, mp_id="mp-149"),
    "Fe": dict(name="Fe", crystalstructure="bcc", a=2.87, mp_id="mp-13"),
    "Ni": dict(name="Ni", crystalstructure="fcc", a=3.52, mp_id="mp-23"),
    "W":  dict(name="W",  crystalstructure="bcc", a=3.16, mp_id="mp-91"),
}


def build_structure(spec):
    from ase.build import bulk
    return bulk(spec["name"], spec["crystalstructure"], a=spec["a"], cubic=True)


def grace_elastic(structure):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorpotential.calculator import grace_fm
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants

    calc = grace_fm("GRACE-2L-SMAX-large")
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=calc,
        working_directory=os.path.abspath("grace_workdir"),
    )
    wf = calculate_elastic_constants(structure=structure, engine=engine, relax_initial=True)
    out = wf.run()
    return out["elastic_constants"]


def mp_reference(mp_id, api_key):
    from mp_api.client import MPRester
    with MPRester(api_key) as m:
        docs = m.materials.elasticity.search(
            material_ids=[mp_id],
            fields=["material_id", "formula_pretty", "bulk_modulus",
                    "shear_modulus", "elastic_tensor", "homogeneous_poisson",
                    "young_modulus", "universal_anisotropy"],
        )
    d = docs[0]
    C = np.asarray(d.elastic_tensor.ieee_format)
    return {
        "K_VRH": d.bulk_modulus["vrh"],
        "G_VRH": d.shear_modulus["vrh"],
        "youngs_modulus": getattr(d, "young_modulus", None),
        "poisson_ratio": d.homogeneous_poisson,
        "universal_anisotropy": d.universal_anisotropy,
        "C11": C[0, 0], "C12": C[0, 1], "C44": C[3, 3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True, choices=list(MATERIALS))
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spec = MATERIALS[args.material]
    structure = build_structure(spec)
    grace = grace_elastic(structure)
    mp = mp_reference(spec["mp_id"], os.environ["MP_API_KEY"])

    Cg = np.asarray(grace["elastic_tensor_ieee"])
    row = {
        "material": args.material, "mp_id": spec["mp_id"],
        "grace_K_VRH": grace["K_VRH"], "mp_K_VRH": mp["K_VRH"],
        "grace_G_VRH": grace["G_VRH"], "mp_G_VRH": mp["G_VRH"],
        "grace_E": grace["youngs_modulus"], "mp_E": mp["youngs_modulus"],
        "grace_poisson": grace["poisson_ratio"], "mp_poisson": mp["poisson_ratio"],
        "grace_A_U": grace["universal_anisotropy"], "mp_A_U": mp["universal_anisotropy"],
        "grace_C11": Cg[0, 0], "mp_C11": mp["C11"],
        "grace_C12": Cg[0, 1], "mp_C12": mp["C12"],
        "grace_C44": Cg[3, 3], "mp_C44": mp["C44"],
        "K_pct_err": 100 * (grace["K_VRH"] - mp["K_VRH"]) / mp["K_VRH"],
        "G_pct_err": 100 * (grace["G_VRH"] - mp["G_VRH"]) / mp["G_VRH"],
    }
    with open(os.path.join(args.out_dir, f"{args.material}.json"), "w") as f:
        json.dump({"grace": grace, "mp": mp, "row": row}, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, f"{args.material}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader(); w.writerow(row)
    print(json.dumps(row, indent=2, default=float))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run one cheap material**

Run:
```bash
cd /ptmp/hmai/pwa_elastic
GRACE_CACHE=/ptmp/hmai/grace_cache MP_API_KEY=$MP_API_KEY TF_CPP_MIN_LOG_LEVEL=3 \
  .venv/bin/python verification/elastic_grace_vs_mp.py --material Al --out-dir results
```
Expected: prints a JSON row; `grace_K_VRH` and `mp_K_VRH` both positive and within a physically reasonable margin (Al K≈75 GPa). If GRACE moduli are negative, the sign gate failed — fix `_ASE_STRESS_SIGN` and re-run Task 7's test first.

- [ ] **Step 3: Run all six materials (parallel) and aggregate**

Run each material (Al, Cu, Si, Fe, Ni, W) — these are independent processes and can be launched concurrently. Then aggregate into one markdown table `verification/results/SUMMARY.md` with columns: material, K_VRH (GRACE/MP/%err), G_VRH (GRACE/MP/%err), E, ν, A_U, C11/C12/C44. Report the **actual** numbers and % errors — no pass/fail massaging.

- [ ] **Step 4: Write `verification/README.md`**

Document how to run (env vars `GRACE_CACHE`, `MP_API_KEY`), the material set, what the table means, and a one-paragraph interpretation of GRACE-SMAX vs MP DFT agreement.

- [ ] **Step 5: Commit (script + README + results, NO secrets)**

```bash
cd /ptmp/hmai/pwa_elastic
git add verification/elastic_grace_vs_mp.py verification/README.md verification/results/SUMMARY.md verification/results/*.csv
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "test(elastic): GRACE-SMAX vs MP DFT verification harness + results"
```
**Never commit the MP API key.** It is only ever passed via the `MP_API_KEY` env var at run time.

---

### Task 10: Example notebook

**Files:**
- Create: `notebooks/elastic_constants.ipynb`

Follow the style of the existing physics notebooks (`notebooks/eos.ipynb`, `notebooks/surface_energy.ipynb`, `notebooks/foundation_model_example.ipynb`). Build the notebook programmatically with `nbformat` to guarantee valid JSON, then execute it end-to-end so the outputs are real.

- [ ] **Step 1: Generate the notebook**

Write a small builder script (or inline python) using `nbformat` that creates `notebooks/elastic_constants.ipynb` with these cells:
1. **Markdown title/intro:** "Elastic constants with `pyiron_workflow_atomistics`" — explains the MP stress-strain method and that any ASE engine works.
2. **Imports + structure:** `from ase.build import bulk`; build `Cu` fcc cubic.
3. **EMT engine cell** (fast, always runs):
   ```python
   from ase.calculators.emt import EMT
   from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
   from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants
   engine = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory="elastic_emt")
   wf = calculate_elastic_constants(structure=bulk("Cu","fcc",a=3.615,cubic=True), engine=engine, relax_initial=True)
   out = wf.run()
   d = out["elastic_constants"]
   ```
4. **Display cell:** pretty-print the full elastic tensor (`d["elastic_tensor_ieee"]`), compliance, and a small `pandas` table of `K_VRH, G_VRH, youngs_modulus, poisson_ratio, universal_anisotropy, mechanically_stable`.
5. **Markdown:** "Swapping in the GRACE-2L-SMAX foundation model" — explain `GRACE_CACHE` and `grace_fm`.
6. **GRACE engine cell**, guarded so the notebook still runs without the model:
   ```python
   import os
   os.environ.setdefault("GRACE_CACHE", "/ptmp/hmai/grace_cache")
   os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
   try:
       from tensorpotential.calculator import grace_fm
       gengine = ASEEngine(EngineInput=CalcInputStatic(), calculator=grace_fm("GRACE-2L-SMAX-large"), working_directory="elastic_grace")
       gout = calculate_elastic_constants(structure=bulk("Cu","fcc",a=3.615,cubic=True), engine=gengine, relax_initial=True).run()
       print(gout["elastic_constants"]["elastic_tensor_ieee"])
   except Exception as e:
       print("GRACE not available in this environment:", e)
   ```
7. **Markdown:** point to `verification/elastic_grace_vs_mp.py` for the full GRACE-vs-MP cross-check table.

- [ ] **Step 2: Execute the notebook end-to-end**

Run:
```bash
cd /ptmp/hmai/pwa_elastic
GRACE_CACHE=/ptmp/hmai/grace_cache TF_CPP_MIN_LOG_LEVEL=3 \
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/elastic_constants.ipynb
```
(If `jupyter`/`nbconvert` is missing, `.venv/bin/pip install nbconvert ipykernel` first.)
Expected: executes with no errors; EMT cell shows a positive-definite Cu tensor; GRACE cell shows the GRACE tensor (or the graceful "not available" message).

- [ ] **Step 3: Commit**

```bash
cd /ptmp/hmai/pwa_elastic
git add notebooks/elastic_constants.ipynb
git -c user.name=Han -c user.email=ligerzerof@gmail.com commit -q -m "docs(elastic): example notebook (EMT + GRACE-SMAX)"
```

---

## Self-Review notes
- **Spec coverage:** §2 method → Tasks 3,7; full constants list §2 → Task 6; unit/sign §3 → Tasks 1,7 (empirical gate); components §4 → Tasks 1-9; data flow §5 → Task 7 macro; error handling §6 → Task 6 (stability flag), Task 9 (missing MP skip — add try/except per material in aggregation); verification gates §8 → Tasks 7,8 (pytest) and Task 9 (GRACE vs MP).
- **Sign convention** is resolved by one constant `_ASE_STRESS_SIGN`, validated at Task 7. Task 5 uses synthetic stresses so it is sign-agnostic (pairing check only).
- **Type consistency:** node output names (`deformed_structures`, `strains`, `stresses`, `elastic_tensor`, `elastic_constants`) are used identically across Tasks 3-8.
