# Melting Point (Interface/Coexistence Method) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an engine-agnostic melting-point workflow (full interface/coexistence method) to `pyiron_workflow_atomistics`, runnable on the ASE engine (EMT, GRACE) and the wrapped `pyiron_workflow_lammps` `LammpsEngine` (`pair_style grace`).

**Architecture:** Pure analysis quantities go in the general `analysis/` package; the method lives in a new `physics/melting/` subpackage. Everything consumes the Engine Protocol via `calculate(structure, engine)` + `CalcInputMD`/`CalcInputMinimize`. Adaptive loops (bisection, convergence) are `@pwf.as_function_node` Python drivers calling `.node_function()` per step with `engine.with_working_directory(...)` isolation (Approach A, matching `physics/free_energy/quasiharmonic.py`).

**Tech Stack:** Python 3.11, `pyiron_workflow`, ASE 3.28, numpy 1.26.4, structuretoolkit (CNA + Voronoi), scikit-learn (KDE), `pyiron_workflow_lammps` + `pyiron_lammps` (LAMMPS engine).

## Global Constraints

- Engine-agnostic: method code imports only from `pyiron_workflow_atomistics.engine` (`Engine`, `calculate`, `CalcInputMD`, `CalcInputMinimize`) + `analysis`/`structure`. Never `import ase.md` / LAMMPS directly in method code.
- Nodes are `@pwf.as_function_node("out_name", ...)`; the top-level orchestrator is `@pwf.as_macro_node(...)`. Loop drivers are `@pwf.as_function_node` that call `<node>.node_function(...)` internally.
- Per-step isolation: `engine.with_working_directory(f"<tag>_{i:03d}")`. Per-step calc params: `dataclasses.replace(engine, EngineInput=<CalcInput...>)`.
- CNA via `structuretoolkit.analyse.get_adaptive_cna_descriptors(structure=s, mode=..., ovito_compatibility=False)` → lowercase keys `{'fcc','bcc','hcp','ico','others'}`. Voronoi via `structuretoolkit.analyse.get_voronoi_volumes(structure)`.
- NVE runs set `CalcInputMD.initial_temperature = 2*temperature` (half-velocity trick).
- Pressure = virial (`EngineOutput.stresses`, handle Voigt-6 **and** 3×3) + kinetic `N·kB·T/V`; report GPa (`eV/Å³ × 160.21766208`).
- Dev venv: `/ptmp/hmai/pwa_melting/.venv`. Run tests: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest`. Fast tests under `tests/unit/`, slow MD under `tests/integration/` (`@pytest.mark.slow`).
- Commit after every task. Ruff-clean (`ruff check`).

---

## File structure

| File | Responsibility |
|---|---|
| `pyiron_workflow_atomistics/analysis/trajectory.py` | `temperatures_from_trajectory`, `pressures_from_trajectory` (on `EngineOutput`) |
| `pyiron_workflow_atomistics/analysis/structure_descriptors.py` | `cna_fractions`, `analyse_reference_structure`, `classify_solid`, `voronoi_max_mean`, `holes_mask` |
| `pyiron_workflow_atomistics/analysis/__init__.py` | export the above |
| `pyiron_workflow_atomistics/physics/melting/inputs.py` | `MeltingInput` |
| `pyiron_workflow_atomistics/physics/melting/outputs.py` | `MeltingIterationRecord`, `MeltingResult` |
| `pyiron_workflow_atomistics/physics/melting/structures.py` | `create_coexistence_supercell`, `freeze_half`, `unfreeze`, `strain_cell_along_z` |
| `pyiron_workflow_atomistics/physics/melting/solid_fraction.py` | `solid_fraction_kde` |
| `pyiron_workflow_atomistics/physics/melting/fitting.py` | `ratio_selection`, `predict_melting_point` |
| `pyiron_workflow_atomistics/physics/melting/md_steps.py` | `with_calc_input`, `npt_relax_solid`, `build_solid_liquid_interface`, `strain_scan_nvt_nve` |
| `pyiron_workflow_atomistics/physics/melting/initial_guess.py` | `estimate_melting_temperature` |
| `pyiron_workflow_atomistics/physics/melting/coexistence.py` | `coexistence_iteration`, `refine_melting_point` |
| `pyiron_workflow_atomistics/physics/melting/study.py` | `calculate_melting_point` macro |
| `pyiron_workflow_atomistics/physics/melting/__init__.py` | public API |
| `scripts/meltingpoint.py` | notebook → runnable EMT/Al Step-1 demo |
| `/ptmp/hmai/pwl_dev/pyiron_workflow_lammps/lammps.py` | velocity-capture patch |

---

### Task 1: Trajectory quantities (temperature & pressure from EngineOutput)

**Files:**
- Create: `pyiron_workflow_atomistics/analysis/trajectory.py`
- Test: `tests/unit/analysis/test_trajectory.py`

**Interfaces:**
- Consumes: `EngineOutput` (`.structures: list[Atoms]` with momenta, `.stresses: list[np.ndarray]` Voigt-6 or 3×3).
- Produces:
  - `temperatures_from_trajectory(engine_output, last_n=20) -> float` (mean K over last `last_n` frames)
  - `pressures_from_trajectory(engine_output, last_n=20) -> float` (mean GPa, virial+kinetic)

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/analysis/test_trajectory.py
import numpy as np
from ase import Atoms, units
from pyiron_workflow_atomistics.engine.protocol import EngineOutput
from pyiron_workflow_atomistics.analysis.trajectory import (
    temperatures_from_trajectory,
    pressures_from_trajectory,
)

def _frame_at_temperature(T_target, n=64, a=10.0, seed=0):
    rng = np.random.RandomState(seed)
    atoms = Atoms(f"Ar{n}", positions=rng.rand(n, 3) * a, cell=[a, a, a], pbc=True)
    atoms.set_momenta(rng.standard_normal((n, 3)))
    atoms.set_momenta(atoms.get_momenta() * np.sqrt(T_target / atoms.get_temperature()))
    return atoms

def test_temperatures_from_trajectory_mean():
    frames = [_frame_at_temperature(300.0, seed=i) for i in range(5)]
    out = EngineOutput(final_structure=frames[-1], final_energy=0.0,
                       converged=True, structures=frames)
    T = temperatures_from_trajectory.node_function(out, last_n=5)
    assert abs(T - 300.0) < 1e-6

def test_pressures_from_trajectory_virial_plus_kinetic():
    n, a = 64, 10.0
    V = a ** 3
    frame = _frame_at_temperature(300.0, n=n, a=a, seed=1)
    p_vir = 0.01  # eV/Å^3
    svoigt = np.array([-p_vir, -p_vir, -p_vir, 0.0, 0.0, 0.0])
    out = EngineOutput(final_structure=frame, final_energy=0.0, converged=True,
                       structures=[frame], stresses=[svoigt])
    P = pressures_from_trajectory.node_function(out, last_n=1)
    p_kin = n * units.kB * 300.0 / V          # eV/Å^3
    expected = (p_vir + p_kin) * 160.21766208  # GPa
    assert abs(P - expected) < 1e-6

def test_pressures_accepts_full_3x3_stress():
    n, a = 64, 10.0
    frame = _frame_at_temperature(300.0, n=n, a=a, seed=2)
    p_vir = 0.02
    full = np.diag([-p_vir, -p_vir, -p_vir])
    out = EngineOutput(final_structure=frame, final_energy=0.0, converged=True,
                       structures=[frame], stresses=[full])
    P = pressures_from_trajectory.node_function(out, last_n=1)
    p_kin = n * units.kB * 300.0 / (a ** 3)
    assert abs(P - (p_vir + p_kin) * 160.21766208) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis/test_trajectory.py -v`
Expected: FAIL (ModuleNotFoundError: analysis.trajectory)

- [ ] **Step 3: Write minimal implementation**
```python
# pyiron_workflow_atomistics/analysis/trajectory.py
"""Derived quantities computed from an EngineOutput trajectory.

Engine-agnostic: works for any engine whose EngineOutput.structures carry
per-atom momenta (ASE engine; patched LAMMPS engine) and whose .stresses hold
per-frame virial stress (Voigt-6 or 3x3, in eV/Å^3).
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import units

_EV_PER_A3_TO_GPA = 160.21766208


def _virial_pressure_ev_per_a3(stress: np.ndarray) -> float:
    """Hydrostatic virial pressure P = -tr(sigma)/3 from Voigt-6 or 3x3 stress."""
    s = np.asarray(stress, dtype=float)
    if s.shape == (6,):
        trace = s[0] + s[1] + s[2]
    elif s.shape == (3, 3):
        trace = s[0, 0] + s[1, 1] + s[2, 2]
    else:
        raise ValueError(f"Unexpected stress shape {s.shape}; expected (6,) or (3, 3)")
    return -trace / 3.0


@pwf.as_function_node("temperature")
def temperatures_from_trajectory(engine_output, last_n: int = 20) -> float:
    """Mean kinetic temperature (K) over the last ``last_n`` trajectory frames."""
    frames = engine_output.structures
    if not frames:
        raise ValueError("engine_output.structures is empty; need an MD trajectory")
    window = frames[-last_n:]
    temperature = float(np.mean([f.get_temperature() for f in window]))
    return temperature


@pwf.as_function_node("pressure")
def pressures_from_trajectory(engine_output, last_n: int = 20) -> float:
    """Mean total pressure (GPa) over the last ``last_n`` frames: virial + kinetic."""
    frames = engine_output.structures
    stresses = engine_output.stresses
    if not frames or not stresses:
        raise ValueError("engine_output needs both .structures and .stresses")
    window_f = frames[-last_n:]
    window_s = stresses[-last_n:]
    pressures = []
    for frame, stress in zip(window_f, window_s):
        p_vir = _virial_pressure_ev_per_a3(stress)             # eV/Å^3
        p_kin = (len(frame) * units.kB * frame.get_temperature()
                 / frame.get_volume())                          # eV/Å^3
        pressures.append((p_vir + p_kin) * _EV_PER_A3_TO_GPA)   # GPa
    return float(np.mean(pressures))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis/test_trajectory.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/analysis/trajectory.py tests/unit/analysis/test_trajectory.py
git commit -m "feat(analysis): trajectory temperature & pressure quantities from EngineOutput"
```

---

### Task 2: CNA structure descriptors (fractions, reference analysis, classify)

**Files:**
- Create: `pyiron_workflow_atomistics/analysis/structure_descriptors.py`
- Test: `tests/unit/analysis/test_structure_descriptors.py`

**Interfaces:**
- Produces:
  - `cna_fractions(structure) -> dict[str, int]` (counts, lowercase keys)
  - `analyse_reference_structure(structure) -> tuple[str, int, float]` → `(key_max, n_atoms, distribution_half)`
  - `classify_solid(structure, key_max, distribution_half) -> bool`

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/analysis/test_structure_descriptors.py
import numpy as np
from ase.build import bulk
from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    cna_fractions, analyse_reference_structure, classify_solid,
)

def _fcc_al():
    return bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 4))

def test_cna_fractions_fcc_dominant():
    counts = cna_fractions.node_function(_fcc_al())
    assert counts["fcc"] / sum(counts.values()) > 0.95

def test_analyse_reference_structure_fcc():
    key_max, n_atoms, half = analyse_reference_structure.node_function(_fcc_al())
    assert key_max == "fcc"
    assert n_atoms == 256
    assert abs(half - 0.5) < 0.05

def test_classify_solid_true_for_crystal():
    s = _fcc_al()
    key_max, _, half = analyse_reference_structure.node_function(s)
    assert classify_solid.node_function(s, key_max, half) is True

def test_classify_solid_false_for_disordered():
    s = _fcc_al()
    key_max, _, half = analyse_reference_structure.node_function(s)
    rng = np.random.RandomState(0)
    s.set_positions(s.get_positions() + rng.standard_normal((len(s), 3)) * 1.5)
    assert classify_solid.node_function(s, key_max, half) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis/test_structure_descriptors.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# pyiron_workflow_atomistics/analysis/structure_descriptors.py
"""Common-neighbour-analysis and Voronoi structure descriptors (general)."""

from __future__ import annotations

import operator

import numpy as np
import pyiron_workflow as pwf
from structuretoolkit.analyse import (
    get_adaptive_cna_descriptors,
    get_voronoi_volumes,
)


@pwf.as_function_node("counts")
def cna_fractions(structure) -> dict:
    """Adaptive CNA counts, lowercase keys {'fcc','bcc','hcp','ico','others'}."""
    counts = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=False
    )
    return dict(counts)


@pwf.as_function_node("key_max", "n_atoms", "distribution_half")
def analyse_reference_structure(structure):
    """Dominant CNA phase, atom count, and half its population fraction.

    ``distribution_half`` is the solid/liquid threshold: a structure counts as
    *solid* while the dominant-phase fraction stays above this value.
    """
    counts = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=False
    )
    key_max = max(counts.items(), key=operator.itemgetter(1))[0]
    n_atoms = len(structure)
    distribution_half = (counts[key_max] / n_atoms) / 2.0
    return key_max, n_atoms, distribution_half


@pwf.as_function_node("is_solid")
def classify_solid(structure, key_max: str, distribution_half: float) -> bool:
    """True if the dominant-phase fraction exceeds ``distribution_half``."""
    counts = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=False
    )
    fraction = counts.get(key_max, 0) / len(structure)
    return bool(fraction > distribution_half)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis/test_structure_descriptors.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/analysis/structure_descriptors.py tests/unit/analysis/test_structure_descriptors.py
git commit -m "feat(analysis): CNA structure descriptors (fractions, reference, classify)"
```

---

### Task 3: Voronoi void detection

**Files:**
- Modify: `pyiron_workflow_atomistics/analysis/structure_descriptors.py`
- Test: `tests/unit/analysis/test_structure_descriptors.py` (append)

**Interfaces:**
- Produces:
  - `voronoi_max_mean(structure) -> tuple[float, float]` → `(max_volume, mean_volume)`
  - `holes_mask(max_volumes, mean_volumes, factor=2.0) -> list[bool]` (True = no hole)

- [ ] **Step 1: Write the failing test (append)**
```python
# append to tests/unit/analysis/test_structure_descriptors.py
from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    voronoi_max_mean, holes_mask,
)

def test_voronoi_max_mean_uniform_fcc():
    vmax, vmean = voronoi_max_mean.node_function(_fcc_al())
    assert vmax / vmean < 1.2  # uniform crystal: max ~ mean

def test_holes_mask_flags_large_void():
    # one entry has max >> 2*mean -> hole (False), others fine (True)
    keep = holes_mask.node_function([1.0, 1.0, 5.0], [1.0, 1.0, 1.0], factor=2.0)
    assert keep == [True, True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis/test_structure_descriptors.py -k voronoi -v`
Expected: FAIL (ImportError: voronoi_max_mean)

- [ ] **Step 3: Add implementation (append to structure_descriptors.py)**
```python
@pwf.as_function_node("max_volume", "mean_volume")
def voronoi_max_mean(structure):
    """Max and mean per-atom Voronoi volume (Å^3)."""
    volumes = get_voronoi_volumes(structure)
    return float(np.max(volumes)), float(np.mean(volumes))


@pwf.as_function_node("keep_mask")
def holes_mask(max_volumes, mean_volumes, factor: float = 2.0) -> list:
    """Per-entry True where no cavity: max_volume < factor * mean(mean_volumes)."""
    threshold = factor * float(np.mean(mean_volumes))
    return [bool(m < threshold) for m in max_volumes]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/analysis/test_structure_descriptors.py -v`
Expected: all PASS

- [ ] **Step 5: Update analysis/__init__.py and commit**

Add to `pyiron_workflow_atomistics/analysis/__init__.py` (preserve existing lines):
```python
from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    analyse_reference_structure,
    classify_solid,
    cna_fractions,
    holes_mask,
    voronoi_max_mean,
)
from pyiron_workflow_atomistics.analysis.trajectory import (
    pressures_from_trajectory,
    temperatures_from_trajectory,
)
```
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/analysis/structure_descriptors.py pyiron_workflow_atomistics/analysis/__init__.py tests/unit/analysis/test_structure_descriptors.py
git commit -m "feat(analysis): Voronoi void detection + export analysis quantities"
```

---

### Task 4: Melting input/output dataclasses

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/__init__.py` (empty placeholder for now)
- Create: `pyiron_workflow_atomistics/physics/melting/inputs.py`
- Create: `pyiron_workflow_atomistics/physics/melting/outputs.py`
- Test: `tests/unit/physics/melting/test_dataclasses.py`

**Interfaces:**
- Produces: `MeltingInput`, `MeltingIterationRecord`, `MeltingResult` (plain `@dataclass`; `MeltingResult.to_dict()` excludes nothing heavy here).

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/physics/melting/test_dataclasses.py
from pyiron_workflow_atomistics.physics.melting.inputs import MeltingInput
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingIterationRecord, MeltingResult,
)

def test_melting_input_defaults():
    mi = MeltingInput(element="Al")
    assert mi.n_atoms == 8000
    assert mi.timestep_lst == [2.0, 2.0, 1.0]
    assert mi.convergence_goal == 1.0

def test_melting_result_to_dict():
    rec = MeltingIterationRecord(temperature_in=900.0, temperature_next=905.0,
                                 strains=[1.0], ratios=[0.5], pressures=[0.0],
                                 temperatures=[905.0], converged=True)
    res = MeltingResult(melting_temperature=905.0, converged=True, n_iterations=1,
                        element="Al", crystalstructure="fcc", n_atoms=4000,
                        initial_guess=914.0, iterations=[rec], report={})
    d = res.to_dict()
    assert d["melting_temperature"] == 905.0
    assert len(d["iterations"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_dataclasses.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/__init__.py
"""Melting-point via the interface/coexistence method."""
```
```python
# pyiron_workflow_atomistics/physics/melting/inputs.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeltingInput:
    """Parameters for an interface-method melting-point run (notebook defaults)."""

    element: str
    crystalstructure: str | None = None      # default: ASE reference state
    a: float | None = None
    n_atoms: int = 8000                       # solid cell targets n_atoms/2
    temperature_left: float = 0.0
    temperature_right: float = 1000.0
    convergence_goal: float = 1.0             # K
    timestep_lst: list[float] = field(default_factory=lambda: [2.0, 2.0, 1.0])
    fit_range_lst: list[float] = field(default_factory=lambda: [0.05, 0.01, 0.01])
    nve_steps_lst: list[int] = field(default_factory=lambda: [25000, 20000, 50000])
    nvt_run_steps: int = 10000
    npt_run_steps: int = 50000
    strain_run_steps: int = 1000
    n_strain_points: int = 21
    ratio_boundary: float = 0.25
    boundary_value: float = 0.25
    delta_t_melt: float = 1000.0              # superheat for interface build
    seed: int | None = None
```
```python
# pyiron_workflow_atomistics/physics/melting/outputs.py
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class MeltingIterationRecord:
    temperature_in: float
    temperature_next: float
    strains: list[float]
    ratios: list[float]
    pressures: list[float]
    temperatures: list[float]
    converged: bool


@dataclass
class MeltingResult:
    melting_temperature: float
    converged: bool
    n_iterations: int
    element: str
    crystalstructure: str
    n_atoms: int
    initial_guess: float
    iterations: list[MeltingIterationRecord] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_dataclasses.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/__init__.py pyiron_workflow_atomistics/physics/melting/inputs.py pyiron_workflow_atomistics/physics/melting/outputs.py tests/unit/physics/melting/test_dataclasses.py
git commit -m "feat(melting): MeltingInput/MeltingResult dataclasses"
```

---

### Task 5: Coexistence structure operations

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/structures.py`
- Test: `tests/unit/physics/melting/test_structures.py`

**Interfaces:**
- Produces:
  - `create_coexistence_supercell(element, crystalstructure=None, a=None, n_atoms=8000) -> Atoms`
  - `freeze_half(structure, axis=2, fraction=0.5) -> Atoms` (FixAtoms on lower half)
  - `unfreeze(structure) -> Atoms`
  - `strain_cell_along_z(structure, strain) -> Atoms`

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/physics/melting/test_structures.py
import numpy as np
from ase.constraints import FixAtoms
from pyiron_workflow_atomistics.physics.melting.structures import (
    create_coexistence_supercell, freeze_half, unfreeze, strain_cell_along_z,
)

def test_supercell_targets_half_n_atoms():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=8000)
    assert len(s) == 4000  # fcc cubic = 4 atoms; 4*10^3 = 4000 ~ 8000/2

def test_freeze_half_fixes_lower_half():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=2000)
    frozen = freeze_half.node_function(s, axis=2, fraction=0.5)
    cons = [c for c in frozen.constraints if isinstance(c, FixAtoms)]
    assert len(cons) == 1
    fixed = set(cons[0].get_indices())
    zsc = frozen.get_scaled_positions()[:, 2]
    expected = set(np.where(zsc < 0.5)[0])
    assert fixed == expected

def test_unfreeze_clears_constraints():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=2000)
    assert len(unfreeze.node_function(freeze_half.node_function(s)).constraints) == 0

def test_strain_scales_only_c():
    s = create_coexistence_supercell.node_function("Al", "fcc", a=4.05, n_atoms=2000)
    c0 = s.cell[2, 2]
    strained = strain_cell_along_z.node_function(s, 1.05)
    assert abs(strained.cell[2, 2] - 1.05 * c0) < 1e-8
    assert abs(strained.cell[0, 0] - s.cell[0, 0]) < 1e-8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_structures.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/structures.py
from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase.build import bulk
from ase.constraints import FixAtoms


@pwf.as_function_node("structure")
def create_coexistence_supercell(element: str, crystalstructure: str | None = None,
                                 a: float | None = None, n_atoms: int = 8000):
    """Cubic bulk repeated i×i×i so the atom count is closest to ``n_atoms/2``."""
    base = bulk(element, crystalstructure, a=a, cubic=True)
    target = n_atoms / 2.0
    reps = range(2, 30)
    cells = [base.repeat((i, i, i)) for i in reps]
    structure = cells[int(np.argmin([abs(len(c) - target) for c in cells]))]
    return structure


@pwf.as_function_node("structure")
def freeze_half(structure, axis: int = 2, fraction: float = 0.5):
    """Fix atoms whose scaled coordinate along ``axis`` is below ``fraction``."""
    s = structure.copy()
    scaled = s.get_scaled_positions()[:, axis]
    s.set_constraint(FixAtoms(indices=np.where(scaled < fraction)[0]))
    return s


@pwf.as_function_node("structure")
def unfreeze(structure):
    """Remove all constraints."""
    s = structure.copy()
    s.set_constraint()
    return s


@pwf.as_function_node("structure")
def strain_cell_along_z(structure, strain: float):
    """Scale cell vector c by ``strain`` (scale_atoms=True)."""
    s = structure.copy()
    cell = s.cell.copy()
    cell[2, 2] *= strain
    s.set_cell(cell, scale_atoms=True)
    return s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_structures.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/structures.py tests/unit/physics/melting/test_structures.py
git commit -m "feat(melting): coexistence supercell, freeze/unfreeze, z-strain"
```

---

### Task 6: Solid-fraction (CNA + KDE)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/solid_fraction.py`
- Test: `tests/unit/physics/melting/test_solid_fraction.py`

**Interfaces:**
- Produces: `solid_fraction_kde(structure, crystalstructure, threshold=0.1) -> float` (0..1)

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/physics/melting/test_solid_fraction.py
import numpy as np
from ase.build import bulk
from pyiron_workflow_atomistics.physics.melting.solid_fraction import solid_fraction_kde

def _half_solid_half_liquid():
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))  # long in z
    z = s.get_scaled_positions()[:, 2]
    upper = np.where(z >= 0.5)[0]
    rng = np.random.RandomState(0)
    pos = s.get_positions()
    pos[upper] += rng.standard_normal((len(upper), 3)) * 1.6  # melt upper half
    s.set_positions(pos)
    return s

def test_full_solid_ratio_near_one():
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))
    assert solid_fraction_kde.node_function(s, "fcc") > 0.9

def test_half_solid_ratio_near_half():
    frac = solid_fraction_kde.node_function(_half_solid_half_liquid(), "fcc")
    assert 0.3 < frac < 0.7

def test_all_liquid_ratio_zero():
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))
    rng = np.random.RandomState(1)
    s.set_positions(s.get_positions() + rng.standard_normal((len(s), 3)) * 2.0)
    assert solid_fraction_kde.node_function(s, "fcc") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_solid_fraction.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/solid_fraction.py
from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from sklearn.neighbors import KernelDensity
from structuretoolkit.analyse import get_adaptive_cna_descriptors


@pwf.as_function_node("solid_fraction")
def solid_fraction_kde(structure, crystalstructure: str, threshold: float = 0.1) -> float:
    """Fraction of the z-extent occupied by the crystalline phase.

    Per-atom CNA labels the target phase; a 1-D KDE of those atoms' z-positions
    gives the solid slab width relative to the cell. Mirrors the interface
    method's ``plot_solid_liquid_ratio`` (minus plotting).
    """
    target = crystalstructure.lower()
    labels = np.array(
        get_adaptive_cna_descriptors(structure=structure, mode="str",
                                     ovito_compatibility=False)
    )
    z = structure.get_positions()[:, 2]
    mask = labels == target
    if mask.sum() <= 0.05 * len(structure):
        return 0.0
    bandwidth = (structure.get_volume() / len(structure)) ** (1.0 / 3.0)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(z[mask].reshape(-1, 1))
    grid = np.linspace(z.min(), z.max(), 1000)
    g = np.exp(kde.score_samples(grid.reshape(-1, 1)))
    g = g / g.max()
    above = grid[g > threshold]
    below = grid[g < threshold]
    span = grid.max() - grid.min()
    ratio_above = (above.max() - above.min()) / span if len(above) else 1.0
    ratio_below = 1.0 - (below.max() - below.min()) / span if len(below) else 0.0
    if ratio_below == 0.0:
        ratio = ratio_above
    elif ratio_above == 1.0:
        ratio = ratio_below
    else:
        ratio = min(ratio_below, ratio_above)
    return float(ratio)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_solid_fraction.py -v`
Expected: 3 PASS (if `test_half_solid_ratio_near_half` is borderline, widen the rattle to 2.0 — the boundary is physical, not a code bug)

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/solid_fraction.py tests/unit/physics/melting/test_solid_fraction.py
git commit -m "feat(melting): KDE solid-fraction from per-atom CNA"
```

---

### Task 7: Coexistence fitting (ratio selection + T(P=0) extrapolation)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/fitting.py`
- Test: `tests/unit/physics/melting/test_fitting.py`

**Interfaces:**
- Produces:
  - `ratio_selection(strains, ratios, pressures, temperatures, ratio_boundary=0.25) -> tuple[list, list, list, list, int]` → `(sel_strains, sel_ratios, sel_pressures, sel_temperatures, sl_flag)`
  - `predict_melting_point(strains, pressures, temperatures, boundary_value=0.25) -> tuple[float, float, float, float]` → `(t_next, t_mean, t_left, t_right)`

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/physics/melting/test_fitting.py
from pyiron_workflow_atomistics.physics.melting.fitting import (
    ratio_selection, predict_melting_point,
)

def test_ratio_selection_picks_coexistence_window():
    strains = [0.96, 0.98, 1.00, 1.02, 1.04]
    ratios = [0.05, 0.45, 0.50, 0.55, 0.95]   # middle three near 0.5
    pressures = [2.0, 1.0, 0.0, -1.0, -2.0]
    temps = [900, 905, 910, 915, 920]
    ss, rr, pp, tt, flag = ratio_selection.node_function(
        strains, ratios, pressures, temps, ratio_boundary=0.25)
    assert list(ss) == [0.98, 1.00, 1.02]
    assert flag in (1, -1)

def test_predict_melting_point_extrapolates_to_zero_pressure():
    # T = 910 + 250*P  => T(P=0) = 910; P linear in strain
    strains = [0.98, 1.00, 1.02]
    pressures = [1.0, 0.0, -1.0]
    temps = [910 + 250 * p for p in pressures]
    t_next, t_mean, t_left, t_right = predict_melting_point.node_function(
        strains, pressures, temps, boundary_value=0.25)
    assert abs(t_next - 910.0) < 1e-6
    assert t_left < t_mean < t_right
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_fitting.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/fitting.py
from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf


@pwf.as_function_node("strains", "ratios", "pressures", "temperatures", "sl_flag")
def ratio_selection(strains, ratios, pressures, temperatures, ratio_boundary: float = 0.25):
    """Keep the longest contiguous strain window with ratio in 0.5±boundary.

    ``sl_flag`` is +1 if the selected window is mostly solid (>0.5), else -1.
    """
    groups, current = [], []
    for r in ratios:
        if (0.5 - ratio_boundary) < r < (0.5 + ratio_boundary):
            current.append(r)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if not groups:
        flag = 1 if np.mean(ratios) > 0.5 else -1
        return [], [], [], [], flag
    best = groups[int(np.argmax([len(g) for g in groups]))]
    keep = [r in best for r in ratios]
    sel_r = np.array(ratios)[keep]
    flag = 1 if np.mean(sel_r) > 0.5 else -1
    return (
        np.array(strains)[keep].tolist(),
        sel_r.tolist(),
        np.array(pressures)[keep].tolist(),
        np.array(temperatures)[keep].tolist(),
        flag,
    )


@pwf.as_function_node("t_next", "t_mean", "t_left", "t_right")
def predict_melting_point(strains, pressures, temperatures, boundary_value: float = 0.25):
    """Extrapolate temperature to zero pressure; bracket via boundary_value."""
    fit_temp_from_press = np.poly1d(np.polyfit(pressures, temperatures, 1))
    t_next = float(fit_temp_from_press(0.0))
    t_min, t_max = float(np.min(temperatures)), float(np.max(temperatures))
    span = t_max - t_min
    t_mean = t_min + span * 0.5
    t_left = t_min + span * (0.5 - boundary_value)
    t_right = t_min + span * (0.5 + boundary_value)
    return t_next, t_mean, t_left, t_right
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/physics/melting/test_fitting.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/fitting.py tests/unit/physics/melting/test_fitting.py
git commit -m "feat(melting): ratio selection + T(P=0) melting-point extrapolation"
```

---

### Task 8: ASE NPT robustness (verify/patch compressibility wiring)

**Files:**
- Modify (if needed): `pyiron_workflow_atomistics/engine/ase.py` (NPT-Berendsen branch)
- Test: `tests/integration/test_ase_npt_metal.py`

**Interfaces:**
- Consumes: `ASEEngine`, `CalcInputMD(mode="NPT", thermostat="berendsen", compressibility=...)`.
- Produces: a confirmed-working NPT path that respects `CalcInputMD.compressibility`.

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_ase_npt_metal.py
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD, calculate

@pytest.mark.slow
def test_npt_berendsen_metal_runs(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    md = CalcInputMD(mode="NPT", thermostat="berendsen", temperature=300.0,
                     pressure=0.0, n_ionic_steps=20, n_print=5, time_step=2.0,
                     thermostat_time_constant=100.0, pressure_damping_timescale=1000.0,
                     compressibility=1e-6, seed=1, initial_temperature=600.0)
    eng = ASEEngine(EngineInput=md, calculator=EMT(), working_directory=str(tmp_path))
    out = calculate.node_function(atoms, engine=eng)
    assert out.converged is True
    assert out.structures and out.structures[-1].get_temperature() > 0
```

- [ ] **Step 2: Run test**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_ase_npt_metal.py -v -m slow`
Expected: PASS, OR fail with a `NPTBerendsen` compressibility/argument error.

- [ ] **Step 3: If it failed, patch `engine/ase.py`**

In the `md_input.mode == "NPT"` / `thermostat == "berendsen"` branch, ensure compressibility is forwarded:
```python
dyn = NPTBerendsen(
    atoms, dt, temperature_K=T, pressure_au=P_bar,
    taut=ttime, taup=taup,
    compressibility_au=md_input.compressibility / (1.0 / units.bar),
)
```
(If the running ASE `NPTBerendsen` signature uses `compressibility` rather than `compressibility_au`, use that name; confirm via `python -c "import ase.md.nptberendsen, inspect; print(inspect.signature(ase.md.nptberendsen.NPTBerendsen.__init__))"`. Convert `CalcInputMD.compressibility` (bar⁻¹) to ASE units accordingly. If already passing, skip this step.)

- [ ] **Step 4: Re-run to verify pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_ase_npt_metal.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add tests/integration/test_ase_npt_metal.py pyiron_workflow_atomistics/engine/ase.py
git commit -m "test(engine): NPT-Berendsen metal smoke + honour CalcInputMD.compressibility"
```

---

### Task 9: MD step drivers (engine-agnostic)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/md_steps.py`
- Test: `tests/integration/test_melting_md_steps.py`

**Interfaces:**
- Consumes: `calculate`, `CalcInputMD` (engine), `strain_cell_along_z`, `freeze_half`, `unfreeze`, `temperatures_from_trajectory`, `pressures_from_trajectory`, `solid_fraction_kde`, `voronoi_max_mean`.
- Produces:
  - `with_calc_input(engine, calc_input) -> Engine`
  - `npt_relax_solid(structure, engine, temperature, n_steps=10000, timestep=2.0, seed=None, subdir="npt_solid") -> tuple[Atoms, EngineOutput]`
  - `build_solid_liquid_interface(structure, engine, t_solid, t_liquid, n_steps=10000, timestep=2.0, seed=None, subdir="interface") -> Atoms`
  - `strain_scan_nvt_nve(structure, engine, temperature, strains, crystalstructure, nvt_steps=10000, nve_steps=20000, timestep=2.0, seed=None, last_n=20, subdir="strain") -> list[dict]` — each dict: `{strain, mean_T, mean_P, solid_fraction, voronoi_max, voronoi_mean}`

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_melting_md_steps.py
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting.md_steps import (
    npt_relax_solid, build_solid_liquid_interface, strain_scan_nvt_nve,
)

def _engine(tmp_path):
    return ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(),
                     working_directory=str(tmp_path))

@pytest.mark.slow
def test_npt_relax_solid_returns_structure(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    struct, out = npt_relax_solid.node_function(
        s, _engine(tmp_path), temperature=300.0, n_steps=20, timestep=2.0, seed=1)
    assert len(struct) == len(s)
    assert out.converged is True

@pytest.mark.slow
def test_strain_scan_returns_records(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 6))
    recs = strain_scan_nvt_nve.node_function(
        s, _engine(tmp_path), temperature=300.0, strains=[0.99, 1.01],
        crystalstructure="fcc", nvt_steps=20, nve_steps=20, timestep=2.0, seed=1)
    assert len(recs) == 2
    for r in recs:
        assert set(r) >= {"strain", "mean_T", "mean_P", "solid_fraction",
                          "voronoi_max", "voronoi_mean"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_md_steps.py -v -m slow`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/md_steps.py
from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf
from pyiron_workflow_atomistics.engine import CalcInputMD, calculate
from pyiron_workflow_atomistics.analysis.trajectory import (
    pressures_from_trajectory, temperatures_from_trajectory,
)
from pyiron_workflow_atomistics.physics.melting.solid_fraction import solid_fraction_kde
from pyiron_workflow_atomistics.analysis.structure_descriptors import voronoi_max_mean
from pyiron_workflow_atomistics.physics.melting.structures import (
    freeze_half, strain_cell_along_z, unfreeze,
)


def _engine_with(engine, calc_input, subdir):
    return replace(engine, EngineInput=calc_input).with_working_directory(subdir)


@pwf.as_function_node("engine_out")
def with_calc_input(engine, calc_input):
    return replace(engine, EngineInput=calc_input)


@pwf.as_function_node("relaxed_structure", "engine_output")
def npt_relax_solid(structure, engine, temperature, n_steps=10000, timestep=2.0,
                    seed=None, subdir="npt_solid"):
    md = CalcInputMD(mode="NPT", thermostat="berendsen", temperature=temperature,
                     pressure=0.0, n_ionic_steps=n_steps,
                     n_print=max(1, n_steps // 100), time_step=timestep,
                     initial_temperature=2.0 * temperature, seed=seed,
                     compressibility=1e-6)
    out = calculate.node_function(structure, engine=_engine_with(engine, md, subdir))
    return out.final_structure, out


@pwf.as_function_node("interface_structure")
def build_solid_liquid_interface(structure, engine, t_solid, t_liquid,
                                 n_steps=10000, timestep=2.0, seed=None,
                                 subdir="interface"):
    """Freeze lower half, melt upper half at t_liquid (NVT), recool to t_solid."""
    frozen = freeze_half.node_function(structure)
    melt_md = CalcInputMD(mode="NVT", thermostat="langevin", temperature=t_liquid,
                          n_ionic_steps=n_steps, n_print=max(1, n_steps // 100),
                          time_step=timestep, initial_temperature=2.0 * t_liquid, seed=seed)
    melted = calculate.node_function(
        frozen, engine=_engine_with(engine, melt_md, f"{subdir}_melt")).final_structure
    cool_md = CalcInputMD(mode="NVT", thermostat="langevin", temperature=t_solid,
                          n_ionic_steps=n_steps, n_print=max(1, n_steps // 100),
                          time_step=timestep, initial_temperature=2.0 * t_solid, seed=seed)
    cooled = calculate.node_function(
        melted, engine=_engine_with(engine, cool_md, f"{subdir}_cool")).final_structure
    return unfreeze.node_function(cooled)


@pwf.as_function_node("records")
def strain_scan_nvt_nve(structure, engine, temperature, strains, crystalstructure,
                        nvt_steps=10000, nve_steps=20000, timestep=2.0, seed=None,
                        last_n=20, subdir="strain"):
    records = []
    for i, strain in enumerate(strains):
        strained = strain_cell_along_z.node_function(structure, strain)
        nvt_md = CalcInputMD(mode="NVT", thermostat="langevin", temperature=temperature,
                             n_ionic_steps=nvt_steps, n_print=max(1, nvt_steps // 100),
                             time_step=timestep, initial_temperature=2.0 * temperature,
                             seed=seed)
        equil = calculate.node_function(
            strained, engine=_engine_with(engine, nvt_md, f"{subdir}_nvt_{i:03d}")
        ).final_structure
        nve_md = CalcInputMD(mode="NVE", temperature=temperature,
                             n_ionic_steps=nve_steps, n_print=max(1, nve_steps // 100),
                             time_step=timestep, initial_temperature=2.0 * temperature,
                             seed=seed)
        nve_out = calculate.node_function(
            equil, engine=_engine_with(engine, nve_md, f"{subdir}_nve_{i:03d}"))
        vmax, vmean = voronoi_max_mean.node_function(nve_out.final_structure)
        records.append({
            "strain": strain,
            "mean_T": temperatures_from_trajectory.node_function(nve_out, last_n=last_n),
            "mean_P": pressures_from_trajectory.node_function(nve_out, last_n=last_n),
            "solid_fraction": solid_fraction_kde.node_function(
                nve_out.final_structure, crystalstructure),
            "voronoi_max": vmax,
            "voronoi_mean": vmean,
        })
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_md_steps.py -v -m slow`
Expected: 2 PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/md_steps.py tests/integration/test_melting_md_steps.py
git commit -m "feat(melting): engine-agnostic MD step drivers (npt solid, interface, strain scan)"
```

---

### Task 10: Step 1 — initial melting-temperature guess (bisection)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/initial_guess.py`
- Test: `tests/integration/test_melting_initial_guess.py`

**Interfaces:**
- Consumes: `calculate`, `CalcInputMD`, `cna_fractions`.
- Produces: `estimate_melting_temperature(structure, engine, key_max, distribution_half, crystalstructure, temperature_left=0.0, temperature_right=1000.0, strain_run_steps=1000, timestep=2.0, seed=None, t_step_min=10.0, subdir="guess") -> tuple[int, Atoms]` → `(t_guess, structure_at_guess)`

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_melting_initial_guess.py
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.analysis.structure_descriptors import analyse_reference_structure
from pyiron_workflow_atomistics.physics.melting.initial_guess import estimate_melting_temperature

@pytest.mark.slow
def test_initial_guess_brackets_a_temperature(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    key_max, _, half = analyse_reference_structure.node_function(s)
    eng = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(),
                    working_directory=str(tmp_path))
    t_guess, struct = estimate_melting_temperature.node_function(
        s, eng, key_max=key_max, distribution_half=half, crystalstructure="fcc",
        temperature_left=0.0, temperature_right=1400.0, strain_run_steps=40,
        timestep=2.0, seed=1, t_step_min=200.0)  # coarse for speed
    assert 0.0 <= t_guess <= 1400.0
    assert len(struct) == len(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_initial_guess.py -v -m slow`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/initial_guess.py
from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf
from pyiron_workflow_atomistics.engine import CalcInputMD, calculate
from pyiron_workflow_atomistics.analysis.structure_descriptors import cna_fractions


def _fraction(structure, key_max):
    counts = cna_fractions.node_function(structure)
    return counts.get(key_max, 0) / len(structure)


@pwf.as_function_node("t_guess", "structure")
def estimate_melting_temperature(structure, engine, key_max, distribution_half,
                                 crystalstructure, temperature_left=0.0,
                                 temperature_right=1000.0, strain_run_steps=1000,
                                 timestep=2.0, seed=None, t_step_min=10.0,
                                 subdir="guess"):
    """Bisection: heat the bulk solid in NPT; raise T while it stays crystalline.

    Port of ``get_initial_melting_temperature_guess`` + ``next_step_funct``.
    """
    def heated(temperature):
        md = CalcInputMD(mode="NPT", thermostat="berendsen", temperature=temperature,
                         pressure=0.0, n_ionic_steps=strain_run_steps,
                         n_print=max(1, strain_run_steps // 10), time_step=timestep,
                         initial_temperature=2.0 * temperature, seed=seed,
                         compressibility=1e-6)
        tag = f"{subdir}_{int(round(temperature))}"
        eng = replace(engine, EngineInput=md).with_working_directory(tag)
        return calculate.node_function(structure, engine=eng).final_structure

    t_left, t_right = temperature_left, temperature_right
    struct_left = structure
    struct_right = heated(t_right)
    step = t_right - t_left
    while step > t_step_min:
        f_left = _fraction(struct_left, key_max)
        f_right = _fraction(struct_right, key_max)
        diff = t_right - t_left
        if f_left > distribution_half and f_right > distribution_half:
            struct_left, t_left = struct_right.copy(), t_right
            t_right += diff
            struct_right = heated(t_right)
        elif f_left > distribution_half >= f_right:
            diff /= 2.0
            t_left += diff
            struct_left = heated(t_left)
        else:  # both molten
            diff /= 2.0
            t_right, struct_right = t_left, struct_left.copy()
            t_left -= diff
            struct_left = heated(t_left)
        step = t_right - t_left
    return int(round(t_left)), struct_left
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_initial_guess.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/initial_guess.py tests/integration/test_melting_initial_guess.py
git commit -m "feat(melting): Step 1 initial melting-temperature bisection"
```

---

### Task 11: Step 2 — coexistence iteration + convergence loop

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/coexistence.py`
- Test: `tests/integration/test_melting_coexistence.py`

**Interfaces:**
- Consumes: `npt_relax_solid`, `build_solid_liquid_interface`, `strain_scan_nvt_nve`, `ratio_selection`, `predict_melting_point`, `holes_mask`, `MeltingIterationRecord`.
- Produces:
  - `coexistence_iteration(structure, engine, temperature, crystalstructure, fit_range, n_strain_points, nvt_steps, nve_steps, npt_steps, timestep, delta_t_melt, ratio_boundary, boundary_value, seed, subdir) -> MeltingIterationRecord`
  - `refine_melting_point(structure, engine, t_guess, melting_input, crystalstructure) -> MeltingResult`

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_melting_coexistence.py
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting.coexistence import coexistence_iteration

@pytest.mark.slow
def test_one_coexistence_iteration_runs(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 6))
    eng = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(),
                    working_directory=str(tmp_path))
    rec = coexistence_iteration.node_function(
        s, eng, temperature=900.0, crystalstructure="fcc", fit_range=0.05,
        n_strain_points=5, nvt_steps=20, nve_steps=20, npt_steps=20, timestep=2.0,
        delta_t_melt=1000.0, ratio_boundary=0.4, boundary_value=0.25, seed=1,
        subdir="iter0")
    assert rec.temperature_in == 900.0
    assert isinstance(rec.temperature_next, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_coexistence.py -v -m slow`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/coexistence.py
from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from pyiron_workflow_atomistics.analysis.structure_descriptors import holes_mask
from pyiron_workflow_atomistics.physics.melting.fitting import (
    predict_melting_point, ratio_selection,
)
from pyiron_workflow_atomistics.physics.melting.md_steps import (
    build_solid_liquid_interface, npt_relax_solid, strain_scan_nvt_nve,
)
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingIterationRecord, MeltingResult,
)


def _strain_grid(center, fit_range, n_points):
    return [round(float(s), 4)
            for s in np.linspace(center - fit_range, center + fit_range, n_points)]


@pwf.as_function_node("record")
def coexistence_iteration(structure, engine, temperature, crystalstructure,
                          fit_range=0.05, n_strain_points=21, nvt_steps=10000,
                          nve_steps=20000, npt_steps=50000, timestep=2.0,
                          delta_t_melt=1000.0, ratio_boundary=0.25,
                          boundary_value=0.25, seed=None, subdir="iter"):
    """One interface-method temperature iteration → next-T estimate."""
    solid, _ = npt_relax_solid.node_function(
        structure, engine, temperature=temperature, n_steps=npt_steps,
        timestep=timestep, seed=seed, subdir=f"{subdir}_npt")
    interface = build_solid_liquid_interface.node_function(
        solid, engine, t_solid=temperature, t_liquid=temperature + delta_t_melt,
        n_steps=npt_steps, timestep=timestep, seed=seed, subdir=f"{subdir}_iface")
    strains = _strain_grid(1.0, fit_range, n_strain_points)
    records = strain_scan_nvt_nve.node_function(
        interface, engine, temperature=temperature, strains=strains,
        crystalstructure=crystalstructure, nvt_steps=nvt_steps, nve_steps=nve_steps,
        timestep=timestep, seed=seed, subdir=f"{subdir}_strain")
    ratios = [r["solid_fraction"] for r in records]
    pressures = [r["mean_P"] for r in records]
    temps = [r["mean_T"] for r in records]
    sel_s, sel_r, sel_p, sel_t, flag = ratio_selection.node_function(
        strains, ratios, pressures, temps, ratio_boundary=ratio_boundary)
    if len(sel_s) > 2:
        sel_index = [strains.index(s) for s in sel_s]
        vmax = [records[i]["voronoi_max"] for i in sel_index]
        vmean = [records[i]["voronoi_mean"] for i in sel_index]
        keep = holes_mask.node_function(vmax, vmean, factor=2.0)
        sel_s = [s for s, k in zip(sel_s, keep) if k]
        sel_p = [p for p, k in zip(sel_p, keep) if k]
        sel_t = [t for t, k in zip(sel_t, keep) if k]
    if len(sel_s) > 2:
        t_next, _, _, _ = predict_melting_point.node_function(
            sel_s, sel_p, sel_t, boundary_value=boundary_value)
    else:
        t_next = temperature * (1.10 if flag > 0 else 0.90)
    return MeltingIterationRecord(
        temperature_in=float(temperature), temperature_next=float(t_next),
        strains=list(strains), ratios=list(ratios), pressures=list(pressures),
        temperatures=list(temps), converged=False)


@pwf.as_function_node("result")
def refine_melting_point(structure, engine, t_guess, melting_input, crystalstructure):
    """Convergence loop over the refinement schedules until |ΔT| ≤ goal."""
    mi = melting_input
    schedules = list(zip(mi.timestep_lst, mi.fit_range_lst, mi.nve_steps_lst))
    temperature = float(t_guess)
    iterations: list[MeltingIterationRecord] = []
    converged = False
    for step_idx, (timestep, fit_range, nve_steps) in enumerate(schedules):
        rec = coexistence_iteration.node_function(
            structure, engine, temperature=temperature,
            crystalstructure=crystalstructure, fit_range=fit_range,
            n_strain_points=mi.n_strain_points, nvt_steps=mi.nvt_run_steps,
            nve_steps=nve_steps, npt_steps=mi.npt_run_steps, timestep=timestep,
            delta_t_melt=mi.delta_t_melt, ratio_boundary=mi.ratio_boundary,
            boundary_value=mi.boundary_value, seed=mi.seed, subdir=f"iter_{step_idx}")
        iterations.append(rec)
        if abs(rec.temperature_next - temperature) <= mi.convergence_goal:
            converged = True
            temperature = rec.temperature_next
            break
        temperature = rec.temperature_next
    return MeltingResult(
        melting_temperature=float(temperature), converged=converged,
        n_iterations=len(iterations), element=mi.element,
        crystalstructure=crystalstructure, n_atoms=len(structure),
        initial_guess=float(t_guess), iterations=iterations,
        report={"schedules": schedules})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_coexistence.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/coexistence.py tests/integration/test_melting_coexistence.py
git commit -m "feat(melting): Step 2 coexistence iteration + convergence loop"
```

---

### Task 12: Top-level macro + public API

**Files:**
- Create: `pyiron_workflow_atomistics/physics/melting/study.py`
- Modify: `pyiron_workflow_atomistics/physics/melting/__init__.py`
- Test: `tests/integration/test_melting_study.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `calculate_melting_point(engine, melting_input) -> MeltingResult` (callable as `@pwf.as_function_node` driver returning a `MeltingResult`).

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_melting_study.py
import pytest
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting import (
    MeltingInput, calculate_melting_point,
)

@pytest.mark.slow
def test_calculate_melting_point_end_to_end(tmp_path):
    mi = MeltingInput(element="Al", crystalstructure="fcc", a=4.05, n_atoms=500,
                      temperature_right=1400.0, strain_run_steps=40,
                      timestep_lst=[2.0], fit_range_lst=[0.05], nve_steps_lst=[20],
                      nvt_run_steps=20, npt_run_steps=20, n_strain_points=5,
                      ratio_boundary=0.4, seed=1)
    eng = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(),
                    working_directory=str(tmp_path))
    res = calculate_melting_point.node_function(eng, mi)
    assert res.element == "Al"
    assert res.initial_guess >= 0
    assert isinstance(res.melting_temperature, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_study.py -v -m slow`
Expected: FAIL (ImportError: calculate_melting_point)

- [ ] **Step 3: Write implementation**
```python
# pyiron_workflow_atomistics/physics/melting/study.py
from __future__ import annotations

import pyiron_workflow as pwf
from ase.data import atomic_numbers, reference_states
from pyiron_workflow_atomistics.engine import CalcInputMinimize, calculate
from dataclasses import replace
from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    analyse_reference_structure,
)
from pyiron_workflow_atomistics.physics.melting.coexistence import refine_melting_point
from pyiron_workflow_atomistics.physics.melting.initial_guess import (
    estimate_melting_temperature,
)
from pyiron_workflow_atomistics.physics.melting.structures import (
    create_coexistence_supercell,
)


def _default_crystalstructure(element):
    return reference_states[atomic_numbers[element]]["symmetry"]


@pwf.as_function_node("result")
def calculate_melting_point(engine, melting_input):
    """Full interface-method melting point: build → relax → Step 1 → Step 2."""
    mi = melting_input
    crystalstructure = mi.crystalstructure or _default_crystalstructure(mi.element)
    structure = create_coexistence_supercell.node_function(
        mi.element, crystalstructure, a=mi.a, n_atoms=mi.n_atoms)
    relax_engine = replace(
        engine, EngineInput=CalcInputMinimize(relax_cell=True)
    ).with_working_directory("minimize")
    relaxed = calculate.node_function(structure, engine=relax_engine).final_structure
    key_max, _, distribution_half = analyse_reference_structure.node_function(relaxed)
    t_guess, struct_at_guess = estimate_melting_temperature.node_function(
        relaxed, engine, key_max=key_max, distribution_half=distribution_half,
        crystalstructure=crystalstructure, temperature_left=mi.temperature_left,
        temperature_right=mi.temperature_right, strain_run_steps=mi.strain_run_steps,
        timestep=mi.timestep_lst[0], seed=mi.seed)
    result = refine_melting_point.node_function(
        struct_at_guess, engine, t_guess=t_guess, melting_input=mi,
        crystalstructure=crystalstructure)
    return result
```
Then set `pyiron_workflow_atomistics/physics/melting/__init__.py`:
```python
"""Melting-point via the interface/coexistence method."""

from pyiron_workflow_atomistics.physics.melting.coexistence import (
    coexistence_iteration,
    refine_melting_point,
)
from pyiron_workflow_atomistics.physics.melting.initial_guess import (
    estimate_melting_temperature,
)
from pyiron_workflow_atomistics.physics.melting.inputs import MeltingInput
from pyiron_workflow_atomistics.physics.melting.outputs import (
    MeltingIterationRecord,
    MeltingResult,
)
from pyiron_workflow_atomistics.physics.melting.study import calculate_melting_point

__all__ = [
    "MeltingInput",
    "MeltingIterationRecord",
    "MeltingResult",
    "calculate_melting_point",
    "coexistence_iteration",
    "estimate_melting_temperature",
    "refine_melting_point",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_melting_study.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add pyiron_workflow_atomistics/physics/melting/study.py pyiron_workflow_atomistics/physics/melting/__init__.py tests/integration/test_melting_study.py
git commit -m "feat(melting): top-level calculate_melting_point + public API"
```

---

### Task 13: LAMMPS velocity-capture patch (in pwl_dev)

**Files:**
- Modify: `/ptmp/hmai/pwl_dev/pyiron_workflow_lammps/lammps.py` (`arrays_to_ase_atoms`, `parse_LammpsOutput`)
- Test: `/ptmp/hmai/pwl_dev/tests/unit/test_velocity_capture.py`

**Interfaces:**
- Produces: trajectory `Atoms` in `EngineOutput.structures` carry per-atom velocities (so `get_temperature()` works on LAMMPS frames, symmetric with ASE).

- [ ] **Step 1: Inspect the exact functions**

Run: `sed -n '120,140p;216,235p' /ptmp/hmai/pwl_dev/pyiron_workflow_lammps/lammps.py`
Confirm: `parse_LammpsOutput` builds each frame via `arrays_to_ase_atoms(...)` from `pyiron_lammps_output["generic"]["positions"][i]`, and that `pyiron_lammps_output["generic"]` also has `"velocities"`.

- [ ] **Step 2: Write the failing test**
```python
# /ptmp/hmai/pwl_dev/tests/unit/test_velocity_capture.py
import numpy as np
from pyiron_workflow_lammps.lammps import arrays_to_ase_atoms

def test_arrays_to_ase_atoms_attaches_velocities():
    species = [["Al", "Al"]]
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell = np.eye(3) * 5.0
    velocities = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    atoms = arrays_to_ase_atoms(species, positions, cell, pbc=True, velocities=velocities)
    assert atoms.get_velocities() is not None
    assert np.allclose(atoms.get_velocities(), velocities)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /ptmp/hmai/pwl_dev && /ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/test_velocity_capture.py -v`
Expected: FAIL (TypeError: unexpected keyword 'velocities')

- [ ] **Step 4: Patch `arrays_to_ase_atoms` and `parse_LammpsOutput`**

Add an optional `velocities=None` parameter to `arrays_to_ase_atoms`; after building the `Atoms`, if `velocities is not None: atoms.set_velocities(np.asarray(velocities))`. In `parse_LammpsOutput`, pass `velocities=pyiron_lammps_output["generic"]["velocities"][i]` when the key exists (guard with `.get("velocities")`), else `None`. Exact edit:
```python
def arrays_to_ase_atoms(species_lists, positions, cell, pbc=True, velocities=None):
    pos = positions
    atoms = Atoms(symbols=species_lists[-1], positions=pos, cell=cell, pbc=pbc)
    if velocities is not None:
        atoms.set_velocities(np.asarray(velocities))
    return atoms
```
And in `parse_LammpsOutput`'s frame loop:
```python
generic = pyiron_lammps_output["generic"]
vel_traj = generic.get("velocities")
... 
arrays_to_ase_atoms(
    species_lists,
    positions=generic["positions"][i],
    cell=generic["cells"][i],
    pbc=True,
    velocities=(vel_traj[i] if vel_traj is not None else None),
)
```

- [ ] **Step 5: Run test to verify pass + commit**

Run: `cd /ptmp/hmai/pwl_dev && /ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit/test_velocity_capture.py -v`
Expected: PASS
```bash
cd /ptmp/hmai/pwl_dev
git add pyiron_workflow_lammps/lammps.py tests/unit/test_velocity_capture.py
git commit -m "feat: attach per-atom velocities to trajectory frames (coexistence T)"
```

---

### Task 14: Notebook → runnable script

**Files:**
- Create: `scripts/meltingpoint.py`
- Test: `tests/integration/test_meltingpoint_script.py`

**Interfaces:**
- Produces: a CLI script that runs Step 1 (and optionally the full method) on EMT/Al and prints the result; importable `run(...)` for the test.

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_meltingpoint_script.py
import pytest

@pytest.mark.slow
def test_script_run_returns_result(tmp_path):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "meltingpoint", pathlib.Path("scripts/meltingpoint.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.run(element="Al", a=4.05, n_atoms=500, working_directory=str(tmp_path),
                  full=False, temperature_right=1400.0, strain_run_steps=40, seed=1)
    assert res["initial_guess"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_meltingpoint_script.py -v -m slow`
Expected: FAIL (FileNotFoundError / no module)

- [ ] **Step 3: Write the script**
```python
#!/usr/bin/env python
"""Interface-method melting point — runnable port of meltingpoint.ipynb.

Step 1 (rough estimate) by default; pass --full for the full coexistence method.
Uses the ASE engine with EMT by default (swap in a GRACE calculator for production).
"""

from __future__ import annotations

import argparse


def run(element="Al", crystalstructure="fcc", a=4.05, n_atoms=4000,
        working_directory="melting_run", full=False, temperature_right=1000.0,
        strain_run_steps=1000, seed=12345):
    from ase.calculators.emt import EMT
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.analysis.structure_descriptors import (
        analyse_reference_structure,
    )
    from pyiron_workflow_atomistics.physics.melting import (
        MeltingInput, calculate_melting_point,
    )
    from pyiron_workflow_atomistics.physics.melting.initial_guess import (
        estimate_melting_temperature,
    )
    from pyiron_workflow_atomistics.physics.melting.structures import (
        create_coexistence_supercell,
    )

    engine = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(),
                       working_directory=working_directory)
    if full:
        mi = MeltingInput(element=element, crystalstructure=crystalstructure, a=a,
                          n_atoms=n_atoms, temperature_right=temperature_right,
                          strain_run_steps=strain_run_steps, seed=seed)
        res = calculate_melting_point.node_function(engine, mi).to_dict()
        print(f"Melting temperature: {res['melting_temperature']:.1f} K "
              f"(converged={res['converged']}, guess={res['initial_guess']:.0f} K)")
        return res

    structure = create_coexistence_supercell.node_function(
        element, crystalstructure, a=a, n_atoms=n_atoms)
    key_max, _, half = analyse_reference_structure.node_function(structure)
    t_guess, _ = estimate_melting_temperature.node_function(
        structure, engine, key_max=key_max, distribution_half=half,
        crystalstructure=crystalstructure, temperature_right=temperature_right,
        strain_run_steps=strain_run_steps, seed=seed)
    print(f"Step-1 melting-temperature estimate: {t_guess} K")
    return {"initial_guess": float(t_guess)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--element", default="Al")
    p.add_argument("--crystalstructure", default="fcc")
    p.add_argument("--a", type=float, default=4.05)
    p.add_argument("--n-atoms", type=int, default=4000)
    p.add_argument("--working-directory", default="melting_run")
    p.add_argument("--full", action="store_true")
    p.add_argument("--temperature-right", type=float, default=1000.0)
    p.add_argument("--strain-run-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    run(element=args.element, crystalstructure=args.crystalstructure, a=args.a,
        n_atoms=args.n_atoms, working_directory=args.working_directory, full=args.full,
        temperature_right=args.temperature_right, strain_run_steps=args.strain_run_steps,
        seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify pass**

Run: `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/integration/test_meltingpoint_script.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
cd /ptmp/hmai/pwa_melting
git add scripts/meltingpoint.py tests/integration/test_meltingpoint_script.py
git commit -m "feat(melting): runnable meltingpoint.py script (notebook port)"
```

---

## Verification phase (after all tasks)

V1. **Full fast suite:** `/ptmp/hmai/pwa_melting/.venv/bin/python -m pytest tests/unit -q` → all green.
V2. **EMT/Al slow suite:** `… -m slow tests/integration -q` → all green; capture Step-1 guess + a `calculate_melting_point` Tm and report paths.
V3. **GRACE (ASE):** in `.venv-grace`, build an `ASEEngine` with `grace_fm(<cached model>)` (`GRACE_CACHE=/ptmp/hmai/grace_cache`), run Step 1 on a real element; report Tm.
V4. **LAMMPS (`pair_style grace`):** build a `LammpsEngine` (`command=/ptmp/hmai/lammps/build/lmp …`, `path_to_model=<exported GRACE>`, patched velocity capture), run one `coexistence_iteration` on a scaled cell; confirm the *same* algorithm runs unchanged and `mean_T` is finite.
V5. Report all output directories under `/ptmp/...` per the show-output-paths preference.

## Self-review checklist (done at plan close)
- Spec coverage: analysis quantities (T1–T3), structures (T5), solid fraction (T6), fitting (T7), MD drivers (T9), Step 1 (T10), Step 2 (T11), top-level (T12), LAMMPS wrap+patch (T13), script (T14), NPT robustness (T8), dataclasses (T4). All spec §3.5 nodes mapped.
- Type consistency: `analyse_reference_structure` → `(key_max, n_atoms, distribution_half)` consumed identically in T6/T9/T10/T12; `EngineOutput.structures/.stresses` consumed in T1; `MeltingIterationRecord`/`MeltingResult` produced in T4, consumed in T11/T12.
- No placeholders: every step has runnable code/commands.
