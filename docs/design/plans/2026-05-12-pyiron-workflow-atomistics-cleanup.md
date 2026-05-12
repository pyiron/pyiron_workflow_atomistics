# Cleanup and Reorganise pyiron_workflow_atomistics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape `pyiron_workflow_atomistics` into a clean, LLM-legible package with a formal `Engine` Protocol, physics-level input dataclasses, four top-level subpackages (`engine`/`structure`/`physics`/`analysis`), and a curated public API — without losing functionality.

**Architecture:** Six logically-grouped commits on the `cleanup-and-reorganise` branch. Commit 2 is the "atomic blast radius" — the entire engine layer is rewritten and every importer flipped in one commit so no compatibility shims are needed. Commits 3–5 are smaller plumbing relocations on top. Commit 6 curates the public API and ports notebooks + tests.

**Tech Stack:** Python 3.11, `pyiron_workflow==0.15.2` (`@pwf.as_function_node` / `@pwf.as_macro_node` decorators), `ase==3.26.0`, `pymatgen==2025.6.14`, `nbclient` for notebook execution. Dev env: `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp` (already has every dep installed).

**Spec:** `docs/design/specs/2026-05-12-pyiron-workflow-atomistics-cleanup-design.md`.

**Working directory for every shell step:** `/home/liger/pyiron_workflow_atomistics`. **Python interpreter:** `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python`. **Pytest:** `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest`.

---

## File-Structure Map

### Final layout

```
pyiron_workflow_atomistics/
├── __init__.py                       # MODIFY — only __version__
├── py.typed                          # CREATE — empty marker
├── engine/
│   ├── __init__.py                   # CREATE
│   ├── protocol.py                   # CREATE
│   ├── inputs.py                     # CREATE
│   └── ase.py                        # CREATE
├── structure/
│   ├── __init__.py                   # MODIFY (currently empty)
│   ├── build.py                      # CREATE
│   ├── transform.py                  # MOVE from structure_manipulator/tools.py
│   └── defects.py                    # CREATE (consolidates interstitials + vacancy)
├── physics/
│   ├── __init__.py                   # CREATE — empty by design
│   ├── bulk.py                       # MOVE from bulk.py
│   ├── surface.py                    # MOVE from surface/surface_study.py
│   ├── point_defect.py               # CONSOLIDATE from bulk_defect/{point,vacancy}.py
│   ├── grain_boundary.py             # CONSOLIDATE from gb/{gb_study,cleavage,segregation,optimiser,utils,hcp_generator}.py
│   └── _grain_boundary_code/         # MOVE from gb/gb_code/
│       ├── __init__.py
│       ├── constructor.py
│       └── searcher.py
├── analysis/
│   ├── __init__.py                   # CREATE
│   ├── featurisers.py                # MOVE from featurisers.py (with rename)
│   ├── gb_plane.py                   # MOVE from gb/analysis.py
│   └── quantities.py                 # MOVE get_per_atom_quantity from utils.py
└── _internal/
    ├── __init__.py                   # CREATE — empty
    ├── kwargs_helpers.py             # CREATE
    ├── dataclass_helpers.py          # CREATE
    └── workdir.py                    # CREATE
```

### Deletions

- `pyiron_workflow_atomistics_test/` (checked-in venv, ~7 MB)
- `pyiron_module_template/`
- `working_surface_example.py`
- `engine_ase/` (whole directory)
- `dataclass_storage.py`
- `calculator.py`
- `utils.py`
- `bulk.py` (after move)
- `bulk_defect/` (after consolidation)
- `gb/` (after consolidation; `gb/gb_code/` becomes `physics/_grain_boundary_code/`)
- `structure_manipulator/` (after move)
- `surface/` (after move; `surface/builder.py` → split into `structure/build.py` + delete)
- `featurisers.py` (after move)
- `tests/unit/test_calculator.py` (after migrating still-relevant tests)
- `tests/unit/test_utils.py` (after migrating still-relevant tests)
- `notebooks/notebook_blank.ipynb` (literal stub)
- `notebooks/equations_of_state.ipynb` + `notebooks/equations_of_state_ase.ipynb` (merged into `notebooks/eos.ipynb`)

---

## Pre-Flight

### Task 0: Confirm starting state

- [ ] **Step 1: Confirm branch and working directory**

```bash
cd /home/liger/pyiron_workflow_atomistics
git branch --show-current
git log -1 --oneline
```

Expected: branch `cleanup-and-reorganise`, last commit `e0b84c9 docs(design): cleanup-and-reorganise design spec`.

- [ ] **Step 2: Confirm test env**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "import pyiron_workflow_atomistics, ase, pyiron_workflow; print(pyiron_workflow_atomistics.__version__, ase.__version__, pyiron_workflow.__version__)"
```

Expected: version triple printed without error.

- [ ] **Step 3: Record baseline test count**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" --co 2>&1 | tail -5
```

Expected: collected `<N>` items. Write that number into your scratch notes — every commit must collect at least the symbols-that-survive subset green.

---

## Commit 1 — `chore: repo hygiene`

### Task 1: Delete checked-in venv, template stub, and root stray

- [ ] **Step 1: Delete the checked-in virtualenv**

```bash
git rm -rf pyiron_workflow_atomistics_test
```

Expected: ~hundreds of files removed from index.

- [ ] **Step 2: Delete the empty module template**

```bash
git rm -rf pyiron_workflow_atomistics/pyiron_module_template
```

- [ ] **Step 3: Delete the root-level dev stray**

```bash
git rm working_surface_example.py
```

- [ ] **Step 4: Delete tracked cache directories**

```bash
git rm -rf --ignore-unmatch __pycache__ .pytest_cache hcp_gb_generator/.pytest_cache pyiron_workflow_atomistics/__pycache__ pyiron_workflow_atomistics.egg-info
```

`--ignore-unmatch` because some of these may already be untracked.

### Task 2: Add `.gitignore`

- [ ] **Step 1: Append/replace `.gitignore`**

Show me the current content first, then merge:

```bash
cat .gitignore
```

The existing `.gitignore` (per the repo): mostly Python defaults. Replace it with the consolidated version below.

Write to `/home/liger/pyiron_workflow_atomistics/.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg
*.egg-info/
.eggs/
build/
dist/
pip-wheel-metadata/

# Test / coverage
.pytest_cache/
.coverage
htmlcov/

# Notebooks
.ipynb_checkpoints/

# Virtual environments
.venv/
venv/
env/
*.venv*/
pyiron_workflow_atomistics_test/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Stage `.gitignore`**

```bash
git add .gitignore
```

### Task 3: Commit hygiene

- [ ] **Step 1: Sanity-check the index**

```bash
git status -s | head -20
git diff --cached --stat | tail -5
```

Expected: deletions only, plus the `.gitignore` modification. No additions.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore: remove repository debris and tighten .gitignore

- Delete checked-in pyiron_workflow_atomistics_test/ virtualenv (~7 MB).
- Delete unused pyiron_module_template/ stub directory.
- Delete root-level working_surface_example.py dev stray.
- Untrack __pycache__/, .pytest_cache/, egg-info — now properly ignored.

No source code touched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

```bash
git log -1 --stat | tail -10
```

Expected: commit landed with file count visible.

---

## Commit 2 — `refactor(engine): protocol, EngineOutput dataclass, consolidate engine_ase`

This is the atomic "blast radius" commit. We do TDD per new module, then migrate every importer in one shot.

### Task 4: Build `engine/protocol.py` with TDD

**Files:**
- Create: `pyiron_workflow_atomistics/engine/__init__.py` (initially empty)
- Create: `pyiron_workflow_atomistics/engine/protocol.py`
- Create: `tests/unit/engine/__init__.py`
- Create: `tests/unit/engine/test_protocol.py`

- [ ] **Step 1: Create the test scaffold (failing — module doesn't exist yet)**

Create `tests/unit/engine/__init__.py` (empty file).

Create `tests/unit/engine/test_protocol.py`:

```python
"""Tests for the Engine Protocol, EngineOutput dataclass, and run() node."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


def test_engine_protocol_is_runtime_checkable():
    """A duck-typed object with the right attrs should isinstance() as Engine."""
    from pyiron_workflow_atomistics.engine.protocol import Engine

    @dataclass
    class FakeEngine:
        working_directory: str = "fake"

        def get_calculate_fn(self, structure: Atoms):
            return (lambda **kw: None, {})

        def with_working_directory(self, subdir: str) -> "FakeEngine":
            return FakeEngine(working_directory=f"{self.working_directory}/{subdir}")

    assert isinstance(FakeEngine(), Engine)


def test_engine_output_is_dataclass_with_required_fields():
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    output = EngineOutput(
        final_structure=structure,
        final_energy=-1.23,
        converged=True,
    )
    assert output.final_energy == pytest.approx(-1.23)
    assert output.converged is True
    # Optional fields default to None
    assert output.final_forces is None
    assert output.final_stress is None
    assert output.final_stress_voigt is None


def test_engine_output_to_dict_round_trip():
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput

    output = EngineOutput(
        final_structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        final_energy=-1.23,
        converged=True,
        final_forces=np.zeros((1, 3)),
    )
    d = output.to_dict()
    assert d["final_energy"] == pytest.approx(-1.23)
    assert d["converged"] is True
    assert d["final_forces"].shape == (1, 3)


def test_run_node_dispatches_to_engine():
    """run(structure, engine) calls engine.get_calculate_fn and invokes the result."""
    from pyiron_workflow_atomistics.engine.protocol import EngineOutput, run

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    sentinel_output = EngineOutput(
        final_structure=structure, final_energy=42.0, converged=True
    )

    @dataclass
    class StubEngine:
        working_directory: str = "."

        def get_calculate_fn(self, structure: Atoms):
            def fn(structure, **kwargs):
                return sentinel_output
            return fn, {"some": "kwarg"}

        def with_working_directory(self, subdir: str) -> "StubEngine":
            return StubEngine(working_directory=f"./{subdir}")

    out = run.node_function(structure=structure, engine=StubEngine())
    assert out is sentinel_output
```

- [ ] **Step 2: Run the test — confirm import failure**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_protocol.py -q
```

Expected: collection or import error, e.g. `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.engine'`.

- [ ] **Step 3: Create `engine/__init__.py`**

Create `pyiron_workflow_atomistics/engine/__init__.py` (empty for now — Task 8 populates it).

- [ ] **Step 4: Implement `engine/protocol.py`**

Create `pyiron_workflow_atomistics/engine/protocol.py`:

```python
"""Engine Protocol, EngineOutput dataclass, and the single run() entry point.

The Engine Protocol defines the contract every compute engine (ASE, VASP,
LAMMPS, ...) must satisfy so physics workflows can use them interchangeably.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms


@runtime_checkable
class Engine(Protocol):
    """An engine computes properties of a structure.

    Implementations live alongside their backends:
    :class:`pyiron_workflow_atomistics.engine.ase.ASEEngine` here, future
    ``VaspEngine`` / ``LammpsEngine`` in their own packages.

    Contract
    --------
    Engines MUST be pickleable so workflows can be checkpointed or submitted
    to SLURM. Engines MUST implement ``with_working_directory`` purely (no
    mutation of self). These properties are documented but not enforced via
    ``__reduce__`` — relying on duck typing keeps the contract simple.

    Attributes
    ----------
    working_directory
        Root directory the engine writes calc artefacts into. Sub-workflows
        compose paths by calling :meth:`with_working_directory`.
    """

    working_directory: str

    def get_calculate_fn(
        self, structure: Atoms
    ) -> tuple[Callable[..., "EngineOutput"], dict[str, Any]]:
        """Return ``(callable, kwargs)``. The callable will be invoked as
        ``callable(structure=structure, **kwargs)`` and must return an
        :class:`EngineOutput`."""

    def with_working_directory(self, subdir: str) -> "Engine":
        """Return a *copy* of this engine whose ``working_directory`` is
        ``os.path.join(self.working_directory, subdir)``.

        Pure — never mutates ``self``. Replaces the historical
        ``duplicate_engine`` helper.
        """


@dataclass
class EngineOutput:
    """Structured result of a single engine evaluation.

    Required
    --------
    final_structure
        The final atomic structure (post-relaxation or last MD step).
    final_energy
        Total potential energy in eV.
    converged
        True if the engine reports the calculation converged.

    Optional per-property
    ---------------------
    Single trailing values; engines fill what they compute.

    Trajectory (relax / MD only; ``None`` for static)
    -------------------------------------------------
    Lists indexed by ionic step.

    Examples
    --------
    >>> from ase.build import bulk
    >>> out = EngineOutput(
    ...     final_structure=bulk("Cu", "fcc", a=3.6, cubic=True),
    ...     final_energy=-3.5,
    ...     converged=True,
    ... )
    >>> out.to_dict()["final_energy"]
    -3.5
    """

    final_structure: Atoms
    final_energy: float
    converged: bool

    final_forces: np.ndarray | None = None
    final_stress: np.ndarray | None = None              # (3, 3)
    final_stress_voigt: np.ndarray | None = None        # (6,)
    final_volume: float | None = None
    final_magmoms: np.ndarray | None = None

    energies: list[float] | None = None
    forces: list[np.ndarray] | None = None
    stresses: list[np.ndarray] | None = None
    structures: list[Atoms] | None = None
    n_ionic_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of the dataclass fields (ASE objects preserved by reference)."""
        return asdict(self)


@pwf.as_function_node("engine_output")
def run(structure: Atoms, engine: Engine) -> EngineOutput:
    """Execute ``engine`` on ``structure``.

    The one node every physics workflow uses to compute things.

    Examples
    --------
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize, run
    >>> engine = ASEEngine(
    ...     EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
    ...     calculator=EMT(),
    ...     working_directory="./_demo",
    ... )
    >>> out = run.node_function(bulk("Cu", "fcc", a=3.6, cubic=True), engine)  # doctest: +SKIP
    """
    fn, kwargs = engine.get_calculate_fn(structure)
    return fn(structure=structure, **kwargs)
```

- [ ] **Step 5: Run the test — confirm PASS**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_protocol.py -q
```

Expected: `4 passed`.

### Task 5: Build `engine/inputs.py` with TDD

**Files:**
- Create: `pyiron_workflow_atomistics/engine/inputs.py`
- Create: `tests/unit/engine/test_inputs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/engine/test_inputs.py`:

```python
"""Tests for the physics-level engine input dataclasses."""
from __future__ import annotations

import pytest


def test_calc_input_static_is_empty_dataclass():
    from pyiron_workflow_atomistics.engine.inputs import CalcInputStatic

    inp = CalcInputStatic()
    assert hasattr(inp, "__dataclass_fields__")


def test_calc_input_minimize_defaults():
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMinimize

    inp = CalcInputMinimize()
    assert inp.force_convergence_tolerance > 0
    assert inp.energy_convergence_tolerance > 0
    assert inp.max_iterations > 0
    assert inp.relax_cell is False


def test_calc_input_md_renamed_field():
    """`thermostat_time_constant` replaces the old `temperature_damping_timescale`."""
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    inp = CalcInputMD()
    assert hasattr(inp, "thermostat_time_constant")
    assert not hasattr(inp, "temperature_damping_timescale")


def test_calc_input_md_time_step_in_fs():
    """time_step is in fs (default ~1 fs), not ps."""
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    inp = CalcInputMD()
    # 1 fs is the conventional ASE/LAMMPS default — anything < 100 means fs
    assert inp.time_step < 100.0


def test_calc_input_md_no_dropped_fields():
    """delta_temp and delta_press are removed."""
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    inp = CalcInputMD()
    assert not hasattr(inp, "delta_temp")
    assert not hasattr(inp, "delta_press")


def test_calc_input_md_no_lammps_jargon_in_docstring():
    from pyiron_workflow_atomistics.engine.inputs import CalcInputMD

    assert "LAMMPS units style" not in (CalcInputMD.__doc__ or "")
```

- [ ] **Step 2: Run — confirm failure**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_inputs.py -q
```

Expected: `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.engine.inputs'`.

- [ ] **Step 3: Implement `engine/inputs.py`**

Create `pyiron_workflow_atomistics/engine/inputs.py`:

```python
"""Physics-level input dataclasses for engine calculations.

These dataclasses describe *what* you want the engine to do in
physics-level terms (force tolerance, temperature, ensemble) — never in
engine-specific jargon (no EDIFFG, no LAMMPS units style). The engine is
responsible for translating these to its native parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CalcInputStatic:
    """A single-point energy/force evaluation. No tunable parameters."""

    pass


@dataclass
class CalcInputMinimize:
    """Structural relaxation parameters.

    Attributes
    ----------
    force_convergence_tolerance
        Max allowed force component on any atom, in eV/Å. Default 1e-2.
    energy_convergence_tolerance
        Energy change between consecutive steps, in eV. Default 1e-5.
    max_iterations
        Hard cap on optimiser steps.
    relax_cell
        If True, relax cell vectors too (variable-cell relaxation).
    """

    force_convergence_tolerance: float = 1e-2
    energy_convergence_tolerance: float = 1e-5
    max_iterations: int = 1_000_000
    relax_cell: bool = False


@dataclass
class CalcInputMD:
    """Molecular-dynamics parameters with selectable ensemble and thermostat.

    Attributes
    ----------
    mode
        Ensemble: ``"NVE"``, ``"NVT"``, or ``"NPT"``.
    thermostat
        Coupling algorithm. ``"nose-hoover"`` (deterministic), ``"langevin"``
        (stochastic), ``"berendsen"`` (weak-coupling), ``"andersen"`` (random
        collisions).
    temperature
        Target temperature in Kelvin.
    n_ionic_steps
        Number of MD timesteps to run.
    n_print
        Frequency of thermo output (steps).
    pressure
        Target pressure in Pascal (used by NPT mode only).
    time_step
        Integration timestep in **femtoseconds**.
    thermostat_time_constant
        Thermostat coupling timescale in femtoseconds.
    pressure_damping_timescale
        Barostat coupling timescale in femtoseconds.
    seed
        RNG seed for stochastic thermostats. ``None`` ⇒ non-deterministic.
    initial_temperature
        Temperature used to initialise velocities (defaults to ``temperature``).
    """

    mode: Literal["NVE", "NVT", "NPT"] = "NVT"
    thermostat: Literal[
        "nose-hoover", "langevin", "berendsen", "andersen"
    ] = "langevin"
    temperature: float = 300.0
    n_ionic_steps: int = 10_000
    n_print: int = 100
    pressure: float | None = None
    time_step: float = 1.0                       # fs
    thermostat_time_constant: float = 100.0      # fs
    pressure_damping_timescale: float = 1000.0   # fs
    seed: int | None = None
    initial_temperature: float | None = None
```

- [ ] **Step 4: Run — confirm PASS**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_inputs.py -q
```

Expected: `6 passed`.

### Task 6: Build `engine/ase.py` (consolidated ASEEngine) — characterisation tests first

This is the largest single file in the engine layer. Build it by writing characterisation tests against the public surface, then porting the canonical `ase_calc_structure`, `ase_md_calc_structure`, and `ASEEngine` from the three duplicate files in `engine_ase/`.

**Files:**
- Create: `pyiron_workflow_atomistics/engine/ase.py`
- Create: `tests/unit/engine/test_ase.py`

- [ ] **Step 1: Write characterisation tests**

Create `tests/unit/engine/test_ase.py`:

```python
"""Characterisation tests for ASEEngine: real EMT round-trip + pickle round-trip."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


def test_ase_engine_isinstance_engine_protocol():
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.engine.protocol import Engine

    eng = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory="./_t",
    )
    assert isinstance(eng, Engine)


def test_ase_engine_static_run_returns_engine_output(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputStatic,
        EngineOutput,
        run,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=structure, engine=engine)

    assert isinstance(out, EngineOutput)
    assert out.converged is True
    assert isinstance(out.final_energy, float)
    assert out.final_forces is not None
    assert out.final_forces.shape == (len(structure), 3)
    assert out.final_volume == pytest.approx(structure.get_volume())


def test_ase_engine_minimize_run_reduces_force(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputMinimize,
        run,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    # Perturb so the optimiser has work to do
    structure.rattle(0.05, seed=0)
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05, max_iterations=200),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=structure, engine=engine)

    # All forces are below the tolerance the optimiser reported converged on,
    # OR optimiser hit max steps with reduced forces — either way forces dropped.
    assert out.final_forces is not None
    final_fmax = float(np.linalg.norm(out.final_forces, axis=1).max())
    assert final_fmax < 1.0  # generous bound; rattle 0.05 yields ~few eV/Å initially


def test_ase_engine_with_working_directory_is_pure(tmp_path: Path):
    """with_working_directory returns a copy; original is untouched."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic

    eng = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    sub = eng.with_working_directory("substep")

    assert eng.working_directory == str(tmp_path)
    assert sub.working_directory == f"{tmp_path}/substep"
    assert eng is not sub


def test_ase_engine_pickle_round_trip(tmp_path: Path):
    """ASEEngine with EMT() calculator must pickle and unpickle cleanly."""
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize

    eng = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    blob = pickle.dumps(eng)
    restored = pickle.loads(blob)
    assert restored.working_directory == eng.working_directory
    assert restored.EngineInput.force_convergence_tolerance == 0.05
```

- [ ] **Step 2: Run — confirm failure**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_ase.py -q
```

Expected: import errors on `pyiron_workflow_atomistics.engine.ASEEngine`.

- [ ] **Step 3: Implement `engine/ase.py`**

This file is large (~400 lines). It consolidates the three existing files:
- `engine_ase/ase_calculator.py` is the canonical implementation — port `ase_calc_structure` and `ase_md_calc_structure` from there.
- `engine_ase/ase.py` defines `ASEEngine` — port that, simplify.
- `engine_ase/ase_engine.py` is fully obsolete (older, dict-returning version) — DO NOT port from it.

Create `pyiron_workflow_atomistics/engine/ase.py`:

```python
"""ASE-backed Engine implementation.

Consolidates and replaces engine_ase/{ase.py, ase_calculator.py, ase_engine.py}.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write
from ase.optimize import BFGS

from pyiron_workflow_atomistics.engine.inputs import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
)
from pyiron_workflow_atomistics.engine.protocol import EngineOutput


# ---------------------------------------------------------------------------
# Low-level helpers: gather() + attach_props()
# ---------------------------------------------------------------------------

def _gather(atoms: Atoms, properties: Tuple[str, ...]) -> dict[str, Any]:
    props = [p.strip() for p in properties]
    results: dict[str, Any] = {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces().tolist(),
        "cell": atoms.get_cell().tolist(),
        "volume": atoms.get_volume(),
        "positions": atoms.get_positions().tolist(),
        "numbers": atoms.get_atomic_numbers().tolist(),
        "masses": atoms.get_masses().tolist(),
    }
    if "stresses" in props:
        try:
            results["stresses"] = atoms.get_stress().tolist()
        except Exception:
            pass
    optional_map = {
        "charges": "get_charges",
        "dipole": "get_dipole_moment",
        "magmoms": "get_magnetic_moments",
        "virial": "get_virial",
        "pressure": "get_pressure",
    }
    for key, method in optional_map.items():
        if key in props:
            try:
                val = getattr(atoms, method)()
                results[key] = val.tolist() if hasattr(val, "tolist") else val
            except Exception:
                pass
    missing = [p for p in props if p not in results]
    if missing:
        raise KeyError(f"Requested properties not available: {missing}")
    return {p: results[p] for p in props}


def _attach_props(atoms: Atoms, results: dict[str, Any]) -> Atoms:
    if "energy" in results:
        atoms.info["energy"] = results["energy"]
    if "forces" in results:
        atoms.set_array("forces", np.array(results["forces"]))
    if "stresses" in results:
        atoms.info["stresses"] = results["stresses"]
    return atoms


def _build_engine_output(
    *,
    final_atoms: Atoms,
    final_res: dict[str, Any],
    trajectory: list[dict[str, Any]],
    converged: bool,
) -> EngineOutput:
    out = EngineOutput(
        final_structure=final_atoms,
        final_energy=float(final_res["energy"]),
        converged=bool(converged),
    )
    if "forces" in final_res:
        out.final_forces = np.array(final_res["forces"])
    if "stresses" in final_res:
        s = np.array(final_res["stresses"])
        if s.shape == (6,):
            out.final_stress_voigt = s
            # Reconstruct full 3x3 from Voigt convention (xx,yy,zz,yz,xz,xy)
            out.final_stress = np.array(
                [[s[0], s[5], s[4]],
                 [s[5], s[1], s[3]],
                 [s[4], s[3], s[2]]]
            )
        elif s.shape == (3, 3):
            out.final_stress = s
            out.final_stress_voigt = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
            )
    out.final_volume = float(final_res.get("volume", final_atoms.get_volume()))
    if "magmoms" in final_res:
        out.final_magmoms = np.array(final_res["magmoms"])
    if trajectory:
        out.energies = [step["results"].get("energy") for step in trajectory]
        out.forces = [
            np.array(step["results"]["forces"])
            for step in trajectory
            if "forces" in step["results"]
        ]
        if "stresses" in trajectory[0]["results"]:
            out.stresses = [np.array(step["results"]["stresses"]) for step in trajectory]
        out.structures = [step["structure"] for step in trajectory]
        out.n_ionic_steps = len(trajectory)
    return out


# ---------------------------------------------------------------------------
# Core calc functions (static/minimize and MD)
# ---------------------------------------------------------------------------

def ase_calc_structure(
    structure: Atoms,
    calc: Calculator,
    optimizer_class: Optional[type] = BFGS,
    optimizer_kwargs: Optional[dict[str, Any]] = None,
    record_interval: int = 1,
    fmax: float = 0.01,
    max_steps: int = 10_000,
    relax_cell: bool = False,
    energy_convergence_tolerance: Optional[float] = None,
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume"),
    write_to_disk: bool = False,
    working_directory: str = "calc_output",
    initial_struct_path: Optional[str] = "initial_structure.xyz",
    initial_results_path: Optional[str] = "initial_results.json",
    traj_struct_path: Optional[str] = "trajectory.xyz",
    traj_results_path: Optional[str] = "trajectory_results.json",
    final_struct_path: Optional[str] = "final_structure.xyz",
    final_results_path: Optional[str] = "final_results.json",
    data_pickle: str = "job_data.pkl.gz",
) -> EngineOutput:
    """Relax (or single-point) an ASE Atoms object and return an EngineOutput."""
    os.makedirs(working_directory, exist_ok=True)
    optimizer_kwargs = dict(optimizer_kwargs or {})

    atoms = structure.copy()
    atoms.calc = calc

    initial_res = _gather(atoms, properties)
    if write_to_disk and initial_struct_path:
        ase_write(os.path.join(working_directory, initial_struct_path), _attach_props(atoms.copy(), initial_res))
    if write_to_disk and initial_results_path:
        with open(os.path.join(working_directory, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    trajectory: list[dict[str, Any]] = []

    if optimizer_class is None:
        # Static
        snap = atoms.copy()
        snap_res = _gather(atoms, properties)
        trajectory.append({"structure": _attach_props(snap, snap_res), "results": snap_res})
        converged = True
    else:
        # Relaxation
        if relax_cell:
            from ase.constraints import ExpCellFilter
            atoms_filtered = ExpCellFilter(atoms)
            optimizer = optimizer_class(atoms_filtered, **optimizer_kwargs)

            def record_step():
                actual = atoms_filtered.atoms.copy()
                snap_res = _gather(actual, properties)
                trajectory.append({"structure": _attach_props(actual, snap_res), "results": snap_res})
                if write_to_disk and traj_struct_path:
                    ase_write(os.path.join(working_directory, traj_struct_path), _attach_props(actual.copy(), snap_res), append=True)
            optimizer.attach(record_step, interval=record_interval)
            converged = optimizer.run(fmax=fmax, steps=max_steps)
            atoms = atoms_filtered.atoms.copy()
        else:
            optimizer = optimizer_class(atoms, **optimizer_kwargs)

            def record_step():
                snap = atoms.copy()
                snap_res = _gather(atoms, properties)
                trajectory.append({"structure": _attach_props(snap, snap_res), "results": snap_res})
                if write_to_disk and traj_struct_path:
                    ase_write(os.path.join(working_directory, traj_struct_path), _attach_props(atoms.copy(), snap_res), append=True)
            optimizer.attach(record_step, interval=record_interval)
            converged = optimizer.run(fmax=fmax, steps=max_steps)

        if energy_convergence_tolerance and len(trajectory) >= 2:
            ediff = abs(trajectory[-1]["results"]["energy"] - trajectory[-2]["results"]["energy"])
            if ediff < energy_convergence_tolerance:
                converged = True

    final_res = _gather(atoms, properties)
    final_atoms = _attach_props(atoms.copy(), final_res)

    if write_to_disk and final_struct_path:
        ase_write(os.path.join(working_directory, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(working_directory, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)
    if write_to_disk and traj_results_path:
        with open(os.path.join(working_directory, traj_results_path), "w") as f:
            json.dump([step["results"] for step in trajectory], f, indent=2)

    df = pd.DataFrame([{"structure": s["structure"], **s["results"]} for s in trajectory])
    df.to_pickle(os.path.join(working_directory, data_pickle), compression="gzip")

    return _build_engine_output(
        final_atoms=final_atoms,
        final_res=final_res,
        trajectory=trajectory,
        converged=converged,
    )


def ase_md_calc_structure(
    structure: Atoms,
    calc: Calculator,
    md_input: CalcInputMD,
    record_interval: int = 1,
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume"),
    write_to_disk: bool = False,
    working_directory: str = "calc_output",
    initial_struct_path: Optional[str] = "initial_structure.xyz",
    initial_results_path: Optional[str] = "initial_results.json",
    traj_struct_path: Optional[str] = "trajectory.xyz",
    traj_results_path: Optional[str] = "trajectory_results.json",
    final_struct_path: Optional[str] = "final_structure.xyz",
    final_results_path: Optional[str] = "final_results.json",
    data_pickle: str = "job_data.pkl.gz",
) -> EngineOutput:
    """Run MD with ASE using the CalcInputMD dataclass for ensemble settings."""
    from ase import units
    from ase.md import Langevin
    from ase.md.npt import NPT
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    os.makedirs(working_directory, exist_ok=True)
    atoms = structure.copy()
    atoms.calc = calc

    initial_res = _gather(atoms, properties)
    if write_to_disk and initial_struct_path:
        ase_write(os.path.join(working_directory, initial_struct_path), _attach_props(atoms.copy(), initial_res))
    if write_to_disk and initial_results_path:
        with open(os.path.join(working_directory, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    T0 = md_input.initial_temperature or md_input.temperature
    if T0 > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=np.random.RandomState(md_input.seed))

    trajectory: list[dict[str, Any]] = []

    def record_step():
        snap = atoms.copy()
        snap_res = _gather(atoms, properties)
        trajectory.append({"structure": _attach_props(snap, snap_res), "results": snap_res})
        if write_to_disk and traj_struct_path:
            ase_write(os.path.join(working_directory, traj_struct_path), _attach_props(atoms.copy(), snap_res), append=True)

    dt = md_input.time_step * units.fs   # CalcInputMD.time_step is in fs
    T = md_input.temperature
    ttime = md_input.thermostat_time_constant * units.fs

    if md_input.mode == "NVE":
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(atoms, dt)
    elif md_input.mode == "NVT":
        if md_input.thermostat == "nose-hoover":
            from ase.md.nvt import NVT
            dyn = NVT(atoms, dt, temperature_K=T, ttime=ttime)
        elif md_input.thermostat == "berendsen":
            dyn = NVTBerendsen(atoms, dt, temperature_K=T, taut=ttime)
        else:  # langevin or andersen → langevin
            dyn = Langevin(atoms, dt, temperature_K=T, friction=1.0 / ttime,
                           rng=np.random.RandomState(md_input.seed))
    elif md_input.mode == "NPT":
        if md_input.pressure is None:
            raise ValueError("Pressure must be specified for NPT ensemble")
        P_bar = md_input.pressure / 1e5
        taup = md_input.pressure_damping_timescale * units.fs
        if md_input.thermostat == "nose-hoover":
            dyn = NPT(atoms, dt, temperature_K=T, externalstress=P_bar, ttime=ttime, pfactor=taup)
        elif md_input.thermostat == "berendsen":
            dyn = NPTBerendsen(atoms, dt, temperature_K=T, pressure_au=P_bar, taut=ttime, taup=taup)
        else:
            raise ValueError(f"NPT supports only 'nose-hoover' or 'berendsen', got {md_input.thermostat!r}")
    else:
        raise ValueError(f"Unknown MD mode: {md_input.mode!r}")

    dyn.attach(record_step, interval=record_interval)
    dyn.run(md_input.n_ionic_steps)

    final_res = _gather(atoms, properties)
    final_atoms = _attach_props(atoms.copy(), final_res)

    if write_to_disk and final_struct_path:
        ase_write(os.path.join(working_directory, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(working_directory, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)
    if write_to_disk and traj_results_path:
        with open(os.path.join(working_directory, traj_results_path), "w") as f:
            json.dump([s["results"] for s in trajectory], f, indent=2)

    df = pd.DataFrame([{"structure": s["structure"], **s["results"]} for s in trajectory])
    df.to_pickle(os.path.join(working_directory, data_pickle), compression="gzip")

    return _build_engine_output(
        final_atoms=final_atoms,
        final_res=final_res,
        trajectory=trajectory,
        converged=True,
    )


# ---------------------------------------------------------------------------
# ASEEngine — the user-facing class
# ---------------------------------------------------------------------------

@dataclass
class ASEEngine:
    """An :class:`pyiron_workflow_atomistics.engine.protocol.Engine` backed by ASE."""

    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    calculator: Calculator
    working_directory: str = field(default_factory=os.getcwd)
    optimizer_class: type | None = BFGS
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    record_interval: int = 1
    max_steps: int = 10_000
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume")
    write_to_disk: bool = False
    initial_struct_path: Optional[str] = "initial_structure.xyz"
    initial_results_path: Optional[str] = "initial_results.json"
    traj_struct_path: Optional[str] = "trajectory.xyz"
    traj_results_path: Optional[str] = "trajectory_results.json"
    final_struct_path: Optional[str] = "final_structure.xyz"
    final_results_path: Optional[str] = "final_results.json"
    data_pickle: str = "job_data.pkl.gz"

    def get_calculate_fn(self, structure: Atoms):
        common = dict(
            calc=self.calculator,
            working_directory=self.working_directory,
            properties=self.properties,
            write_to_disk=self.write_to_disk,
            initial_struct_path=self.initial_struct_path,
            initial_results_path=self.initial_results_path,
            traj_struct_path=self.traj_struct_path,
            traj_results_path=self.traj_results_path,
            final_struct_path=self.final_struct_path,
            final_results_path=self.final_results_path,
            data_pickle=self.data_pickle,
        )
        if isinstance(self.EngineInput, CalcInputStatic):
            kwargs = {**common, "optimizer_class": None, "optimizer_kwargs": {},
                      "record_interval": 1, "fmax": 0.0, "max_steps": 0}
            return ase_calc_structure, kwargs
        if isinstance(self.EngineInput, CalcInputMinimize):
            mi = self.EngineInput
            kwargs = {**common,
                      "optimizer_class": self.optimizer_class,
                      "optimizer_kwargs": self.optimizer_kwargs,
                      "record_interval": self.record_interval,
                      "fmax": mi.force_convergence_tolerance,
                      "max_steps": self.max_steps if self.max_steps else mi.max_iterations,
                      "relax_cell": mi.relax_cell,
                      "energy_convergence_tolerance": mi.energy_convergence_tolerance}
            return ase_calc_structure, kwargs
        if isinstance(self.EngineInput, CalcInputMD):
            kwargs = {**common, "md_input": self.EngineInput, "record_interval": self.record_interval}
            return ase_md_calc_structure, kwargs
        raise TypeError(f"Unsupported EngineInput type: {type(self.EngineInput).__name__}")

    def with_working_directory(self, subdir: str) -> "ASEEngine":
        return replace(self, working_directory=os.path.join(self.working_directory, subdir))
```

- [ ] **Step 4: Populate `engine/__init__.py` minimally so tests can import**

```python
"""Engine layer: Protocol, dataclasses, ASEEngine, run().

Public API:
    Engine (Protocol), EngineOutput, run,
    CalcInputStatic, CalcInputMinimize, CalcInputMD,
    ASEEngine.

Internal helpers live in ``pyiron_workflow_atomistics._internal``.
"""
from .protocol import Engine, EngineOutput, run
from .inputs import CalcInputStatic, CalcInputMinimize, CalcInputMD
from .ase import ASEEngine

__all__ = [
    "Engine",
    "EngineOutput",
    "run",
    "CalcInputStatic",
    "CalcInputMinimize",
    "CalcInputMD",
    "ASEEngine",
]
```

- [ ] **Step 5: Run engine tests — confirm all PASS**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine -q
```

Expected: `15 passed` (4 protocol + 6 inputs + 5 ase).

### Task 7: Migrate physics modules to the new engine paths

This task rewrites every importer of `calculator.py` / `dataclass_storage.py` / `engine_ase` in-place. The files stay where they are (commit 3 moves them). The pattern is:

**Before (typical physics module):**
```python
from pyiron_workflow_atomistics.calculator import (
    calculate_structure_node,
    validate_calculation_inputs,
)
from pyiron_workflow_atomistics.dataclass_storage import Engine
from pyiron_workflow_atomistics.utils import (
    duplicate_engine,
    get_calc_fn_calc_fn_kwargs_from_calculation_engine,
)
```

**After:**
```python
from pyiron_workflow_atomistics.engine import Engine, run
# duplicate_engine → engine.with_working_directory()
# validate_calculation_inputs → deleted; just pass engine
# get_calc_fn_calc_fn_kwargs_from_calculation_engine → deleted; run does it
```

Call-site rewrites:
- `calculate_structure_node(struct, calculation_engine=eng)` → `run(struct, engine=eng)`
- `duplicate_engine(eng, subdir)` → `eng.with_working_directory(subdir)`
- Anywhere using the legacy dual-mode `(_calc_structure_fn, _calc_structure_fn_kwargs)` → use `run` with a wrapped engine

- [ ] **Step 1: Inventory the importers**

```bash
rtk proxy grep -lE "from pyiron_workflow_atomistics\.(calculator|dataclass_storage|engine_ase)" pyiron_workflow_atomistics/ 2>&1
```

Expected files in the list:
- `pyiron_workflow_atomistics/bulk.py`
- `pyiron_workflow_atomistics/bulk_defect/point.py`
- `pyiron_workflow_atomistics/bulk_defect/vacancy.py`
- `pyiron_workflow_atomistics/surface/surface_study.py`
- `pyiron_workflow_atomistics/gb/gb_study.py`
- `pyiron_workflow_atomistics/gb/cleavage.py`
- `pyiron_workflow_atomistics/gb/segregation.py`
- `pyiron_workflow_atomistics/gb/optimiser.py`

Also grep for `duplicate_engine`:

```bash
rtk proxy grep -lE "duplicate_engine|get_calc_fn_calc_fn_kwargs_from_calculation_engine" pyiron_workflow_atomistics/ 2>&1
```

- [ ] **Step 2: Migrate `bulk.py`**

Read `pyiron_workflow_atomistics/bulk.py`. Replace:

```python
from pyiron_workflow_atomistics.calculator import validate_calculation_inputs
from pyiron_workflow_atomistics.dataclass_storage import Engine
from pyiron_workflow_atomistics.utils import get_per_atom_quantity
from .calculator import calculate_structure_node
```

with:

```python
from pyiron_workflow_atomistics.engine import Engine, run
from pyiron_workflow_atomistics.utils import get_per_atom_quantity
```

In `evaluate_structures`, replace the `validate_calculation_inputs(...)` + `calculation_engine.get_calculate_fn(...)` + manual kwargs munging block with:

```python
@pwf.as_function_node("engine_output_lst")
def evaluate_structures(
    structures: list[Atoms],
    engine: Engine,
    parent_working_directory: str = ".",
):
    engine_output_lst = []
    for i, struct in enumerate(structures):
        sub_engine = engine.with_working_directory(f"strain_{i:03d}")
        engine_output_lst.append(run.node_function(structure=struct, engine=sub_engine))
    return engine_output_lst
```

**Naming convention adopted in commit 2 and onward:** new and rewritten public macros take the engine as `engine: Engine`. Where pre-existing macros took `calculation_engine`, that name is renamed to `engine` as part of this commit (the user approved breaking the public API). Update doctests / notebook usage accordingly.

In `eos_volume_scan`, drop the `calc_structure_fn` / `calc_structure_fn_kwargs` parameters — the engine is the only input. Use `extract_output_values_from_EngineOutput` is being deleted; replace with inline list comprehension:

```python
wf.energies   = pwf.as_function_node("energies")(
    lambda outs: [o.final_energy for o in outs]
)(wf.evaluation.outputs.engine_output_lst)
```

Apply the analogous changes to `optimise_cubic_lattice_parameter` (drop `calc_structure_fn*` params, use engine only).

- [ ] **Step 3: Migrate `bulk_defect/vacancy.py` and `bulk_defect/point.py`**

Both files have an `Engine` parameter, call `duplicate_engine`, and call `calculate_structure_node`.

Replace:

```python
from pyiron_workflow_atomistics.calculator import calculate_structure_node
from pyiron_workflow_atomistics.utils import duplicate_engine
```

with:

```python
from pyiron_workflow_atomistics.engine import Engine, run
```

Rewrite the macro body:

```python
@pwf.as_macro_node("calc_supercell", "calc_vacancy", "vacancy_formation_energy")
def get_vacancy_formation_energy(wf,
                                 structure,
                                 engine: Engine,
                                 remove_atom_index=0,
                                 min_dimensions=[12, 12, 12],
                                 vacancy_subdir="vacancy_supercell",
                                 supercell_subdir="supercell"):
    wf.structure_supercell      = create_supercell_with_min_dimensions(structure, min_dimensions=min_dimensions)
    wf.structure_with_vacancy   = create_vacancy_structure(wf.structure_supercell, remove_atom_index=remove_atom_index)
    wf.vacancy_calc   = run(wf.structure_with_vacancy, engine=engine.with_working_directory(vacancy_subdir))
    wf.supercell_calc = run(wf.structure_supercell,    engine=engine.with_working_directory(supercell_subdir))
    wf.vacancy_formation_energy = calculate_vacancy_formation_energy(
        wf.vacancy_calc.outputs.engine_output.final_energy,
        wf.supercell_calc.outputs.engine_output.final_energy,
    )
    return wf.supercell_calc, wf.vacancy_calc, wf.vacancy_formation_energy
```

`bulk_defect/point.py` is a duplicate of `vacancy.py`; in this commit just delete its duplicated function bodies (they'll be fully removed in commit 3). For now, replace `bulk_defect/point.py` with:

```python
"""Deprecated stub — symbols moved to physics/point_defect.py in commit 3."""
```

- [ ] **Step 4: Migrate `surface/surface_study.py`**

Same pattern. Drop `calc_structure_fn`/`calc_structure_fn_kwargs` params, use only `engine: Engine`. Replace `calculate_structure_node(struct, calculation_engine=eng)` with `run(struct, engine=eng)`. The `_calculate_if_not_present_` helper becomes:

```python
@pwf.as_function_node("mu_bulk_out")
def _calculate_if_not_present_(input_structure, engine: Engine, mu_bulk=None):
    if mu_bulk is None:
        output = run.node_function(input_structure, engine=engine)
        return output.final_energy / len(input_structure)
    return mu_bulk
```

- [ ] **Step 5: Migrate `gb/*.py`**

Four files: `gb_study.py`, `cleavage.py`, `segregation.py`, `optimiser.py`. Same import-flip pattern.

The trickiest piece: `gb/optimiser.py` and `gb/cleavage.py` use `get_working_subdir_kwargs` + manual kwargs munging. Replace each occurrence:

```python
kwargs = get_working_subdir_kwargs(calc_structure_fn_kwargs=kw, base_working_directory=base, new_working_directory=sub)
out = calculate_structure_node(struct, _calc_structure_fn=fn, _calc_structure_fn_kwargs=kwargs)
```

with:

```python
out = run.node_function(struct, engine=engine.with_working_directory(sub))
```

This will substantially shrink each file. After migration, run:

```bash
rtk proxy grep -nE "calculate_structure_node|duplicate_engine|get_calc_fn_calc_fn_kwargs_from_calculation_engine|validate_calculation_inputs" pyiron_workflow_atomistics/ 2>&1
```

Expected: **no results**. If any remain, hunt them down before continuing.

- [ ] **Step 6: Update `featurisers.py` import (no behavioural change)**

`featurisers.py` doesn't import from the old engine layer, but it uses `voronoiSiteFeaturiser` etc. Leave it alone in this commit; the rename happens in commit 3 alongside the file move.

- [ ] **Step 7: Run full unit test suite (some failures expected)**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -30
```

Expected:
- `tests/unit/engine/` — all pass (15).
- `tests/unit/test_calculator.py` — many failures (we're about to delete most of it).
- `tests/unit/test_utils.py` — many failures (same).
- `tests/unit/test_bulk.py` — possibly some failures from the API change; will fix in the test rewrite step.

This is OK — the next step prunes the failing tests.

### Task 8: Delete old engine code and dead tests

- [ ] **Step 1: Delete the obsolete engine_ase directory**

```bash
git rm -rf pyiron_workflow_atomistics/engine_ase
```

- [ ] **Step 2: Delete `calculator.py` and `dataclass_storage.py`**

```bash
git rm pyiron_workflow_atomistics/calculator.py pyiron_workflow_atomistics/dataclass_storage.py
```

- [ ] **Step 3: Delete the tests for symbols we just removed**

`tests/unit/test_calculator.py` tests `calculate_structure_node`, `validate_calculation_inputs`, `convert_EngineOutput_to_output_dict`, `extract_output_values_from_EngineOutput`, `extract_values_from_dict`, `fillin_default_calckwargs`, `generate_kwargs_variant(s)`, `add_arg_to_kwargs_list`.

Of these, `fillin_default_calckwargs`, `generate_kwargs_variant`, `generate_kwargs_variants` are moving to `_internal/` in commit 5 — keep those tests by **moving them out** into a new `tests/unit/_internal/test_kwargs_helpers.py` (we'll wire that up in commit 5). For now:

```bash
git rm tests/unit/test_calculator.py
```

`tests/unit/test_utils.py` tests `add_string`, `convert_structure`, `convert_EngineOutput_to_output_dict`, `extract_outputs_from_EngineOutputs`, `get_calc_fn_calc_fn_kwargs_from_calculation_engine`, `get_per_atom_quantity`, `get_subdirpaths`, `get_working_subdir_kwargs`, `modify_dataclass`, `modify_dataclass_multi`, `modify_dict`.

- `add_string`, `convert_structure`, `get_calc_fn_calc_fn_kwargs_from_calculation_engine`: deleted symbols → delete their tests.
- `get_per_atom_quantity`: keep (moving to `analysis/quantities.py` in commit 3).
- `modify_dataclass*`, `modify_dict`, `get_subdirpaths`, `get_working_subdir_kwargs`: moving to `_internal/` in commit 5.

For commit 2's atomicity, just delete `test_utils.py` outright; commit 5 reconstructs the relevant tests under `tests/unit/_internal/`:

```bash
git rm tests/unit/test_utils.py
```

- [ ] **Step 4: Run full test suite — expect green**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -10
```

Expected: all tests pass. If any fail in `tests/unit/test_bulk.py` because `bulk.py`'s public function signature changed (drop of `calc_structure_fn` params), update those tests now.

- [ ] **Step 5: Confirm imports work**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import pyiron_workflow_atomistics
from pyiron_workflow_atomistics.engine import (
    Engine, EngineOutput, run,
    CalcInputStatic, CalcInputMinimize, CalcInputMD,
    ASEEngine,
)
print('engine import OK')
import pyiron_workflow_atomistics.bulk
import pyiron_workflow_atomistics.surface.surface_study
import pyiron_workflow_atomistics.bulk_defect.vacancy
import pyiron_workflow_atomistics.gb.gb_study
print('physics imports OK')
"
```

Expected: both prints succeed.

### Task 9: Commit the engine refactor

- [ ] **Step 1: Review the staged diff**

```bash
git add -A
git diff --cached --stat | tail -20
```

Expected: large deletion (engine_ase/, calculator.py, dataclass_storage.py, two test files) and creation of `engine/{protocol,inputs,ase,__init__}.py` plus three new test files plus modified physics modules.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(engine): introduce Engine Protocol, EngineOutput dataclass, consolidate ASE engine

- Add `engine/protocol.py` with a runtime-checkable `Engine` Protocol,
  a real `@dataclass` `EngineOutput` (replacing the
  PrintableClass-with-class-attrs spread across the codebase), and a
  single `run(structure, engine)` node that replaces the dual-mode
  `calculate_structure_node`.
- Add `engine/inputs.py` with jargon-stripped `CalcInputStatic`,
  `CalcInputMinimize`, `CalcInputMD`. `time_step` units are now fs;
  `temperature_damping_timescale` renamed to `thermostat_time_constant`;
  unused `delta_temp` / `delta_press` fields dropped.
- Add `engine/ase.py` consolidating the three duplicate files under
  `engine_ase/`. Now the canonical ASE engine returns the new
  `EngineOutput` dataclass and supports `with_working_directory()`
  (pure copy via `dataclasses.replace`) — replaces the free-floating
  `duplicate_engine` helper that was defined three times.
- Flip every physics module (`bulk.py`, `surface/surface_study.py`,
  `bulk_defect/*.py`, `gb/*.py`) to import from `engine.*` and use
  `run` + `with_working_directory` directly. Drop the legacy
  `calc_structure_fn` / `calc_structure_fn_kwargs` dual-mode args.
- Delete `engine_ase/`, `dataclass_storage.py`, `calculator.py`, and
  the unit tests that covered the now-deleted symbols.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

```bash
git log -1 --stat | tail -20
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -5
```

Expected: commit landed; tests pass.

---

## Commit 3 — `refactor(physics+structure): module reshuffle`

Pure path relocation + a handful of public-symbol renames. All imports were updated in commit 2.

### Task 10: Build `structure/`

- [ ] **Step 1: Create the subpackage layout**

```bash
mkdir -p pyiron_workflow_atomistics/structure
touch pyiron_workflow_atomistics/structure/__init__.py
```

- [ ] **Step 2: Move `structure_manipulator/tools.py` → `structure/transform.py`**

```bash
git mv pyiron_workflow_atomistics/structure_manipulator/tools.py pyiron_workflow_atomistics/structure/transform.py
```

Edit `pyiron_workflow_atomistics/structure/transform.py` to rename `rattle_structure` → `rattle` (function definition + the `@pwf.as_function_node` decorator's output name).

- [ ] **Step 3: Move `structure_manipulator/interstitials.py` → `structure/defects.py`**

```bash
git mv pyiron_workflow_atomistics/structure_manipulator/interstitials.py pyiron_workflow_atomistics/structure/defects.py
```

In the moved file, rename `substitutional_swap_one_site` → `substitutional_swap`.

- [ ] **Step 4: Add `create_vacancy` to `structure/defects.py`**

Open `pyiron_workflow_atomistics/structure/defects.py` and add (at the top):

```python
import pyiron_workflow as pwf
from ase import Atoms


@pwf.as_function_node("vacancy_structure")
def create_vacancy(structure: Atoms, remove_atom_index: int = 0) -> Atoms:
    """Return a copy of ``structure`` with one atom removed.

    Examples
    --------
    >>> from ase.build import bulk
    >>> bulk_struct = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    >>> vac = create_vacancy.node_function(bulk_struct, remove_atom_index=0)
    >>> len(vac) == len(bulk_struct) - 1
    True
    """
    out = structure.copy()
    out.pop(remove_atom_index)
    return out
```

- [ ] **Step 5: Create `structure/build.py`**

Cut the bulk/surface builder code out of `bulk.py` and `surface/builder.py`. Create `pyiron_workflow_atomistics/structure/build.py`:

```python
"""Constructors for crystalline / surface structures."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import pyiron_workflow as pwf
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.build import surface as ase_surface


@pwf.as_function_node("equil_struct")
def get_bulk(
    name: str,
    crystalstructure: Optional[str] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    alpha: Optional[float] = None,
    covera: Optional[float] = None,
    u: Optional[float] = None,
    orthorhombic: bool = False,
    cubic: bool = False,
    basis: Optional[list] = None,
) -> Atoms:
    """Build a bulk crystal via ``ase.build.bulk``.

    Examples
    --------
    >>> cu = get_bulk.node_function("Cu", crystalstructure="fcc", a=3.6, cubic=True)
    >>> len(cu)
    4
    """
    return ase_bulk(
        name=name, crystalstructure=crystalstructure, a=a, b=b, c=c,
        alpha=alpha, covera=covera, u=u, orthorhombic=orthorhombic,
        cubic=cubic, basis=basis,
    )


@pwf.as_function_node("surface_slab")
def create_surface_slab(
    bulk_structure: Atoms,
    miller_indices: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (1, 1, 1),
    layers: int = 3,
    vacuum: float = 10.0,
    periodic: bool = True,
) -> Atoms:
    """Cut a slab from a bulk structure via ``ase.build.surface``.

    Examples
    --------
    >>> cu = get_bulk.node_function("Cu", crystalstructure="fcc", a=3.6, cubic=True)
    >>> slab = create_surface_slab.node_function(cu, miller_indices=(1, 1, 1), layers=3)
    >>> slab.pbc.all()
    True
    """
    return ase_surface(
        bulk_structure, indices=miller_indices, layers=layers,
        vacuum=vacuum, periodic=periodic,
    )
```

- [ ] **Step 6: Update `bulk.py` to drop `get_bulk_structure` (now in `structure/build.py`)**

Edit `pyiron_workflow_atomistics/bulk.py` to remove `get_bulk_structure` and add `from pyiron_workflow_atomistics.structure.build import get_bulk as get_bulk_structure` if any internal code still uses the old name (look for `bulk.get_bulk_structure` callsites; rename them to `get_bulk`).

- [ ] **Step 7: Populate `structure/__init__.py`**

```python
"""Structure manipulation — engine-agnostic builders, transformations, defects."""
from .build     import get_bulk, create_surface_slab
from .transform import (
    add_vacuum,
    create_supercell,
    create_supercell_with_min_dimensions,
    rattle,
)
from .defects   import (
    create_vacancy,
    substitutional_swap,
    # interstitial site finders (whatever interstitials.py exports)
)

__all__ = [
    "get_bulk", "create_surface_slab",
    "add_vacuum", "create_supercell", "create_supercell_with_min_dimensions", "rattle",
    "create_vacancy", "substitutional_swap",
]
```

Inspect `pyiron_workflow_atomistics/structure/defects.py` for the interstitial-site-finder names exposed (currently the file came from `structure_manipulator/interstitials.py` which has `get_octahedral_positions`, `get_tetrahedral_positions`, etc.). Add them to the import + `__all__` after renaming to `find_octahedral_sites` / `find_tetrahedral_sites`.

- [ ] **Step 8: Verify imports**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from pyiron_workflow_atomistics.structure import (
    get_bulk, create_surface_slab,
    add_vacuum, create_supercell, create_supercell_with_min_dimensions, rattle,
    create_vacancy, substitutional_swap,
)
print('structure imports OK')
"
```

### Task 11: Build `physics/`

- [ ] **Step 1: Create the subpackage**

```bash
mkdir -p pyiron_workflow_atomistics/physics
touch pyiron_workflow_atomistics/physics/__init__.py
```

Add the module docstring to `physics/__init__.py`:

```python
"""Physics workflows organised by topic.

Import per-topic, not from this package directly::

    from pyiron_workflow_atomistics.physics.bulk           import eos_volume_scan
    from pyiron_workflow_atomistics.physics.surface        import calculate_surface_energy
    from pyiron_workflow_atomistics.physics.point_defect   import get_vacancy_formation_energy
    from pyiron_workflow_atomistics.physics.grain_boundary import pure_gb_study

This package intentionally re-exports nothing so the import path tells you
which topic each macro belongs to.
"""
```

- [ ] **Step 2: Move `bulk.py` → `physics/bulk.py`**

```bash
git mv pyiron_workflow_atomistics/bulk.py pyiron_workflow_atomistics/physics/bulk.py
```

Update internal imports inside `physics/bulk.py`:
- `from .calculator import ...` → already removed in commit 2; no action.
- Anywhere referencing `get_bulk_structure` → `from pyiron_workflow_atomistics.structure.build import get_bulk` and update call sites to `get_bulk(name=…)`.

- [ ] **Step 3: Move `surface/surface_study.py` → `physics/surface.py`**

```bash
git mv pyiron_workflow_atomistics/surface/surface_study.py pyiron_workflow_atomistics/physics/surface.py
```

In `physics/surface.py`, replace `from pyiron_workflow_atomistics.surface.builder import create_surface` with `from pyiron_workflow_atomistics.structure.build import create_surface_slab` and update call sites (`create_surface(...)` → `create_surface_slab(...)`).

- [ ] **Step 4: Delete `surface/`**

```bash
git rm -rf pyiron_workflow_atomistics/surface
```

(This deletes `surface/builder.py` and `surface/__init__.py`. `surface_study.py` is already moved.)

- [ ] **Step 5: Consolidate `bulk_defect/` → `physics/point_defect.py`**

Create `pyiron_workflow_atomistics/physics/point_defect.py` from the content of `bulk_defect/vacancy.py`. `bulk_defect/point.py` is already a stub from commit 2.

```python
"""Point-defect formation energies (vacancy, substitutional)."""
from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, run
from pyiron_workflow_atomistics.structure.defects import create_vacancy, substitutional_swap
from pyiron_workflow_atomistics.structure.transform import create_supercell_with_min_dimensions


@pwf.as_function_node("formation_energy")
def calculate_vacancy_formation_energy(vacancy_energy: float, supercell_energy: float) -> float:
    """Vacancy formation energy as the bare difference (preserves the
    existing semantics in `bulk_defect/vacancy.py`).

    NOTE: the textbook formula uses the per-atom chemical potential
    ``mu_bulk`` rather than the bulk supercell energy
    (E_f = E_vac − (N−1)·mu_bulk). The current macro intentionally keeps
    the simpler `vacancy − supercell` form for backwards compatibility;
    switching to the (N−1)/N normalisation is tracked as a separate
    physics improvement — out of scope for the cleanup.
    """
    return vacancy_energy - supercell_energy


@pwf.as_macro_node("supercell_calc", "vacancy_calc", "vacancy_formation_energy")
def get_vacancy_formation_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    remove_atom_index: int = 0,
    min_dimensions: list = [12, 12, 12],
    vacancy_subdir: str = "vacancy",
    supercell_subdir: str = "supercell",
):
    """Standard vacancy formation energy macro.

    Examples
    --------
    See ``notebooks/vacancy_formation_energy.ipynb``.
    """
    wf.structure_supercell    = create_supercell_with_min_dimensions(structure, min_dimensions=min_dimensions)
    wf.structure_with_vacancy = create_vacancy(wf.structure_supercell, remove_atom_index=remove_atom_index)
    wf.supercell_calc = run(wf.structure_supercell,    engine=engine.with_working_directory(supercell_subdir))
    wf.vacancy_calc   = run(wf.structure_with_vacancy, engine=engine.with_working_directory(vacancy_subdir))
    wf.vacancy_formation_energy = calculate_vacancy_formation_energy(
        vacancy_energy=wf.vacancy_calc.outputs.engine_output.final_energy,
        supercell_energy=wf.supercell_calc.outputs.engine_output.final_energy,
    )
    return wf.supercell_calc, wf.vacancy_calc, wf.vacancy_formation_energy


@pwf.as_macro_node("supercell_calc", "substitutional_calc", "substitutional_formation_energy")
def get_substitutional_formation_energy(
    wf,
    structure: Atoms,
    engine: Engine,
    defect_site: int = 0,
    new_symbol: str = "Ni",
    mu_solute: float = 0.0,
    mu_host: float = 0.0,
    min_dimensions: list = [12, 12, 12],
    sub_subdir: str = "substitutional",
    supercell_subdir: str = "supercell",
):
    """Dilute substitutional formation energy:
    E_f = E_sub - E_supercell - mu_solute + mu_host.
    """
    wf.structure_supercell        = create_supercell_with_min_dimensions(structure, min_dimensions=min_dimensions)
    wf.structure_with_substitute  = substitutional_swap(wf.structure_supercell, defect_site=defect_site, new_symbol=new_symbol)
    wf.supercell_calc       = run(wf.structure_supercell,       engine=engine.with_working_directory(supercell_subdir))
    wf.substitutional_calc  = run(wf.structure_with_substitute, engine=engine.with_working_directory(sub_subdir))

    @pwf.as_function_node("E_f")
    def _form_energy(E_sub, E_bulk, mu_solute, mu_host):
        return E_sub - E_bulk - mu_solute + mu_host

    wf.substitutional_formation_energy = _form_energy(
        wf.substitutional_calc.outputs.engine_output.final_energy,
        wf.supercell_calc.outputs.engine_output.final_energy,
        mu_solute, mu_host,
    )
    return wf.supercell_calc, wf.substitutional_calc, wf.substitutional_formation_energy
```

- [ ] **Step 6: Delete `bulk_defect/`**

```bash
git rm -rf pyiron_workflow_atomistics/bulk_defect
```

- [ ] **Step 7: Consolidate `gb/*.py` → `physics/grain_boundary.py`**

This is the biggest single-file consolidation. The six files (`gb_study.py`, `cleavage.py`, `segregation.py`, `optimiser.py`, `utils.py`, `hcp_generator.py`) together are ~2500 lines.

Strategy: concatenate the *public* macros from each into `physics/grain_boundary.py`, but push the internal helpers (`axis_to_index`, geometry helpers in `gb/utils.py`, the rigid-cleavage Plotter) to a new private sub-module:

```bash
mkdir -p pyiron_workflow_atomistics/physics/_grain_boundary_helpers
touch pyiron_workflow_atomistics/physics/_grain_boundary_helpers/__init__.py
```

- Move `gb/utils.py` → `physics/_grain_boundary_helpers/geometry.py`
- Move `gb/hcp_generator.py` → `physics/_grain_boundary_helpers/hcp.py`
- Move `gb/gb_code/` → `physics/_grain_boundary_code/`:

```bash
mkdir -p pyiron_workflow_atomistics/physics/_grain_boundary_code
git mv pyiron_workflow_atomistics/gb/gb_code/* pyiron_workflow_atomistics/physics/_grain_boundary_code/
```

Then create `physics/grain_boundary.py` as the merged public surface:

```python
"""Grain-boundary workflows: pure_gb_study, cleavage_study, segregation_study."""
# Import macros that need to be exposed publicly. Internal helpers (which
# may number in the dozens) stay in _grain_boundary_helpers / _grain_boundary_code.

# pure_gb_study:  paste body from gb/gb_study.py, replace imports and engine calls
# cleavage_study: paste body from gb/cleavage.py
# segregation_study: paste body from gb/segregation.py
# (full contents preserved verbatim modulo import paths and engine API)
```

Because this file is ~1500 lines of dense GB code, **do not retype it** in this commit. Instead, follow this concrete procedure:

```bash
# 1. Concatenate the four public-surface files into physics/grain_boundary.py
cat pyiron_workflow_atomistics/gb/gb_study.py \
    pyiron_workflow_atomistics/gb/cleavage.py \
    pyiron_workflow_atomistics/gb/segregation.py \
    pyiron_workflow_atomistics/gb/optimiser.py \
    > pyiron_workflow_atomistics/physics/grain_boundary.py

# 2. Hand-edit physics/grain_boundary.py:
#    - Deduplicate any imports
#    - Replace `from pyiron_workflow_atomistics.gb.utils import ...` with
#      `from pyiron_workflow_atomistics.physics._grain_boundary_helpers.geometry import ...`
#    - Replace `from pyiron_workflow_atomistics.gb.gb_code... import ...` with
#      `from pyiron_workflow_atomistics.physics._grain_boundary_code... import ...`
#    - Confirm `from pyiron_workflow_atomistics.engine import ...` is present
#      (the migration in commit 2 should already have flipped these imports;
#      this is a safety check after concatenation)

# 3. Delete the old gb/ directory
git rm -rf pyiron_workflow_atomistics/gb
```

After the merge, ensure no `import pyiron_workflow_atomistics.gb` remains anywhere:

```bash
rtk proxy grep -rE "pyiron_workflow_atomistics\.gb([^_]|$)" pyiron_workflow_atomistics/ tests/ 2>&1
```

Expected: no results (or only false positives from `_grain_boundary_*`).

- [ ] **Step 8: Run unit tests**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -10
```

Existing GB tests may need import-path updates: `tests/unit/test_gb_analysis.py`, `tests/unit/test_gb_cleavage.py`, `tests/unit/test_gb_code.py`. Update their imports from `pyiron_workflow_atomistics.gb.*` to `pyiron_workflow_atomistics.physics.grain_boundary` (for public macros) or `pyiron_workflow_atomistics.physics._grain_boundary_code` (for code internals).

### Task 12: Build `analysis/`

- [ ] **Step 1: Create the subpackage**

```bash
mkdir -p pyiron_workflow_atomistics/analysis
touch pyiron_workflow_atomistics/analysis/__init__.py
```

- [ ] **Step 2: Move `featurisers.py` → `analysis/featurisers.py` with rename**

```bash
git mv pyiron_workflow_atomistics/featurisers.py pyiron_workflow_atomistics/analysis/featurisers.py
```

Inside the file, rename:
- `voronoiSiteFeaturiser` → `voronoi_site_featuriser`
- `distanceMatrixSiteFeaturiser` → `distance_matrix_site_featuriser`
- `soapSiteFeaturiser` → `soap_site_featuriser`

Then `rtk proxy grep -lE "voronoiSiteFeaturiser|distanceMatrixSiteFeaturiser|soapSiteFeaturiser"` and update every call site (likely in `physics/grain_boundary.py` and test files).

- [ ] **Step 3: Move `gb/analysis.py` → `analysis/gb_plane.py`**

The file is already gone from `gb/` after Task 11 step 7. Recover it from the merged `physics/grain_boundary.py` is incorrect — `gb/analysis.py` was a separate file. Recover before Step 7's `git rm -rf`:

If you already deleted it, restore from HEAD~1:

```bash
git checkout HEAD~1 -- pyiron_workflow_atomistics/gb/analysis.py
mv pyiron_workflow_atomistics/gb/analysis.py pyiron_workflow_atomistics/analysis/gb_plane.py
rm -rf pyiron_workflow_atomistics/gb
```

(If you haven't deleted yet, just `git mv` it before Task 11 step 7's `git rm -rf gb/`.)

Inside `analysis/gb_plane.py`, rename the public functions to snake_case (e.g. `find_GB_plane` → `find_gb_plane`, `plot_GB_plane` → `plot_gb_plane`). Update callers in `physics/grain_boundary.py`.

- [ ] **Step 4: Create `analysis/quantities.py`**

```python
"""Derived scalar quantities (per-atom values, etc.)."""
from __future__ import annotations

import pyiron_workflow as pwf


@pwf.as_function_node("per_atom_quantity")
def get_per_atom_quantity(quantity: float, structure) -> float:
    """Divide a total-cell quantity by the number of atoms."""
    return quantity / len(structure)
```

- [ ] **Step 5: Populate `analysis/__init__.py`**

```python
"""Featurisation, post-processing, and derived quantities."""
from .featurisers import (
    voronoi_site_featuriser,
    distance_matrix_site_featuriser,
    soap_site_featuriser,
    summarize_cosine_groups,
    pca_whiten,
)
from .gb_plane import find_gb_plane, plot_gb_plane
from .quantities import get_per_atom_quantity

__all__ = [
    "voronoi_site_featuriser",
    "distance_matrix_site_featuriser",
    "soap_site_featuriser",
    "summarize_cosine_groups",
    "pca_whiten",
    "find_gb_plane",
    "plot_gb_plane",
    "get_per_atom_quantity",
]
```

- [ ] **Step 6: Delete the `structure_manipulator/` directory**

```bash
git rm -rf pyiron_workflow_atomistics/structure_manipulator
```

(After this, `structure_manipulator/` is fully consumed by `structure/`.)

### Task 13: Run tests + commit commit 3

- [ ] **Step 1: Run the full unit suite**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -10
```

Update any remaining tests whose imports broke. The big batch will be `test_gb_*` (paths) and `test_featurisers` (rename + path).

- [ ] **Step 2: Confirm top-level imports**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from pyiron_workflow_atomistics.structure import get_bulk, create_surface_slab, add_vacuum, create_supercell, rattle, create_vacancy, substitutional_swap
from pyiron_workflow_atomistics.physics.bulk import eos_volume_scan, optimise_cubic_lattice_parameter
from pyiron_workflow_atomistics.physics.surface import calculate_surface_energy
from pyiron_workflow_atomistics.physics.point_defect import get_vacancy_formation_energy, get_substitutional_formation_energy
from pyiron_workflow_atomistics.physics.grain_boundary import pure_gb_study
from pyiron_workflow_atomistics.analysis import voronoi_site_featuriser, find_gb_plane, get_per_atom_quantity
print('all subpackage imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add -A
git diff --cached --stat | tail -10
git commit -m "$(cat <<'EOF'
refactor(physics+structure): reshuffle modules into engine/structure/physics/analysis

- Move `structure_manipulator/tools.py` → `structure/transform.py`
  (rename `rattle_structure` → `rattle`).
- Move `structure_manipulator/interstitials.py` → `structure/defects.py`
  (rename `substitutional_swap_one_site` → `substitutional_swap`).
- Move `bulk_defect/vacancy.py:create_vacancy_structure` →
  `structure/defects.py:create_vacancy`.
- New `structure/build.py`: `get_bulk` (was `get_bulk_structure`) and
  `create_surface_slab` (was `create_surface`).
- Move `bulk.py` → `physics/bulk.py` (workflow macros only).
- Move `surface/surface_study.py` → `physics/surface.py`; delete
  `surface/builder.py` (its helpers are in `structure/build.py` now).
- Consolidate `bulk_defect/{point,vacancy}.py` → `physics/point_defect.py`
  and add the new `get_substitutional_formation_energy` macro.
- Consolidate `gb/{gb_study,cleavage,segregation,optimiser}.py` →
  `physics/grain_boundary.py`. Internal helpers move to
  `physics/_grain_boundary_helpers/`; `gb/gb_code/` becomes
  `physics/_grain_boundary_code/`.
- Move `gb/analysis.py` → `analysis/gb_plane.py` (rename
  `find_GB_plane`/`plot_GB_plane` to snake_case).
- Move `featurisers.py` → `analysis/featurisers.py` (rename the three
  public featuriser functions camelCase → snake_case).
- Move `utils.py:get_per_atom_quantity` → `analysis/quantities.py`.
- Update every internal import accordingly. No behaviour change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 4 — `refactor(engine): finalise with_working_directory adoption`

Commit 2 already introduced `with_working_directory` and used it in the physics modules' new bodies. This commit cleans up any remaining call sites and confirms the `duplicate_engine` function is fully gone.

### Task 14: Audit and finalise

- [ ] **Step 1: Search for any straggler usage**

```bash
rtk proxy grep -rE "duplicate_engine" pyiron_workflow_atomistics/ tests/ 2>&1
```

Expected: no results. If any survive (e.g. an internal call in `physics/grain_boundary.py` that was missed during concatenation), replace with `engine.with_working_directory(subdir)`.

- [ ] **Step 2: Verify the method is on `ASEEngine`**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
eng = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory='/tmp/a')
sub = eng.with_working_directory('b')
assert sub.working_directory == '/tmp/a/b'
assert eng.working_directory == '/tmp/a'
print('with_working_directory pure-copy OK')
"
```

- [ ] **Step 3: Look for direct mutations of `working_directory`**

```bash
rtk proxy grep -rE "\.working_directory\s*=" pyiron_workflow_atomistics/ 2>&1
```

Expected: only the dataclass field assignments inside `engine/ase.py` (via `field(default_factory=...)`). Any direct mutation of an engine's `working_directory` from outside is a bug — fix it to use `with_working_directory`.

- [ ] **Step 4: Run tests; commit (no-op if nothing changed)**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -5
```

If there are no source changes (commit 2 was thorough), this commit is empty — skip and renumber. If there are stragglers, commit them:

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(engine): finalise with_working_directory adoption

Sweep any remaining direct uses of the removed `duplicate_engine` helper
or direct `engine.working_directory = ...` mutation. All path-derivation
now goes through the pure `Engine.with_working_directory(subdir)` method.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 5 — `refactor(internal): hide kwargs plumbing under _internal/`

### Task 15: Build `_internal/`

- [ ] **Step 1: Create the subpackage**

```bash
mkdir -p pyiron_workflow_atomistics/_internal
touch pyiron_workflow_atomistics/_internal/__init__.py
```

- [ ] **Step 2: Create `_internal/kwargs_helpers.py`**

Cut the canonical implementations of `fillin_default_calckwargs`, `generate_kwargs_variant`, `generate_kwargs_variants` from wherever they survived commits 2–3. They originally lived in `calculator.py` (deleted) — they may also have been duplicated inside `engine_ase/*` (also deleted). The canonical source is the one inside `calculator.py`, which you must reconstruct from git history if needed:

```bash
git show HEAD~2:pyiron_workflow_atomistics/calculator.py > /tmp/old_calculator.py
```

then extract the three functions into `_internal/kwargs_helpers.py`:

```python
"""Internal helpers for fan-out parameter sweeps. NOT part of the public API."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pyiron_workflow as pwf


@pwf.as_function_node("full_calc_kwargs2")
def fillin_default_calckwargs(
    calc_kwargs: dict[str, Any],
    default_values: dict[str, Any] | None = None,
    remove_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Merge user kwargs with defaults; coerce ``properties`` to tuple; drop keys."""
    built_in: dict[str, Any] = dict(default_values) if isinstance(default_values, dict) else {}
    full: dict[str, Any] = dict(calc_kwargs)
    for key, default in built_in.items():
        full.setdefault(key, default)
    if "properties" in full:
        full["properties"] = tuple(full["properties"])
    if remove_keys:
        for key in remove_keys:
            full.pop(key, None)
    return full


@pwf.as_function_node("kwargs_variant")
def generate_kwargs_variant(base_kwargs: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Return a deepcopy of ``base_kwargs`` with ``key`` set to ``value``."""
    out = deepcopy(base_kwargs)
    out[key] = value
    return out


@pwf.as_function_node("kwargs_variants")
def generate_kwargs_variants(base_kwargs: dict[str, Any], key: str, values: list[Any]) -> list[dict[str, Any]]:
    """Return one variant per element of ``values`` with ``key`` overridden."""
    return [{**base_kwargs, key: v} for v in values]
```

- [ ] **Step 3: Create `_internal/dataclass_helpers.py`**

Cut `modify_dataclass`, `modify_dataclass_multi`, `modify_dict` from `utils.py`:

```python
"""Internal helpers for mid-graph mutation of dataclass / dict objects."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import pyiron_workflow as pwf


@pwf.as_function_node("modded_dataclass")
def modify_dataclass(dataclass_instance, entry_name: str, entry_value: Any):
    data = deepcopy(asdict(dataclass_instance))
    if entry_name not in data:
        raise KeyError(f"Unknown field: {entry_name!r}")
    data[entry_name] = entry_value
    return type(dataclass_instance)(**data)


@pwf.as_function_node("modded_dataclass_multi")
def modify_dataclass_multi(dataclass_instance, entry_names, entry_values):
    if len(entry_names) != len(entry_values):
        raise ValueError("entry_names and entry_values must have the same length")
    ds = dataclass_instance
    for name, val in zip(entry_names, entry_values):
        ds = modify_dataclass.node_function(ds, name, val)
    return ds


@pwf.as_function_node("modded_dict")
def modify_dict(dict_instance: dict, updates: dict) -> dict:
    new = deepcopy(dict_instance)
    invalid = set(updates) - set(new)
    if invalid:
        raise KeyError(f"Unknown key(s): {sorted(invalid)}")
    new.update(updates)
    return new
```

- [ ] **Step 4: Create `_internal/workdir.py`**

Cut `get_subdirpaths`, `get_working_subdir_kwargs` from `utils.py`:

```python
"""Internal helpers for working-directory composition (fan-out)."""
from __future__ import annotations

import os
from typing import List

import pyiron_workflow as pwf

from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dict


@pwf.as_function_node("output_dirs")
def get_subdirpaths(parent_dir: str, output_subdirs: List[str]) -> List[str]:
    return [os.path.join(parent_dir, sub) for sub in output_subdirs]


@pwf.api.as_function_node("dict_with_adjusted_working_directory")
def get_working_subdir_kwargs(
    calc_structure_fn_kwargs: dict,
    base_working_directory: str,
    new_working_directory: str,
):
    return modify_dict.node_function(
        calc_structure_fn_kwargs,
        {"working_directory": os.path.join(base_working_directory, new_working_directory)},
    )
```

- [ ] **Step 5: Update any importer**

```bash
rtk proxy grep -rlE "(fillin_default_calckwargs|generate_kwargs_variant|modify_dataclass|modify_dict|get_subdirpaths|get_working_subdir_kwargs)" pyiron_workflow_atomistics/ tests/ 2>&1
```

For each hit, replace the `from pyiron_workflow_atomistics.utils import ...` (or `from pyiron_workflow_atomistics.calculator import ...`) with `from pyiron_workflow_atomistics._internal.<module> import ...`.

- [ ] **Step 6: Delete `utils.py`**

```bash
git rm pyiron_workflow_atomistics/utils.py
```

(`get_per_atom_quantity` is already in `analysis/quantities.py` from commit 3; `duplicate_engine`, `add_string`, `convert_structure`, `get_calc_fn_calc_fn_kwargs_from_calculation_engine`, `extract_outputs_from_EngineOutputs` were all deleted in commits 2/3.)

- [ ] **Step 7: Add the kwargs-helper tests back**

Create `tests/unit/_internal/__init__.py` (empty) and `tests/unit/_internal/test_kwargs_helpers.py`:

```python
"""Tests for the private kwargs-plumbing helpers."""
from __future__ import annotations


def test_fillin_default_calckwargs_merges_defaults():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import fillin_default_calckwargs

    out = fillin_default_calckwargs.node_function(
        calc_kwargs={"a": 1},
        default_values={"a": 0, "b": 2},
    )
    assert out == {"a": 1, "b": 2}


def test_fillin_default_calckwargs_drops_keys():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import fillin_default_calckwargs

    out = fillin_default_calckwargs.node_function(
        calc_kwargs={"a": 1, "secret": 2},
        remove_keys=["secret"],
    )
    assert "secret" not in out


def test_generate_kwargs_variants_lists_them():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import generate_kwargs_variants

    out = generate_kwargs_variants.node_function(
        base_kwargs={"x": 0},
        key="x",
        values=[1, 2, 3],
    )
    assert out == [{"x": 1}, {"x": 2}, {"x": 3}]
```

- [ ] **Step 8: Run tests**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -5
```

Expected: green.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(internal): hide kwargs plumbing under _internal/

- `_internal/kwargs_helpers.py`: single canonical
  fillin_default_calckwargs, generate_kwargs_variant(s).
- `_internal/dataclass_helpers.py`: modify_dataclass(_multi), modify_dict.
- `_internal/workdir.py`: get_subdirpaths, get_working_subdir_kwargs.
- Delete top-level utils.py (its remaining public symbols moved to
  analysis/quantities.py or _internal/, the rest were deleted outright).

These helpers are still callable from anywhere via the explicit
_internal._ path, but are no longer part of the public API surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 6 — `refactor(api): curate __init__.py exports, fix tests/notebooks`

### Task 16: Top-level `__init__.py` and `py.typed`

- [ ] **Step 1: Trim the top-level `pyiron_workflow_atomistics/__init__.py`**

Current content imports `_version`. Leave that, but ensure nothing else gets re-exported at the top level (users always go through `engine`, `structure`, `physics`, `analysis`):

```python
"""pyiron_workflow_atomistics — atomistic-simulation workflows for pyiron.

Subpackages:
    engine    — the Engine Protocol, EngineOutput, run, and ASEEngine.
    structure — generic structure builders/transforms/defects.
    physics   — physics workflows (bulk, surface, point_defect, grain_boundary).
    analysis  — featurisation, post-processing, derived quantities.

Internal-only:
    _internal — kwargs plumbing, dataclass/dict mutators, workdir helpers.
"""
from . import _version

__version__ = _version.get_versions()["version"]
__all__ = ["__version__"]
```

- [ ] **Step 2: Add `py.typed`**

```bash
touch pyiron_workflow_atomistics/py.typed
git add pyiron_workflow_atomistics/py.typed
```

- [ ] **Step 3: Confirm every public `__init__.py` has __all__**

```bash
for d in engine structure analysis; do
    echo "--- $d ---"
    cat "pyiron_workflow_atomistics/$d/__init__.py" | grep "__all__"
done
```

Expected: each file has an `__all__` declaration.

### Task 17: Add new physics unit tests

**Files:**
- Create: `tests/unit/physics/__init__.py`
- Create: `tests/unit/physics/test_surface.py`
- Create: `tests/unit/physics/test_point_defect.py`

- [ ] **Step 1: Build `test_surface.py`**

```python
"""Smoke test for physics.surface using ASEEngine + EMT (Cu)."""
from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


@pytest.mark.slow
def test_calculate_surface_energy_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.surface import calculate_surface_energy

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.1, max_iterations=50),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_surface_energy(
        engine=engine, symbol="Cu", miller_indices=(1, 1, 1),
        min_length=8.0, vacuum=8.0, crystalstructure="fcc", a=3.6,
    )
    out.run()
    se = out.outputs.surface_energy.value
    assert se > 0  # surface energy is positive
    assert se < 5  # in J/m², Cu(111) is ~1.5 — generous upper bound
```

- [ ] **Step 2: Build `test_point_defect.py`**

```python
"""Smoke test for physics.point_defect using ASEEngine + EMT (Cu)."""
from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


@pytest.mark.slow
def test_vacancy_formation_energy_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.point_defect import get_vacancy_formation_energy

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.1, max_iterations=50),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = get_vacancy_formation_energy(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        engine=engine,
        min_dimensions=[8, 8, 8],
    )
    wf.run()
    e_f = wf.outputs.vacancy_formation_energy.value
    assert 0.5 < e_f < 2.5  # EMT Cu vacancy ~ 0.9–1.3 eV — generous bounds


@pytest.mark.slow
def test_substitutional_formation_energy_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.point_defect import get_substitutional_formation_energy

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.1, max_iterations=50),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = get_substitutional_formation_energy(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        engine=engine,
        new_symbol="Ni",
        min_dimensions=[8, 8, 8],
    )
    wf.run()
    e_f = wf.outputs.substitutional_formation_energy.value
    assert e_f is not None
```

### Task 18: Add the notebook execution integration test

- [ ] **Step 1: Create the test**

Create `tests/integration/test_notebook_execution.py`:

```python
"""Integration test: execute every notebook end-to-end.

Marked `slow`. Each notebook is executed in-place with a 600 s timeout.
The notebook supplies its own calculator (EMT or EAM) in the first
code cell, so this test has no notebook-specific setup.
"""
from __future__ import annotations

import pathlib

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.slow
def test_notebook_runs(nb_path: pathlib.Path):
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
```

- [ ] **Step 2: Verify it collects but doesn't run yet**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/integration/test_notebook_execution.py --co -q
```

Expected: 9 notebook tests collected (the count after notebook reshuffle, see Task 19).

### Task 19: Port and re-execute the notebooks

This is a sequence of mechanical edits — for each existing notebook, update the imports to point at `engine.*` / `physics.*` / `structure.*` / `analysis.*`, and update API calls to use `run` instead of `calculate_structure_node`. Then add two new notebooks (`substitutional_formation_energy.ipynb`, `gb_cleavage.ipynb`) and merge `equations_of_state*.ipynb` into one `eos.ipynb`.

- [ ] **Step 1: Merge `equations_of_state.ipynb` and `equations_of_state_ase.ipynb` → `eos.ipynb`**

```bash
git rm notebooks/equations_of_state.ipynb
git mv notebooks/equations_of_state_ase.ipynb notebooks/eos.ipynb
```

Edit `notebooks/eos.ipynb` to use the new API. The first code cell should be:

```python
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
from pyiron_workflow_atomistics.structure import get_bulk
from pyiron_workflow_atomistics.physics.bulk import eos_volume_scan

engine = ASEEngine(
    EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
    calculator=EMT(),
    working_directory="./_eos_runs",
)
```

Then the volume scan:

```python
structure = get_bulk.node_function("Cu", crystalstructure="fcc", a=3.6, cubic=True)
wf = eos_volume_scan(base_structure=structure, engine=engine,
                     axes=["a", "b", "c"], strain_range=(-0.05, 0.05), num_points=7)
wf.run()
print(f"v0 = {wf.outputs.v0.value:.3f} Å^3")
print(f"e0 = {wf.outputs.e0.value:.4f} eV")
print(f"B  = {wf.outputs.B.value:.1f} GPa")
```

- [ ] **Step 2: Extract `optimise_lattice_parameter.ipynb`**

Create `notebooks/optimise_lattice_parameter.ipynb` (e.g. start from a copy of `eos.ipynb`'s scaffold) demonstrating the `optimise_cubic_lattice_parameter` macro with EMT/Cu.

- [ ] **Step 3: Port `surface_energy.ipynb` (Fe → keep with EAM)**

Update its first code cell to:

```python
from ase.calculators.eam import EAM
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
from pyiron_workflow_atomistics.physics.surface import calculate_surface_energy

# The Al-Fe.eam.fs potential is checked into notebooks/
engine = ASEEngine(
    EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
    calculator=EAM(potential="Al-Fe.eam.fs"),
    working_directory="./_surf_runs",
)
```

- [ ] **Step 4: Port `vacancy_formation_energy.ipynb` (keep Fe + EAM)**

Same pattern: pass `engine=ASEEngine(calculator=EAM(...), EngineInput=CalcInputMinimize(...))`.

- [ ] **Step 5: Port `bulk_solution_energy.ipynb` (keep Fe + EAM)**

- [ ] **Step 6: Port `structure_optimisation.ipynb` (entry-level demo)**

The simplest possible notebook — make this the one a new user reads first. Demonstrates `run(structure, engine)` for a single perturbed Cu structure. Use EMT for simplicity.

- [ ] **Step 7: Port `pure_grain_boundary_study.ipynb` (keep Fe + EAM)**

Larger refactor — but most of the cells stay the same; just update imports and engine usage.

- [ ] **Step 8: Port `grain_boundary_segregation.ipynb` (keep Fe + EAM)**

- [ ] **Step 9: Create `notebooks/substitutional_formation_energy.ipynb` (NEW, EMT/Cu)**

Use the `get_substitutional_formation_energy` macro defined in `physics/point_defect.py`. Element: Ni-in-Cu (both EMT-supported, well-known dilute heat of solution).

- [ ] **Step 10: Create `notebooks/gb_cleavage.ipynb` (NEW, EAM/Fe)**

Standalone demo of `cleavage_study` extracted from the dense `pure_gb_study` flow.

- [ ] **Step 11: Delete the stub notebook**

```bash
git rm notebooks/notebook_blank.ipynb
```

- [ ] **Step 12: Re-execute every notebook end-to-end**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/integration/test_notebook_execution.py -m slow -v --tb=short 2>&1 | tail -30
```

Expected: 9 notebook tests pass. Total runtime ~30–60 minutes depending on hardware.

If a notebook fails, open it manually, fix the failing cell, then re-run only that notebook:

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/integration/test_notebook_execution.py -m slow -v -k "<notebook_name>"
```

- [ ] **Step 13: Stage re-executed notebooks (now with fresh outputs)**

```bash
git add notebooks/
git diff --cached --stat | tail -15
```

### Task 20: Final test + commit

- [ ] **Step 1: Run the full unit suite one last time**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow" 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 2: Confirm every advertised import works**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
# engine
from pyiron_workflow_atomistics.engine import (
    Engine, EngineOutput, run,
    CalcInputStatic, CalcInputMinimize, CalcInputMD,
    ASEEngine,
)
# structure
from pyiron_workflow_atomistics.structure import (
    get_bulk, create_surface_slab,
    add_vacuum, create_supercell, create_supercell_with_min_dimensions, rattle,
    create_vacancy, substitutional_swap,
)
# physics (topical)
from pyiron_workflow_atomistics.physics.bulk           import eos_volume_scan, optimise_cubic_lattice_parameter, equation_of_state
from pyiron_workflow_atomistics.physics.surface        import calculate_surface_energy
from pyiron_workflow_atomistics.physics.point_defect   import get_vacancy_formation_energy, get_substitutional_formation_energy
from pyiron_workflow_atomistics.physics.grain_boundary import pure_gb_study, cleavage_study, segregation_study
# analysis
from pyiron_workflow_atomistics.analysis import (
    voronoi_site_featuriser, distance_matrix_site_featuriser, soap_site_featuriser,
    find_gb_plane, plot_gb_plane,
    get_per_atom_quantity,
)
print('every advertised import resolved')
"
```

Expected: prints the success message.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(api): curate public __init__.py exports; fix tests and notebooks

- Populate every public subpackage __init__.py with explicit __all__
  re-exports (`engine`, `structure`, `analysis`). `physics/__init__.py`
  stays empty by design — users import per-topic for discoverability.
- Top-level `pyiron_workflow_atomistics/__init__.py` no longer re-exports
  symbols; only exposes `__version__` and a docstring describing the
  subpackage layout. Anything else is reached via the four subpackages.
- Add `py.typed` marker so consumers' type checkers pick up annotations.
- Migrate `tests/unit/{test_calculator,test_utils}.py` content into the
  new test tree (`tests/unit/{engine,physics,_internal}/...`) and delete
  the legacy test files.
- Add `tests/unit/physics/{test_surface,test_point_defect}.py` (no
  previous coverage).
- Add `tests/integration/test_notebook_execution.py`: runs every notebook
  in `notebooks/` via nbclient (10-min timeout each), marked `slow`.
- Notebook refresh: nine notebooks total. Merge equations_of_state* into
  `eos.ipynb`; extract `optimise_lattice_parameter.ipynb`; add the new
  `substitutional_formation_energy.ipynb` and `gb_cleavage.ipynb`; delete
  the `notebook_blank.ipynb` stub. All Fe-based studies keep the existing
  Al-Fe.eam.fs EAM potential. New EMT demos use Cu / Ni-in-Cu. Every
  notebook re-executed with fresh outputs committed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Show the branch history**

```bash
git log --oneline origin/main..HEAD
```

Expected: 6 commits on top of `39f006e` (the spec commit `e0b84c9` plus six cleanup commits):

```
<sha> refactor(api): curate public __init__.py exports; fix tests and notebooks
<sha> refactor(internal): hide kwargs plumbing under _internal/
<sha> refactor(engine): finalise with_working_directory adoption
<sha> refactor(physics+structure): reshuffle modules into engine/structure/physics/analysis
<sha> refactor(engine): introduce Engine Protocol, EngineOutput dataclass, consolidate ASE engine
<sha> chore: remove repository debris and tighten .gitignore
<sha> docs(design): cleanup-and-reorganise design spec
```

---

## Final Verification

### Task 21: End-to-end sanity

- [ ] **Step 1: Static checks**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check pyiron_workflow_atomistics/ 2>&1 | tail -10
```

Expected: clean (the repo already had ruff configured; any new lints are likely import-related and easy to fix).

- [ ] **Step 2: Full test suite, including slow**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/ -q --no-header 2>&1 | tail -5
```

Expected: every test (unit + integration + benchmark, including notebook executions) passes.

- [ ] **Step 3: Confirm net diff**

```bash
git diff --stat origin/main..HEAD | tail -3
```

Expected: net deletion in the −1500 LOC range, consistent with the spec's estimate.

- [ ] **Step 4: Push (optional, user decision)**

Do NOT push without user approval. The user will review the branch locally first.

```bash
# Only after user approval:
# git push -u origin cleanup-and-reorganise
```

---

## Self-Review Checklist

After completing every task above:

- [ ] All public symbols from the spec's "Public API per module" section are exported in some `__init__.py`'s `__all__`.
- [ ] All symbols on the spec's deletion list are gone from the codebase (`rtk proxy grep` finds zero hits).
- [ ] No file imports from `pyiron_workflow_atomistics.{calculator,dataclass_storage,utils,engine_ase,bulk_defect,gb,surface,structure_manipulator,bulk,featurisers}` (these top-level names no longer exist).
- [ ] Every test file lives under `tests/unit/<subpackage>/` or `tests/integration/` matching the source tree.
- [ ] Every notebook in `notebooks/` re-executes end-to-end with fresh outputs committed.
- [ ] `pickle.dumps(ASEEngine(EMT(), ...))` round-trips (asserted in `tests/unit/engine/test_ase.py`).
- [ ] `Engine` Protocol is `@runtime_checkable` and `isinstance(eng, Engine)` returns True for `ASEEngine`.
- [ ] `py.typed` marker exists at `pyiron_workflow_atomistics/py.typed`.
- [ ] Net branch diff is in the −1000 to −2000 LOC range.
