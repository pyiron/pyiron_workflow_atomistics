# phono3py thermal conductivity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `physics/phonons/` subpackage with a single user-facing macro `calculate_phonon_thermal_conductivity(structure, engine, fc2_supercell_matrix, ...)` that returns a `PhononOutput` dataclass with FC2/FC3-derived κ(T), reusing the existing `Engine` Protocol for force evaluations.

**Architecture:** Macro graph of 5 nodes — two parallel displacement-generation + force-evaluation fan-outs (FC2 supercell, FC3 supercell), joined by a synthesis node that rebuilds the `Phono3py` object from the same kwargs, attaches forces, fits FC2/FC3, runs phono3py's BTE solver, and bundles into `PhononOutput`. Lazy imports behind a `_compat.py` module so the `phonopy`/`phono3py`/`symfc` install is an opt-in extra. No Phono3py object crosses node boundaries (it doesn't pickle); determinism between generation and synthesis comes from passing the same construction kwargs plus an auto-resolved `random_seed`.

**Tech Stack:** `pyiron_workflow` (`@pwf.as_function_node`, `@pwf.api.as_macro_node`), ASE `Atoms`, phonopy + phono3py + symfc (optional extras), numpy, pytest.

**Spec:** `docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md`

**Conventions cross-checked against the codebase:**
- `physics/bulk.py` is the canonical reference for "loop `calculate` over perturbed structures" — read `evaluate_structures`, `_extract_energies`, `eos_volume_scan` before starting Task 7.
- Tests live in `tests/unit/physics/test_<topic>.py` and may use the existing `mock_engine_outputs` and `temp_dir` fixtures from `tests/conftest.py`.
- End-to-end macro smoke tests use real ASE EMT and are marked `@pytest.mark.slow` (see `tests/unit/physics/test_bulk_workflows.py`).
- `phono3py` uses `supercell_matrix` for **FC3** and `phonon_supercell_matrix` for **FC2** — opposite of what reading "phonon" might suggest. Every node that constructs a `Phono3py` instance MUST map our `fc2_supercell_matrix` → `phonon_supercell_matrix` and `fc3_supercell_matrix` → `supercell_matrix`. This is the single most likely source of off-by-one bugs.

**One deliberate deviation from the spec:** the spec said `_require_*` shims live "one each in `harmonic.py` and `anharmonic.py`". They're identical, so the plan collapses them into a single `_compat.py` module that both import. DRY beats spec literalism here.

---

## File structure

```
pyiron_workflow_atomistics/physics/phonons/
├── __init__.py          # Task 16 — public re-exports
├── _compat.py           # Task 3 — lazy-import shims
├── output.py            # Task 2 — PhononOutput dataclass
├── harmonic.py          # Tasks 5, 13 — phonopy bits (FC2 supercells, harmonic outputs)
└── anharmonic.py        # Tasks 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15 — the macro and its synthesis node

tests/unit/physics/
└── test_phonons.py      # All Tier 1, 2, 3 tests
```

`samplers.py` is intentionally absent (see spec § Approach: no `DisplacementSampler` Protocol in v1).

---

### Task 1: Add `[phonons]` optional extra and CHANGELOG stub

**Files:**
- Modify: `pyproject.toml:42` (in `[project.optional-dependencies]` block)
- Modify: `CHANGELOG.md` (prepend new top entry)

- [ ] **Step 1: Add the extra to `pyproject.toml`**

Open `pyproject.toml`. Find the existing `[project.optional-dependencies]` block:

```toml
[project.optional-dependencies]
test = [
    "pytest",
    "nbformat",
    "nbclient",
]
```

Replace it with:

```toml
[project.optional-dependencies]
test = [
    "pytest",
    "nbformat",
    "nbclient",
]
phonons = [
    "phonopy",
    "phono3py",
    "symfc",
]
```

No version pins — phono3py's API for `generate_displacements` and `run_thermal_conductivity` has been stable across 2.x and 3.x. If we hit a regression, we pin then.

- [ ] **Step 2: Prepend a draft entry to `CHANGELOG.md`**

The current top entry is `## [0.0.6]`. Prepend (above it):

```markdown
## [Unreleased]

### Added

- **`pyiron_workflow_atomistics.physics.phonons`** — new subpackage for
  phonon properties. The user-facing entry point is
  `calculate_phonon_thermal_conductivity(structure, engine,
  fc2_supercell_matrix, ...)`, which returns a `PhononOutput` dataclass
  containing the lattice thermal conductivity tensor κ(T) plus
  optional mode-resolved data, harmonic side-products, and raw
  force-constant arrays. Reuses the existing `Engine` Protocol — every
  force evaluation goes through `engine.calculate`, no new engine code.
- **`[phonons]` install extra** — `pip install
  pyiron_workflow_atomistics[phonons]` pulls in `phonopy`, `phono3py`,
  and `symfc`. The base install is unaffected; lazy imports keep
  non-phonon users from paying for the extra.

### Out of scope (v2 follow-ups)

- Non-analytic correction (BORN effective charges + ε∞) for polar
  materials. Macro accepts `born_charges` / `epsilon_inf` kwargs and
  raises `NotImplementedError` if either is non-None.
- dynaphopy-based post-MD anharmonic renormalisation.

```

The `[Unreleased]` heading will be renamed to `## [0.0.7] — YYYY-MM-DD` at release time. The spec doesn't ship release tagging — that's a separate manual step after this PR merges.

- [ ] **Step 3: Verify the install resolves**

```bash
cd /home/liger/pyiron_workflow_atomistics
pip install -e ".[test,phonons]"
```

Expected: install completes; `python -c "import phonopy, phono3py, symfc; print('ok')"` prints `ok`.

If a wheel is unavailable for your Python/OS, log the missing wheel and proceed — Tier 2 tests will skip via `pytest.importorskip` and CI can be patched later.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "build: add [phonons] optional install extra

Pulls in phonopy, phono3py, and symfc as opt-in deps for the upcoming
physics/phonons subpackage. Base install is unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `PhononOutput` dataclass

**Files:**
- Create: `pyiron_workflow_atomistics/physics/phonons/__init__.py`
- Create: `pyiron_workflow_atomistics/physics/phonons/output.py`
- Create: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_phonons.py`:

```python
"""Tests for pyiron_workflow_atomistics.physics.phonons.

Tier 1 — cheap unit tests, run always (no phono3py needed).
Tier 2 — gated on `pytest.importorskip("phono3py")`; cover the full graph.
Tier 3 — determinism checks (gated on phono3py when needed).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import numpy as np
import pytest
from ase.build import bulk


# ---------------------------------------------------------------------------
# Tier 1 — PhononOutput dataclass shape
# ---------------------------------------------------------------------------


def test_phonon_output_is_a_dataclass():
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    assert is_dataclass(PhononOutput)


def test_phonon_output_required_fields_have_no_default():
    """Required fields must be the dataclass's positional-required ones."""
    from dataclasses import MISSING

    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    required_names = {
        "structure",
        "fc2_supercell_matrix",
        "fc3_supercell_matrix",
        "temperatures",
        "kappa",
        "converged",
    }
    by_name = {f.name: f for f in fields(PhononOutput)}
    for name in required_names:
        f = by_name[name]
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{name} must be required (no default)"
        )


def test_phonon_output_optional_fields_default_to_none():
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    optional_names = {
        "q_points",
        "frequencies",
        "group_velocities",
        "mode_kappa",
        "gamma",
        "gruneisen",
        "band_structure",
        "dos",
        "free_energy",
        "fc2",
        "fc3",
        "phono3py",
    }
    by_name = {f.name: f for f in fields(PhononOutput)}
    for name in optional_names:
        assert by_name[name].default is None, f"{name} must default to None"


def test_phonon_output_to_dict_round_trip():
    """to_dict() returns plain dict of all fields (per EngineOutput convention)."""
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = PhononOutput(
        structure=cu,
        fc2_supercell_matrix=np.eye(3, dtype=int) * 2,
        fc3_supercell_matrix=np.eye(3, dtype=int) * 2,
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
    )
    d = out.to_dict()
    assert d["temperatures"][0] == 300.0
    assert d["kappa"].shape == (1, 3, 3)
    assert d["converged"] is True
    assert d["q_points"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/liger/pyiron_workflow_atomistics
pytest tests/unit/physics/test_phonons.py -v
```

Expected: 4 collection errors with `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.physics.phonons'`.

- [ ] **Step 3: Create the empty subpackage `__init__.py`**

Create `pyiron_workflow_atomistics/physics/phonons/__init__.py` with a single placeholder line (full re-exports land in Task 16):

```python
"""Phonon workflows (phonopy harmonic side-products + phono3py FC3 + κ(T)).

Public API is filled in by ``__init__.py`` once the macro lands (see Task 16).
"""
```

- [ ] **Step 4: Create `output.py`**

Create `pyiron_workflow_atomistics/physics/phonons/output.py`:

```python
"""PhononOutput dataclass — the structured result of a phonon workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ase import Atoms


@dataclass
class PhononOutput:
    """Structured result of a phonon thermal-conductivity calculation.

    Required fields are always populated. Optional fields are populated only
    when the corresponding macro flag is on:
        mode_resolved=True       → q_points, frequencies, group_velocities,
                                   mode_kappa, gamma, gruneisen
        harmonic_observables=True → band_structure, dos, free_energy
        keep_handles=True        → fc2, fc3, phono3py
    """

    structure: Atoms
    fc2_supercell_matrix: np.ndarray  # (3, 3) int
    fc3_supercell_matrix: np.ndarray  # (3, 3) int
    temperatures: np.ndarray  # (n_T,) K
    kappa: np.ndarray  # (n_T, 3, 3) W/m·K
    converged: bool

    q_points: np.ndarray | None = None  # (n_q, 3) reduced
    frequencies: np.ndarray | None = None  # (n_q, n_band) THz
    group_velocities: np.ndarray | None = None  # (n_q, n_band, 3)
    mode_kappa: np.ndarray | None = None  # (n_T, n_q, n_band, 6) Voigt
    gamma: np.ndarray | None = None  # (n_T, n_q, n_band) linewidths
    gruneisen: np.ndarray | None = None  # (n_q, n_band)

    band_structure: dict | None = None
    dos: dict | None = None
    free_energy: dict | None = None

    fc2: np.ndarray | None = None
    fc3: np.ndarray | None = None
    phono3py: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every field (ASE/phono3py objects by reference)."""
        return asdict(self)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/physics/test_phonons.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/__init__.py \
        pyiron_workflow_atomistics/physics/phonons/output.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): PhononOutput dataclass

Mirrors the EngineOutput pattern: required fields up front, optional
fields default None, to_dict() round-trips. The three opt-in output
tiers (mode_resolved, harmonic_observables, keep_handles) are
documented but not yet populated — that lands with the synthesis node.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Lazy-import shims in `_compat.py`

**Files:**
- Create: `pyiron_workflow_atomistics/physics/phonons/_compat.py`
- Modify: `tests/unit/physics/test_phonons.py` (append Tier 1 ImportError tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — lazy-import shims
# ---------------------------------------------------------------------------


def _patch_missing(monkeypatch, module_name):
    """Make `import <module_name>` raise ImportError inside the shim."""
    import sys

    monkeypatch.setitem(sys.modules, module_name, None)


def test_require_phonopy_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "phonopy")
    with pytest.raises(ImportError) as exc:
        _compat.require_phonopy()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons]" in msg
    assert "phonopy" in msg


def test_require_phono3py_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "phono3py")
    with pytest.raises(ImportError) as exc:
        _compat.require_phono3py()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons]" in msg
    assert "phono3py" in msg


def test_require_symfc_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "symfc")
    with pytest.raises(ImportError) as exc:
        _compat.require_symfc()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons]" in msg
    assert "symfc" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/physics/test_phonons.py::test_require_phono3py_missing_actionable -v
```

Expected: `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.physics.phonons._compat'`.

- [ ] **Step 3: Create `_compat.py`**

Create `pyiron_workflow_atomistics/physics/phonons/_compat.py`:

```python
"""Lazy-import shims for the optional phonopy / phono3py / symfc stack.

These let `import pyiron_workflow_atomistics.physics.phonons` succeed in
environments where the `[phonons]` extra isn't installed. The check fires
only when a workflow that needs the optional dep is actually invoked.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = "pip install pyiron_workflow_atomistics[phonons]"


def _require(module_name: str) -> Any:
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"{module_name} is required for this workflow. Install with: {_INSTALL_HINT}"
        ) from e


def require_phonopy() -> Any:
    """Return the imported phonopy module or raise an actionable ImportError."""
    return _require("phonopy")


def require_phono3py() -> Any:
    """Return the imported phono3py module or raise an actionable ImportError."""
    return _require("phono3py")


def require_symfc() -> Any:
    """Return the imported symfc module or raise an actionable ImportError.

    Only used when fc_calculator='symfc' (random-displacement mode).
    """
    return _require("symfc")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "missing_actionable"
```

Expected: 3 passed. The four Task 2 tests also still pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/_compat.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): lazy-import shims with actionable ImportError

require_phonopy / require_phono3py / require_symfc each raise
ImportError pointing at the [phonons] extra when the dep is missing.
Lets the base install stay lean; users who never call a phonon macro
never pay the import cost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: NotImplementedError on `born_charges` / `epsilon_inf`

This task stubs the user-facing macro's signature far enough to enforce the polar-material early exit. The rest of the body lands in later tasks.

**Files:**
- Create: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — polar-material kwargs early exit
# ---------------------------------------------------------------------------


def test_born_charges_raises_not_implemented():
    """Passing born_charges raises before any phono3py import."""
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError) as exc:
        _check_polar_unsupported(
            born_charges=np.zeros((4, 3, 3)), epsilon_inf=None
        )
    msg = str(exc.value)
    assert "BORN" in msg or "Non-analytic" in msg
    assert "v1" in msg


def test_epsilon_inf_raises_not_implemented():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError):
        _check_polar_unsupported(
            born_charges=None, epsilon_inf=np.eye(3)
        )


def test_no_polar_kwargs_returns_silently():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    # Should return without raising
    _check_polar_unsupported(born_charges=None, epsilon_inf=None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/physics/test_phonons.py::test_born_charges_raises_not_implemented -v
```

Expected: `ModuleNotFoundError` or `ImportError: cannot import _check_polar_unsupported`.

- [ ] **Step 3: Create `anharmonic.py` with the early-exit guard**

Create `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`:

```python
"""phono3py FC3 + lattice thermal conductivity workflow.

The single user-facing entry point is :func:`calculate_phonon_thermal_conductivity`.
Everything else in this module is a private node or helper.
"""

from __future__ import annotations

import numpy as np


def _check_polar_unsupported(
    *,
    born_charges: np.ndarray | None,
    epsilon_inf: np.ndarray | None,
) -> None:
    """Raise NotImplementedError if the caller asked for polar-material support.

    v1 is metals/non-polar only. The follow-up that adds NAC is tracked under
    "NAC / BORN effective charges" in the design spec.
    """
    if born_charges is not None or epsilon_inf is not None:
        raise NotImplementedError(
            "Non-analytic correction (BORN + ε∞) is not supported in v1; "
            "see the 'NAC / BORN effective charges' follow-up at the end of "
            "docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "polar or epsilon_inf or born_charges"
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): NotImplementedError on polar-material kwargs

born_charges or epsilon_inf passed to v1 raises NotImplementedError
before any phono3py import. Auto-detection of polar materials is
deliberately not attempted (brittle); the signal is the user passing
the kwargs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Displacement-supercell generation nodes (FD only)

Random-mode is added in Task 11; this task locks in finite-difference (the default phono3py path).

**Files:**
- Create: `pyiron_workflow_atomistics/physics/phonons/harmonic.py`
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing determinism tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 3 — displacement generation determinism (gated)
# ---------------------------------------------------------------------------


phono3py = pytest.importorskip("phono3py", reason="phonons extra not installed")


def _cu_fcc_primitive():
    return bulk("Cu", "fcc", a=3.6)


def _two_by_two_by_two():
    return (2 * np.eye(3)).astype(int)


@pytest.mark.slow
def test_fd_fc2_supercells_deterministic():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    a = _generate_fc2_supercells.node_function(
        structure=_cu_fcc_primitive(),
        fc2_supercell_matrix=_two_by_two_by_two(),
        displacement_distance=0.03,
        is_plusminus="auto",
    )
    b = _generate_fc2_supercells.node_function(
        structure=_cu_fcc_primitive(),
        fc2_supercell_matrix=_two_by_two_by_two(),
        displacement_distance=0.03,
        is_plusminus="auto",
    )
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.get_positions(), y.get_positions())
        np.testing.assert_allclose(x.get_cell()[:], y.get_cell()[:])


@pytest.mark.slow
def test_fd_fc3_supercells_deterministic():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _generate_fc3_supercells,
    )

    kwargs = dict(
        structure=_cu_fcc_primitive(),
        fc2_supercell_matrix=_two_by_two_by_two(),
        fc3_supercell_matrix=_two_by_two_by_two(),
        displacement_distance=0.03,
        is_plusminus="auto",
        cutoff_pair_distance=None,
        number_of_snapshots=None,
        random_seed=None,
    )
    a = _generate_fc3_supercells.node_function(**kwargs)
    b = _generate_fc3_supercells.node_function(**kwargs)
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.get_positions(), y.get_positions())
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/physics/test_phonons.py::test_fd_fc2_supercells_deterministic -v
```

Expected: `ImportError: cannot import _generate_fc2_supercells from ... .harmonic` (skips if phono3py not installed).

- [ ] **Step 3: Create `harmonic.py` with the FC2 node + ASE/PhonopyAtoms helpers**

Create `pyiron_workflow_atomistics/physics/phonons/harmonic.py`:

```python
"""phonopy FC2 helpers: supercell generation + ASE/PhonopyAtoms conversion.

Harmonic-observable nodes (band structure, DOS, free energy) land in
Task 13 once the synthesis node exists to expose them.
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.physics.phonons._compat import (
    require_phono3py,
    require_phonopy,
)


def _normalise_supercell_matrix(m: ArrayLike) -> np.ndarray:
    """Accept int / list[int] of length 3 / (3,3) ndarray; return (3,3) int."""
    arr = np.asarray(m)
    if arr.ndim == 0:
        return (int(arr) * np.eye(3, dtype=int))
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
    raise ValueError(f"supercell_matrix must be int / (3,) / (3,3); got {arr.shape}")


def _ase_to_phonopy(ase_atoms: Atoms):
    """Convert ASE Atoms → PhonopyAtoms (phonopy's own structure type)."""
    require_phonopy()  # noqa: F841 — only needed for the import side-effect
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=list(ase_atoms.get_chemical_symbols()),
        positions=ase_atoms.get_positions(),
        cell=np.asarray(ase_atoms.get_cell()),
        masses=ase_atoms.get_masses(),
    )


def _phonopy_to_ase(phonopy_atoms) -> Atoms:
    """Convert PhonopyAtoms → ASE Atoms. pbc=True (supercells are always periodic)."""
    return Atoms(
        symbols=list(phonopy_atoms.symbols),
        positions=np.asarray(phonopy_atoms.positions),
        cell=np.asarray(phonopy_atoms.cell),
        pbc=True,
        masses=np.asarray(phonopy_atoms.masses),
    )


def _build_phono3py(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
):
    """Construct a Phono3py instance with both supercell matrices.

    Note: phono3py's `supercell_matrix` is the FC3 supercell and
    `phonon_supercell_matrix` is the FC2 supercell. We expose them under the
    physics-level names (`fc2_*`, `fc3_*`) and translate here.
    """
    phono3py_mod = require_phono3py()
    unitcell = _ase_to_phonopy(structure)
    return phono3py_mod.Phono3py(
        unitcell=unitcell,
        supercell_matrix=_normalise_supercell_matrix(fc3_supercell_matrix),
        phonon_supercell_matrix=_normalise_supercell_matrix(fc2_supercell_matrix),
    )


@pwf.as_function_node("fc2_supercells")
def _generate_fc2_supercells(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    displacement_distance: float = 0.03,
    is_plusminus: str | bool = "auto",
) -> list[Atoms]:
    """FC2 displaced supercells via phono3py.generate_fc2_displacements (FD).

    Returns a list of ASE Atoms. The same kwargs reconstruct an identical
    Phono3py object inside the synthesis node — FD is deterministic in
    structure + supercell + distance + symmetry.
    """
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc2_supercell_matrix,  # placeholder; FC3 grid not used here
    )
    ph3.generate_fc2_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    fc2_supercells = [
        _phonopy_to_ase(s) for s in ph3.phonon_supercells_with_displacements
    ]
    return fc2_supercells
```

- [ ] **Step 4: Add the FC3 generation node to `anharmonic.py`**

Append to `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`:

```python
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _build_phono3py,
    _phonopy_to_ase,
)


@pwf.as_function_node("fc3_supercells")
def _generate_fc3_supercells(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
    displacement_distance: float = 0.03,
    is_plusminus: str | bool = "auto",
    cutoff_pair_distance: float | None = None,
    number_of_snapshots: int | None = None,
    random_seed: int | None = None,
) -> list[Atoms]:
    """FC3 displaced supercells via phono3py.generate_displacements.

    Finite-difference path is used when number_of_snapshots is None;
    random-displacement path (and symfc fitting) lands in Task 11.
    """
    if number_of_snapshots is not None:
        # Filled in by Task 11. Until then, refuse random mode loudly so a
        # user who passes the kwarg too early gets a clear message.
        raise NotImplementedError(
            "Random-displacement FC3 sampling is added in a later task; "
            "set number_of_snapshots=None for the FD path."
        )
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )
    fc3_supercells = [
        _phonopy_to_ase(s) for s in ph3.supercells_with_displacements
    ]
    return fc3_supercells
```

- [ ] **Step 5: Run the determinism tests**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "deterministic"
```

Expected (with `[phonons]` extra installed): 2 passed. (Skipped if not installed.)

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/harmonic.py \
        pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): FC2/FC3 finite-difference supercell generation

Two function nodes (one in harmonic.py for the FC2 supercell, one in
anharmonic.py for the FC3 supercell) and shared helpers for
ASE↔PhonopyAtoms conversion + Phono3py construction. Random-mode
displacement is gated behind an explicit NotImplementedError until
Task 11 wires it up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `_evaluate_supercells` force-eval fan-out

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — _evaluate_supercells using mock engine
# ---------------------------------------------------------------------------


def test_evaluate_supercells_uses_with_working_directory(tmp_path):
    """Each supercell gets its own engine subdir; the node returns one
    EngineOutput per input supercell."""
    from dataclasses import replace as dc_replace

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    supercells = [cu.copy(), cu.copy(), cu.copy()]
    outs = _evaluate_supercells.node_function(
        supercells=supercells, engine=engine, prefix="fc2_disp_"
    )
    assert len(outs) == 3
    assert all(o.converged for o in outs)
    # Each supercell got its own working_directory under tmp_path
    for i in range(3):
        assert (tmp_path / f"fc2_disp_{i:04d}").exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_evaluate_supercells_uses_with_working_directory -v
```

Expected: `ImportError: cannot import _evaluate_supercells`.

- [ ] **Step 3: Add `_evaluate_supercells` to `anharmonic.py`**

Append to `anharmonic.py`:

```python
from pyiron_workflow_atomistics.engine import Engine, EngineOutput, calculate


@pwf.as_function_node("engine_outputs")
def _evaluate_supercells(
    supercells: list[Atoms],
    engine: Engine,
    prefix: str,
) -> list[EngineOutput]:
    """Loop ``calculate`` over a list of supercells, routing each to its own subdir.

    Mirrors ``physics.bulk.evaluate_structures`` — the canonical "fan out
    `calculate` over a list of structures" pattern in this codebase.
    """
    engine_outputs: list[EngineOutput] = []
    for i, supercell in enumerate(supercells):
        sub_engine = engine.with_working_directory(f"{prefix}{i:04d}")
        engine_outputs.append(
            calculate.node_function(structure=supercell, engine=sub_engine)
        )
    return engine_outputs
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/unit/physics/test_phonons.py::test_evaluate_supercells_uses_with_working_directory -v
```

Expected: pass. ~5–10 seconds (3 EMT static calcs).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): _evaluate_supercells force-eval fan-out

Mirrors physics.bulk.evaluate_structures: loops calculate.node_function
over a list of supercells, routing each through
engine.with_working_directory(f'{prefix}{i:04d}'). Only node in the
phonon graph that touches the engine.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `_run_phono3py_thermal_conductivity` synthesis node — required outputs

This is the workhorse. Required fields only; optional output tiers land in Tasks 12–14.

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — synthesis-node smoke (EMT-Cu, ~60s)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_phono3py_thermal_conductivity_emt_smoke(tmp_path):
    """End-to-end smoke through the synthesis node only (skipping macro plumbing).

    Calls the FC2/FC3 generation + evaluate nodes manually, hands the
    EngineOutputs to the synthesis node, and asserts a sensible
    PhononOutput comes out. ~60s with EMT.
    """
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
        _generate_fc3_supercells,
        _run_phono3py_thermal_conductivity,
    )
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )

    fc2_supercells = _generate_fc2_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc
    )
    fc3_supercells = _generate_fc3_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc, fc3_supercell_matrix=sc
    )
    fc2_outs = _evaluate_supercells.node_function(
        supercells=fc2_supercells, engine=engine, prefix="fc2_disp_"
    )
    fc3_outs = _evaluate_supercells.node_function(
        supercells=fc3_supercells, engine=engine, prefix="fc3_disp_"
    )
    out = _run_phono3py_thermal_conductivity.node_function(
        structure=cu,
        fc2_supercell_matrix=sc,
        fc3_supercell_matrix=sc,
        displacement_distance=0.03,
        is_plusminus="auto",
        cutoff_pair_distance=None,
        number_of_snapshots=None,
        random_seed=None,
        fc_calculator=None,
        fc2_engine_outputs=fc2_outs,
        fc3_engine_outputs=fc3_outs,
        temperatures=np.array([300.0]),
        q_mesh=(5, 5, 5),
        mode_resolved=False,
        harmonic_observables=False,
        keep_handles=False,
    )

    assert out.converged is True
    assert out.kappa.shape == (1, 3, 3)
    # Diagonal κ should be positive (BTE), trace > 0.
    diag = np.array([out.kappa[0, i, i] for i in range(3)])
    assert (diag > 0).all(), f"Non-positive diagonal κ: {diag}"
    # Optional fields all None at this tier.
    assert out.q_points is None
    assert out.band_structure is None
    assert out.fc2 is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_run_phono3py_thermal_conductivity_emt_smoke -v
```

Expected: `ImportError: cannot import _run_phono3py_thermal_conductivity`.

- [ ] **Step 3: Implement the synthesis node**

Append to `anharmonic.py`:

```python
import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _build_phono3py,
    _normalise_supercell_matrix,
)
from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput


def _check_all_converged(engine_outputs, label: str) -> None:
    """Raise RuntimeError listing failed supercell indices + working_directory."""
    failed = [
        (i, getattr(out.final_structure, "info", {}).get("working_directory", "<unknown>"))
        for i, out in enumerate(engine_outputs)
        if not out.converged
    ]
    if failed:
        details = ", ".join(f"{i} ({wd})" for i, wd in failed)
        raise RuntimeError(
            f"Force calc failed for {label} supercells: {details}"
        )


def _stack_forces(engine_outputs) -> np.ndarray:
    """(n_supercells, n_atoms, 3) — phono3py's expected forces layout."""
    return np.stack([np.asarray(o.final_forces) for o in engine_outputs], axis=0)


def _kappa_voigt_to_tensor(kappa_voigt: np.ndarray) -> np.ndarray:
    """Convert (n_T, 6) Voigt → (n_T, 3, 3) full tensor.

    phono3py returns κ as (n_T, 6) in (xx, yy, zz, yz, xz, xy) order.
    """
    n_T = kappa_voigt.shape[0]
    out = np.zeros((n_T, 3, 3))
    for t in range(n_T):
        xx, yy, zz, yz, xz, xy = kappa_voigt[t]
        out[t] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return out


@pwf.as_function_node("phonon_output")
def _run_phono3py_thermal_conductivity(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike,
    displacement_distance: float,
    is_plusminus: str | bool,
    cutoff_pair_distance: float | None,
    number_of_snapshots: int | None,
    random_seed: int | None,
    fc_calculator: str | None,
    fc2_engine_outputs: list,
    fc3_engine_outputs: list,
    temperatures: ArrayLike,
    q_mesh: ArrayLike,
    mode_resolved: bool,
    harmonic_observables: bool,
    keep_handles: bool,
) -> PhononOutput:
    """Synthesis node: rebuild Phono3py, attach forces, fit FCs, run BTE."""
    _check_all_converged(fc2_engine_outputs, label="FC2")
    _check_all_converged(fc3_engine_outputs, label="FC3")

    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    # Re-generate displacements identically so the dataset matches the forces.
    ph3.generate_fc2_displacements(
        distance=displacement_distance, is_plusminus=is_plusminus
    )
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )

    fc2_forces = _stack_forces(fc2_engine_outputs)
    fc3_forces = _stack_forces(fc3_engine_outputs)
    if fc2_forces.shape[0] != len(ph3.phonon_supercells_with_displacements):
        raise RuntimeError(
            f"FC2 force/supercell mismatch: {fc2_forces.shape[0]} forces vs "
            f"{len(ph3.phonon_supercells_with_displacements)} expected. "
            "Displacement kwargs likely drifted between generation and synthesis."
        )
    if fc3_forces.shape[0] != len(ph3.supercells_with_displacements):
        raise RuntimeError(
            f"FC3 force/supercell mismatch: {fc3_forces.shape[0]} forces vs "
            f"{len(ph3.supercells_with_displacements)} expected. "
            "Displacement kwargs likely drifted between generation and synthesis."
        )

    ph3.phonon_forces = fc2_forces
    ph3.forces = fc3_forces
    ph3.produce_fc2()
    ph3.produce_fc3(fc_calculator=fc_calculator)

    T = np.asarray(temperatures, dtype=float)
    mesh = np.asarray(q_mesh, dtype=int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ph3.mesh_numbers = mesh
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=T, write_kappa=False)
        converged = not any(
            "not converged" in str(w.message).lower() for w in caught
        )

    tc = ph3.thermal_conductivity
    kappa = _kappa_voigt_to_tensor(np.asarray(tc.kappa[0]))  # (n_T, 3, 3)

    return PhononOutput(
        structure=structure,
        fc2_supercell_matrix=_normalise_supercell_matrix(fc2_supercell_matrix),
        fc3_supercell_matrix=_normalise_supercell_matrix(fc3_supercell_matrix),
        temperatures=T,
        kappa=kappa,
        converged=converged,
    )
```

- [ ] **Step 4: Run the smoke test**

```bash
pytest tests/unit/physics/test_phonons.py::test_run_phono3py_thermal_conductivity_emt_smoke -v
```

Expected: pass. Budget ~60-120s. If it fails on phono3py API mismatch (the `tc.kappa[0]` indexing or `mesh_numbers` setter), check the installed phono3py version against its docs and adapt; the test then becomes the regression guard.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): synthesis node — produce FC2/FC3, run BTE, return PhononOutput

Rebuilds Phono3py from the same kwargs that drove displacement
generation, asserts all force calcs converged, attaches forces, fits
FC2 and FC3, runs thermal_conductivity on the requested q-mesh, and
bundles into PhononOutput (required fields only — optional tiers land
in later tasks).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Error guards — non-convergence and supercell count mismatch

These guards already exist in Task 7's synthesis node; this task adds the tests that lock them in.

**Files:**
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — error guards (gated; need phono3py for _build_phono3py to work)
# ---------------------------------------------------------------------------


def _make_fake_engine_output(*, converged: bool, n_atoms: int = 32):
    """Minimal EngineOutput-shaped object for testing the synthesis-node guards."""
    from pyiron_workflow_atomistics.engine import EngineOutput

    return EngineOutput(
        final_structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        final_energy=-1.0,
        converged=converged,
        final_forces=np.zeros((n_atoms, 3)),
    )


@pytest.mark.slow
def test_synthesis_raises_when_force_calc_failed():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _run_phono3py_thermal_conductivity,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)

    # Build minimal fake FC2 / FC3 force lists with one failed entry at index 3.
    fc2_outs = [_make_fake_engine_output(converged=(i != 3)) for i in range(6)]
    fc3_outs = [_make_fake_engine_output(converged=True) for _ in range(2)]

    with pytest.raises(RuntimeError) as exc:
        _run_phono3py_thermal_conductivity.node_function(
            structure=cu,
            fc2_supercell_matrix=sc,
            fc3_supercell_matrix=sc,
            displacement_distance=0.03,
            is_plusminus="auto",
            cutoff_pair_distance=None,
            number_of_snapshots=None,
            random_seed=None,
            fc_calculator=None,
            fc2_engine_outputs=fc2_outs,
            fc3_engine_outputs=fc3_outs,
            temperatures=np.array([300.0]),
            q_mesh=(5, 5, 5),
            mode_resolved=False,
            harmonic_observables=False,
            keep_handles=False,
        )
    msg = str(exc.value)
    assert "FC2" in msg
    assert "3" in msg  # the failed index


@pytest.mark.slow
def test_synthesis_raises_on_supercell_force_mismatch():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
        _generate_fc3_supercells,
        _run_phono3py_thermal_conductivity,
    )
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)

    fc2_supercells = _generate_fc2_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc
    )
    fc3_supercells = _generate_fc3_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc, fc3_supercell_matrix=sc
    )
    n_fc2 = len(fc2_supercells[0])
    n_fc3 = len(fc3_supercells[0])
    fc2_outs = [
        _make_fake_engine_output(converged=True, n_atoms=n_fc2)
        for _ in range(len(fc2_supercells) - 1)  # ← one too few
    ]
    fc3_outs = [
        _make_fake_engine_output(converged=True, n_atoms=n_fc3)
        for _ in fc3_supercells
    ]

    with pytest.raises(RuntimeError) as exc:
        _run_phono3py_thermal_conductivity.node_function(
            structure=cu,
            fc2_supercell_matrix=sc,
            fc3_supercell_matrix=sc,
            displacement_distance=0.03,
            is_plusminus="auto",
            cutoff_pair_distance=None,
            number_of_snapshots=None,
            random_seed=None,
            fc_calculator=None,
            fc2_engine_outputs=fc2_outs,
            fc3_engine_outputs=fc3_outs,
            temperatures=np.array([300.0]),
            q_mesh=(5, 5, 5),
            mode_resolved=False,
            harmonic_observables=False,
            keep_handles=False,
        )
    msg = str(exc.value)
    assert "FC2 force/supercell mismatch" in msg
    assert str(len(fc2_supercells) - 1) in msg
    assert str(len(fc2_supercells)) in msg
```

- [ ] **Step 2: Run and verify they pass**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "force_calc_failed or supercell_force_mismatch"
```

Expected: 2 passed (no implementation changes needed — Task 7 already implemented the guards).

If they fail because the error message wording drifted from Task 7, fix the message in `anharmonic.py` first.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/physics/test_phonons.py
git commit -m "test(phonons): lock in synthesis-node error guards

Two slow tests covering (1) non-converged supercell force calcs surface
the failed indices in the RuntimeError message, and (2) a force-list /
supercell-count mismatch raises with both counts named. These guard
against the only silent-corruption failure mode the rebuild-from-kwargs
trick can produce.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: User-facing `calculate_phonon_thermal_conductivity` macro

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing macro-level smoke test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — full-macro smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calculate_phonon_thermal_conductivity_macro_emt(tmp_path):
    """End-to-end through the public macro. ~60s with EMT."""
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    out = calculate_phonon_thermal_conductivity(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=sc,
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
    ).run()

    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.converged is True
    assert out.kappa.shape == (1, 3, 3)
    # Engine got the per-supercell subdirs
    assert (tmp_path / "fc2_disp_0000").exists()
    assert (tmp_path / "fc3_disp_0000").exists()
```

Note on the `.run()` and `out["phonon_output"]` shape: pyiron_workflow macros are invoked the same way `bulk.eos_volume_scan` is invoked in `tests/unit/physics/test_bulk_workflows.py` — re-read that test if the call pattern doesn't compile.

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_calculate_phonon_thermal_conductivity_macro_emt -v
```

Expected: `ImportError: cannot import calculate_phonon_thermal_conductivity`.

- [ ] **Step 3: Implement the macro**

Append to `anharmonic.py`:

```python
from typing import Literal


@pwf.api.as_macro_node("phonon_output")
def calculate_phonon_thermal_conductivity(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix: ArrayLike,
    fc3_supercell_matrix: ArrayLike | None = None,
    temperatures: ArrayLike = (300.0,),
    q_mesh: ArrayLike = (11, 11, 11),
    # phono3py.generate_displacements kwargs
    displacement_distance: float = 0.03,
    is_plusminus: bool | Literal["auto"] = "auto",
    cutoff_pair_distance: float | None = None,
    number_of_snapshots: int | None = None,
    random_seed: int | None = None,
    fc_calculator: str | None = None,
    # output tiers
    mode_resolved: bool = False,
    harmonic_observables: bool = False,
    keep_handles: bool = False,
    # polar-material kwargs (v1: must be None)
    born_charges: np.ndarray | None = None,
    epsilon_inf: np.ndarray | None = None,
):
    """Compute lattice thermal conductivity κ(T) via phono3py.

    Reuses the existing Engine Protocol — every supercell force evaluation
    goes through ``engine.calculate``. Returns a :class:`PhononOutput`.

    See spec: docs/design/specs/2026-05-13-phono3py-thermal-conductivity-design.md
    """
    _check_polar_unsupported(born_charges=born_charges, epsilon_inf=epsilon_inf)

    # Default FC3 supercell to FC2 supercell.
    if fc3_supercell_matrix is None:
        fc3_supercell_matrix = fc2_supercell_matrix

    # Dispatch random-mode default solver.
    if number_of_snapshots is not None and fc_calculator is None:
        fc_calculator = "symfc"

    wf.fc2_supercells = _generate_fc2_supercells(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    wf.fc3_supercells = _generate_fc3_supercells(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
    )
    wf.fc2_eval = _evaluate_supercells(
        supercells=wf.fc2_supercells.outputs.fc2_supercells,
        engine=engine,
        prefix="fc2_disp_",
    )
    wf.fc3_eval = _evaluate_supercells(
        supercells=wf.fc3_supercells.outputs.fc3_supercells,
        engine=engine,
        prefix="fc3_disp_",
    )
    wf.synthesis = _run_phono3py_thermal_conductivity(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
        fc_calculator=fc_calculator,
        fc2_engine_outputs=wf.fc2_eval.outputs.engine_outputs,
        fc3_engine_outputs=wf.fc3_eval.outputs.engine_outputs,
        temperatures=temperatures,
        q_mesh=q_mesh,
        mode_resolved=mode_resolved,
        harmonic_observables=harmonic_observables,
        keep_handles=keep_handles,
    )

    return wf.synthesis.outputs.phonon_output
```

The child binding is named `wf.synthesis` (not `wf.phonon_output`) so it doesn't collide with the macro-level declared output port `"phonon_output"`. The return uses the explicit `.outputs.<port>` form, matching the convention in `physics/bulk.py:optimise_cubic_lattice_parameter`.

- [ ] **Step 4: Run the macro smoke test**

```bash
pytest tests/unit/physics/test_phonons.py::test_calculate_phonon_thermal_conductivity_macro_emt -v
```

Expected: pass. Budget ~60–120s.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): user-facing calculate_phonon_thermal_conductivity macro

Wires the five-node graph: FC2 + FC3 displacement generation (parallel),
FC2 + FC3 force-eval fan-out (parallel), and the synthesis node. Dispatches
fc_calculator='symfc' if user picked random mode without overriding. Polar
kwargs raise NotImplementedError before any phono3py import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Random-mode seed auto-resolution

Per spec § Approach (decision 3): if random mode is on and `random_seed is None`, resolve it once at macro entry and thread it through both generation and synthesis.

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — seed auto-resolution helper
# ---------------------------------------------------------------------------


def test_resolve_random_seed_passthrough_when_explicit():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    assert _resolve_random_seed(number_of_snapshots=10, random_seed=42) == 42
    assert _resolve_random_seed(number_of_snapshots=None, random_seed=None) is None
    assert _resolve_random_seed(number_of_snapshots=None, random_seed=7) == 7


def test_resolve_random_seed_auto_fills_when_random_mode_without_seed():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    seed = _resolve_random_seed(number_of_snapshots=10, random_seed=None)
    assert seed is not None
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "resolve_random_seed"
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `_resolve_random_seed`**

Append to `anharmonic.py`:

```python
def _resolve_random_seed(
    *,
    number_of_snapshots: int | None,
    random_seed: int | None,
) -> int | None:
    """Auto-fill the random seed when random mode is on but user didn't pick one.

    Without this, generation and synthesis would draw fresh randomness on
    each rebuild of the Phono3py object → identical supercell counts but
    different positions → silent corruption (the count guard in
    _run_phono3py_thermal_conductivity catches mismatched counts, not
    mismatched positions).
    """
    if number_of_snapshots is None:
        return random_seed
    if random_seed is not None:
        return random_seed
    return int(np.random.SeedSequence().entropy % (2**32))
```

Use this in the macro: compute `resolved_seed` once at macro entry and thread it through the two nodes that take a seed (`_generate_fc3_supercells` and `_run_phono3py_thermal_conductivity`).

In the macro body in `anharmonic.py`, after the `if number_of_snapshots is not None and fc_calculator is None: ...` block, add:

```python
    resolved_seed = _resolve_random_seed(
        number_of_snapshots=number_of_snapshots, random_seed=random_seed
    )
```

Then change `random_seed=random_seed,` to `random_seed=resolved_seed,` in both child-node calls:

```python
    wf.fc3_supercells = _generate_fc3_supercells(
        ...
        random_seed=resolved_seed,
    )
    wf.synthesis = _run_phono3py_thermal_conductivity(
        ...
        random_seed=resolved_seed,
        ...
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "resolve_random_seed"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): auto-resolve random_seed at macro entry

Random-mode determinism between displacement generation and synthesis
depends on the same seed being threaded through both rebuilds of the
Phono3py object. Auto-fills random_seed via SeedSequence().entropy when
number_of_snapshots is set but user didn't pick one — closes the silent-
corruption hole the count guard can't see.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Random-displacement mode (symfc fitter)

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py` (drop the `NotImplementedError` stub in `_generate_fc3_supercells`, add the random-mode branch)
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 3 — random-displacement determinism (gated additionally on symfc)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_random_fc3_supercells_deterministic_with_seed():
    pytest.importorskip("symfc", reason="symfc not installed")
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _generate_fc3_supercells,
    )

    kwargs = dict(
        structure=bulk("Cu", "fcc", a=3.6),
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        fc3_supercell_matrix=(2 * np.eye(3)).astype(int),
        displacement_distance=0.03,
        is_plusminus="auto",
        cutoff_pair_distance=None,
        number_of_snapshots=10,
        random_seed=42,
    )
    a = _generate_fc3_supercells.node_function(**kwargs)
    b = _generate_fc3_supercells.node_function(**kwargs)
    assert len(a) == len(b) == 10
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.get_positions(), y.get_positions())


# ---------------------------------------------------------------------------
# Tier 2 — random-mode end-to-end smoke (gated additionally on symfc)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_random_displacement_macro_emt(tmp_path):
    pytest.importorskip("symfc", reason="symfc not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=sc,
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
        number_of_snapshots=20,
        random_seed=0,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.converged is True
    assert np.all(np.isfinite(out.kappa))
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/physics/test_phonons.py::test_random_fc3_supercells_deterministic_with_seed -v
```

Expected: fails with `NotImplementedError: Random-displacement FC3 sampling is added in a later task...`.

- [ ] **Step 3: Replace the random-mode stub in `_generate_fc3_supercells`**

In `anharmonic.py`, find this block (added in Task 5):

```python
    if number_of_snapshots is not None:
        # Filled in by Task 11. Until then, refuse random mode loudly so a
        # user who passes the kwarg too early gets a clear message.
        raise NotImplementedError(
            "Random-displacement FC3 sampling is added in a later task; "
            "set number_of_snapshots=None for the FD path."
        )
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )
```

Replace with:

```python
    ph3 = _build_phono3py(
        structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc3_supercell_matrix=fc3_supercell_matrix,
    )
    if number_of_snapshots is not None:
        ph3.generate_displacements(
            distance=displacement_distance,
            number_of_snapshots=number_of_snapshots,
            random_seed=random_seed,
        )
    else:
        ph3.generate_displacements(
            distance=displacement_distance,
            is_plusminus=is_plusminus,
            cutoff_pair_distance=cutoff_pair_distance,
        )
```

In `_run_phono3py_thermal_conductivity`, find:

```python
    ph3.generate_displacements(
        distance=displacement_distance,
        is_plusminus=is_plusminus,
        cutoff_pair_distance=cutoff_pair_distance,
    )
```

Replace with:

```python
    if number_of_snapshots is not None:
        ph3.generate_displacements(
            distance=displacement_distance,
            number_of_snapshots=number_of_snapshots,
            random_seed=random_seed,
        )
    else:
        ph3.generate_displacements(
            distance=displacement_distance,
            is_plusminus=is_plusminus,
            cutoff_pair_distance=cutoff_pair_distance,
        )
```

Also gate the symfc import: at the top of `_run_phono3py_thermal_conductivity`, after the convergence checks, add:

```python
    if fc_calculator == "symfc":
        from pyiron_workflow_atomistics.physics.phonons._compat import require_symfc

        require_symfc()
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "random"
```

Expected: 2 passed (or 2 skipped if symfc isn't installed).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): random-displacement FC3 sampling via symfc

Replaces the NotImplementedError stub with the actual random-mode
branch in both _generate_fc3_supercells (sampling) and the synthesis
node (re-sampling for consistency). Gates require_symfc on
fc_calculator='symfc' so default-FD users don't pay for the symfc
import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Optional tier — `mode_resolved`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py` (extend synthesis node)
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — output tiers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mode_resolved_off_by_default(tmp_path):
    """Without mode_resolved=True, all mode-resolved fields are None."""
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.q_points is None
    assert out.frequencies is None
    assert out.group_velocities is None
    assert out.mode_kappa is None
    assert out.gamma is None
    assert out.gruneisen is None


@pytest.mark.slow
def test_mode_resolved_on_populates_all_fields(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
        mode_resolved=True,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.q_points is not None and out.q_points.shape[1] == 3
    assert out.frequencies is not None and out.frequencies.ndim == 2
    assert out.group_velocities is not None and out.group_velocities.shape[-1] == 3
    assert out.mode_kappa is not None
    assert out.gamma is not None
    n_q = out.q_points.shape[0]
    n_band = out.frequencies.shape[1]
    assert out.frequencies.shape == (n_q, n_band)
    assert out.mode_kappa.shape == (1, n_q, n_band, 6)
```

(`gruneisen` is not asserted on by default — phono3py computes it on a separate gruneisen workflow; expose it as `None` even when `mode_resolved=True` unless the user explicitly opts in later.)

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "mode_resolved"
```

Expected: `test_mode_resolved_off_by_default` may already pass (defaults None); `test_mode_resolved_on_populates_all_fields` fails because nothing populates the optional fields.

- [ ] **Step 3: Extend the synthesis node**

In `_run_phono3py_thermal_conductivity` in `anharmonic.py`, replace the final `return PhononOutput(...)` block with:

```python
    extras: dict = {}
    if mode_resolved:
        # phono3py stores mode-resolved data on .thermal_conductivity
        extras["q_points"] = np.asarray(tc.qpoints)
        extras["frequencies"] = np.asarray(tc.frequencies)
        extras["group_velocities"] = np.asarray(tc.group_velocities)
        extras["mode_kappa"] = np.asarray(tc.mode_kappa[0])  # (n_T, n_q, n_band, 6)
        extras["gamma"] = np.asarray(tc.gamma[0])  # (n_T, n_q, n_band)
        # gruneisen needs phono3py.gruneisen.Gruneisen — skip in v1.

    return PhononOutput(
        structure=structure,
        fc2_supercell_matrix=_normalise_supercell_matrix(fc2_supercell_matrix),
        fc3_supercell_matrix=_normalise_supercell_matrix(fc3_supercell_matrix),
        temperatures=T,
        kappa=kappa,
        converged=converged,
        **extras,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/physics/test_phonons.py -v -k "mode_resolved"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): mode_resolved output tier (q_points, frequencies, mode_kappa, gamma)

When mode_resolved=True, the synthesis node populates q_points,
frequencies, group_velocities, mode_kappa, and gamma from phono3py's
thermal_conductivity object. gruneisen needs a separate phono3py
workflow and is deferred. All default None when the flag is off.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Optional tier — `harmonic_observables` (bands, DOS, F(T))

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/harmonic.py` (add helpers)
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py` (call them when flag is on)
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_harmonic_observables_populates_bands_dos_freeenergy(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0, 500.0],
        q_mesh=(5, 5, 5),
        harmonic_observables=True,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out

    assert out.band_structure is not None
    assert "q" in out.band_structure and "frequencies" in out.band_structure
    assert out.dos is not None
    assert out.dos["frequencies"].ndim == 1
    assert out.dos["dos"].shape == out.dos["frequencies"].shape
    assert out.free_energy is not None
    assert out.free_energy["temperatures"].shape == (2,)
    assert out.free_energy["F"].shape == (2,)
    assert out.free_energy["S"].shape == (2,)
    assert out.free_energy["Cv"].shape == (2,)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_harmonic_observables_populates_bands_dos_freeenergy -v
```

Expected: fail — fields are None.

- [ ] **Step 3: Add `_compute_harmonic_observables` helper to `harmonic.py`**

Append to `pyiron_workflow_atomistics/physics/phonons/harmonic.py`:

```python
def _compute_harmonic_observables(
    ph3,
    temperatures: np.ndarray,
    band_path: list[list[list[float]]] | None = None,
    band_labels: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    """Compute band structure, total DOS, and Helmholtz free energy F(T).

    Uses ph3.phonon (a Phonopy view of the FC2-supercell phonons) so we
    don't double-build the harmonic side. The default band path is the
    automatic ASE primitive-cell path (`ase.dft.kpoints.bandpath`).

    Returns
    -------
    (band_structure, dos, free_energy) — each a plain dict suitable for
    PhononOutput.
    """
    phonopy_view = ph3.phonon  # Phonopy instance built from the FC2 part

    # ---- band structure (auto path from the unit cell) ----
    from ase.dft.kpoints import bandpath as ase_bandpath

    # Use the primitive cell as ASE knows it
    primitive_cell = np.asarray(phonopy_view.primitive.cell)
    bp = ase_bandpath("GXG", cell=primitive_cell, npoints=51)
    q_segment = bp.kpts.tolist()
    phonopy_view.run_band_structure([q_segment])
    bs = phonopy_view.get_band_structure_dict()
    band_structure = {
        "path": bp.path,
        "q": np.asarray(bs["qpoints"][0]),
        "frequencies": np.asarray(bs["frequencies"][0]),
    }

    # ---- total DOS ----
    phonopy_view.run_mesh(mesh=[20, 20, 20])
    phonopy_view.run_total_dos()
    tdos = phonopy_view.get_total_dos_dict()
    dos = {
        "frequencies": np.asarray(tdos["frequency_points"]),
        "dos": np.asarray(tdos["total_dos"]),
    }

    # ---- free energy F(T), entropy S(T), heat capacity Cv(T) ----
    phonopy_view.run_thermal_properties(temperatures=temperatures)
    tp = phonopy_view.get_thermal_properties_dict()
    free_energy = {
        "temperatures": np.asarray(tp["temperatures"]),
        "F": np.asarray(tp["free_energy"]),
        "S": np.asarray(tp["entropy"]),
        "Cv": np.asarray(tp["heat_capacity"]),
    }
    return band_structure, dos, free_energy
```

- [ ] **Step 4: Wire it into the synthesis node**

In `_run_phono3py_thermal_conductivity` in `anharmonic.py`, find the `extras: dict = {}` line added in Task 12. Below the `if mode_resolved:` block, add:

```python
    if harmonic_observables:
        from pyiron_workflow_atomistics.physics.phonons.harmonic import (
            _compute_harmonic_observables,
        )

        band_structure, dos, free_energy = _compute_harmonic_observables(
            ph3=ph3, temperatures=T
        )
        extras["band_structure"] = band_structure
        extras["dos"] = dos
        extras["free_energy"] = free_energy
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/physics/test_phonons.py::test_harmonic_observables_populates_bands_dos_freeenergy -v
```

Expected: pass. If the band-path call fails on the EMT-Cu primitive (ASE may not have a path named "GXG" depending on its lattice classifier), fall back to `ase_bandpath(path=None, cell=primitive_cell, npoints=51)` to let ASE pick the canonical path automatically.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/harmonic.py \
        pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): harmonic_observables output tier (bands, DOS, F(T))

When harmonic_observables=True, the synthesis node populates band_structure,
dos, and free_energy from the phonopy view of phono3py's FC2 supercell.
Free for the user since FC2 is already computed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Optional tier — `keep_handles` (raw FC2/FC3 + Phono3py object)

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_keep_handles_returns_fc2_fc3_and_phono3py_handle(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
        keep_handles=True,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out

    assert out.fc2 is not None
    assert out.fc2.ndim == 4 and out.fc2.shape[-1] == 3
    assert out.fc3 is not None
    assert out.fc3.ndim == 6 and out.fc3.shape[-1] == 3
    # phono3py handle is the live Phono3py object
    assert out.phono3py is not None
    assert hasattr(out.phono3py, "thermal_conductivity")
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_keep_handles_returns_fc2_fc3_and_phono3py_handle -v
```

Expected: fail — fields are None.

- [ ] **Step 3: Wire it into the synthesis node**

In `_run_phono3py_thermal_conductivity`, after the `if harmonic_observables:` block, add:

```python
    if keep_handles:
        extras["fc2"] = np.asarray(ph3.fc2)
        extras["fc3"] = np.asarray(ph3.fc3)
        extras["phono3py"] = ph3
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/unit/physics/test_phonons.py::test_keep_handles_returns_fc2_fc3_and_phono3py_handle -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): keep_handles output tier (raw FC2/FC3 + Phono3py handle)

When keep_handles=True, expose fc2, fc3, and the live Phono3py instance
on the PhononOutput for downstream custom analyses.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: κ-solver non-convergence warning scan

The bones of this guard already exist in Task 7 (`warnings.catch_warnings(record=True)`), but the message-matching predicate is hard to test against EMT-Cu (which converges easily). This task adds a unit test that injects a fake "not converged" warning and asserts `converged=False` falls out.

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py` (factor out the predicate)
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — κ-solver non-convergence predicate
# ---------------------------------------------------------------------------


def test_kappa_convergence_predicate_matches_phono3py_message():
    """The predicate should flag phono3py's documented non-convergence text."""
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _is_kappa_not_converged,
    )

    assert _is_kappa_not_converged(
        ["Iteration is not converged.", "Other warning"]
    ) is True
    assert _is_kappa_not_converged(["NOT CONVERGED in 100 iterations"]) is True
    assert _is_kappa_not_converged(["Successfully ran BTE"]) is False
    assert _is_kappa_not_converged([]) is False
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_kappa_convergence_predicate_matches_phono3py_message -v
```

Expected: `ImportError: cannot import _is_kappa_not_converged`.

- [ ] **Step 3: Factor out the predicate in `anharmonic.py`**

In `anharmonic.py`, find this block from Task 7:

```python
        converged = not any(
            "not converged" in str(w.message).lower() for w in caught
        )
```

Add the helper above the synthesis node (near `_check_all_converged`):

```python
def _is_kappa_not_converged(messages: list[str]) -> bool:
    """Return True if any phono3py warning indicates the κ solver failed.

    phono3py prints variants like 'Iteration is not converged.' or
    'NOT CONVERGED in N iterations'; both lowercase to a stable substring.
    """
    return any("not converged" in str(m).lower() for m in messages)
```

Then replace the `converged = not any(...)` line with:

```python
        converged = not _is_kappa_not_converged([w.message for w in caught])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/physics/test_phonons.py::test_kappa_convergence_predicate_matches_phono3py_message -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        tests/unit/physics/test_phonons.py
git commit -m "refactor(phonons): factor _is_kappa_not_converged predicate

Pulls the κ-solver non-convergence detection out of the synthesis node
so it has a Tier-1 unit test (the EMT-Cu smoke always converges, so
the predicate's behaviour on the failure path wasn't otherwise exercised).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Public re-exports + import smoke + CHANGELOG finalize

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/__init__.py`
- Modify: `tests/unit/physics/test_phonons.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — public re-exports
# ---------------------------------------------------------------------------


def test_public_reexports():
    """All publicly-documented symbols are importable from the subpackage."""
    from pyiron_workflow_atomistics.physics.phonons import (
        PhononOutput,
        calculate_phonon_thermal_conductivity,
    )

    assert PhononOutput is not None
    assert callable(calculate_phonon_thermal_conductivity)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/unit/physics/test_phonons.py::test_public_reexports -v
```

Expected: `ImportError: cannot import PhononOutput from pyiron_workflow_atomistics.physics.phonons` (the `__init__.py` from Task 2 is just a docstring).

- [ ] **Step 3: Replace `__init__.py`**

Overwrite `pyiron_workflow_atomistics/physics/phonons/__init__.py` with:

```python
"""Phonon workflows.

v1 covers phono3py-based lattice thermal conductivity κ(T) on top of a
phonopy harmonic FC2 calculation. Polar-material non-analytic correction
(BORN + ε∞) and dynaphopy-based MD renormalisation are documented in the
design spec as v2 follow-ups.

Public API
----------
- :class:`PhononOutput` — structured result dataclass.
- :func:`calculate_phonon_thermal_conductivity` — the user-facing macro.
"""

from .anharmonic import calculate_phonon_thermal_conductivity
from .output import PhononOutput

__all__ = ["PhononOutput", "calculate_phonon_thermal_conductivity"]
```

- [ ] **Step 4: Finalize the CHANGELOG entry**

In `CHANGELOG.md`, replace the `## [Unreleased]` heading from Task 1 with `## [0.0.7] — 2026-05-14` (today's date per system context). Leave the rest of the body intact. The downstream release workflow uses this once the tag is pushed.

- [ ] **Step 5: Run the full phonons test suite**

```bash
pytest tests/unit/physics/test_phonons.py -v
```

Expected: all tests pass (or skip cleanly if `[phonons]` extra not installed). Spot-check that all Tier 1 tests run regardless of extras availability.

- [ ] **Step 6: Run the full repo test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: green. No regression in existing tests. Total runtime ~5-10 minutes; ~2-3 min of that is the phonon Tier 2 tests.

- [ ] **Step 7: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/__init__.py \
        tests/unit/physics/test_phonons.py \
        CHANGELOG.md
git commit -m "feat(phonons): public re-exports + finalize 0.0.7 changelog

Exposes PhononOutput and calculate_phonon_thermal_conductivity at the
subpackage level. Tags the CHANGELOG entry with today's date for the
0.0.7 release.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final acceptance checklist

Before opening the PR, verify each of these manually:

- [ ] `pip install -e ".[test,phonons]"` succeeds in a fresh venv.
- [ ] `pip install -e .` (without `[phonons]`) succeeds — base install unaffected.
- [ ] `python -c "import pyiron_workflow_atomistics.physics.phonons"` succeeds with or without `[phonons]`.
- [ ] `pytest tests/unit/physics/test_phonons.py -v -k "not slow"` passes without the `[phonons]` extra (Tier 1 only).
- [ ] `pytest tests/unit/physics/test_phonons.py -v` passes fully with the `[phonons]` extra.
- [ ] `pytest tests/ -v` is green — no regression in existing tests.
- [ ] `CHANGELOG.md` has a single new entry at the top, tagged `0.0.7` with today's date.
- [ ] `git log --oneline` shows ~16 commits, one per task, each green when checked out.
