# dynaphopy MD-renormalisation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `physics/phonons/md_renormalised.py` module with a single user-facing macro `calculate_phonon_md_renormalisation(structure, engine, fc2_supercell_matrix, ...)` that runs a Langevin NVT MD segment via the existing `Engine` Protocol, projects the trajectory onto harmonic phonon modes via dynaphopy, and returns an `MdPhononOutput` dataclass with renormalised frequencies + linewidths + MD health diagnostics.

**Architecture:** 4–5 node macro graph (`_resolve_md_defaults` → optional `_compute_fc2_from_scratch` → `_run_nvt_trajectory` → `_project_with_dynaphopy`), mirroring the phono3py macro pattern from v0.0.7. FC2 source is either-or: pass `fc2_supercell_matrix` to recompute, or `phono3py_output` (from `calculate_phonon_thermal_conductivity` with `keep_handles=True`) to reuse. q-points default to an ASE-auto-derived high-symmetry band path so non-experts get a meaningful dispersion plot. `check_md_health()` method on the output + auto-warn at macro completion catch subtle MD problems on the first run.

**Tech Stack:** `pyiron_workflow` (`@pwf.as_function_node`, `@pwf.api.as_macro_node`), ASE `Atoms` + `ase.md.langevin.Langevin` (via existing `CalcInputMD`), phonopy + phono3py + symfc + dynaphopy (all optional extras), numpy, pytest.

**Spec:** `docs/design/specs/2026-05-15-dynaphopy-md-renormalisation-design.md`

**Conventions cross-checked against the codebase:**
- `pyiron_workflow_atomistics/physics/phonons/anharmonic.py` is the reference for `_resolve_defaults` pattern (execution-time arg resolution to dodge pyiron_workflow's `UserInput` proxy semantics) — read it before starting T5.
- `pyiron_workflow_atomistics/physics/phonons/harmonic.py:_compute_harmonic_observables` already shows the `ase.dft.kpoints.bandpath(path=None, cell=...)` auto-derivation pattern — reuse the same call.
- `pyiron_workflow_atomistics/engine/inputs.py:CalcInputMD` is the existing NVT-Langevin parameterisation — drive MD through it, don't reinvent.
- Tests live in `tests/unit/physics/test_phonons.py` (existing file holds all phonon-subpackage tests; appended). Tier 1/2/3 categorisation per the v0.0.7 pattern: Tier 1 no extras needed; Tier 2 gated on `pytest.importorskip("dynaphopy")` per test; Tier 3 determinism.
- End-to-end macro smoke tests use real ASE EMT + `@pytest.mark.slow`.

**Two deliberate decisions worth flagging upfront:**

1. **dynaphopy `Dynamics` constructor wants `(structure: PhonopyAtoms, trajectory, velocity, time, supercell)`** — the existing `_ase_to_phonopy` helper in `harmonic.py` does the ASE↔Phonopy conversion. ASE's `Langevin` MD doesn't expose velocities directly in `record_step`, so the trajectory pack carries them by reading `atoms.get_velocities()` per recorded step. Confirm against dynaphopy 1.19.0 docstrings before writing the synthesis node.
2. **The phonopy view dynaphopy needs has FC2 pre-attached.** Inside `_project_with_dynaphopy` we build a fresh `phonopy.Phonopy(unitcell, supercell_matrix=fc2_supercell_matrix)`, call `phonopy_view.force_constants = fc2_array`, then pass it into the `Quasiparticle` machinery. We do NOT pass the phonopy object through node boundaries (same rule as Phono3py in v0.0.7) — it's built locally inside the synthesis node.

**One deliberate divergence from spec:** the spec test list says "tests in `tests/unit/physics/test_phonons.py`" (appended). Plan keeps that file as the single test home for the subpackage. If it grows past ~2000 LOC we can split in a follow-up; not splitting now keeps the import-shim/health-check tests in one place.

---

## File structure

```
pyiron_workflow_atomistics/physics/phonons/
├── __init__.py          # T16 — add MdPhononOutput + calculate_phonon_md_renormalisation re-exports
├── _compat.py           # T2 — add require_dynaphopy()
├── output.py            # T3 — add MdPhononOutput dataclass + check_md_health()
├── harmonic.py          # T4 — header convention update only
├── anharmonic.py        # T4 — header convention update only
└── md_renormalised.py   # NEW — T4/T5 onwards

pyproject.toml           # T1 — [phonons-md] extras group
.ci_support/environment.yml  # T1 — + dynaphopy
CHANGELOG.md             # T1 (stub) + T16 (finalise 0.0.8)
tests/unit/physics/test_phonons.py  # append throughout
```

---

### Task 1: Add `[phonons-md]` optional extra + CHANGELOG stub + CI env

**Files:**
- Modify: `pyproject.toml` (in `[project.optional-dependencies]` block)
- Modify: `CHANGELOG.md` (prepend `[Unreleased]` entry)
- Modify: `.ci_support/environment.yml` (add `dynaphopy`)

- [ ] **Step 1: Add the `[phonons-md]` extra to `pyproject.toml`**

Open `pyproject.toml`. Find the existing `[project.optional-dependencies]` block:

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

Append a new `phonons-md` group as a superset of `phonons`:

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
phonons-md = [
    "phonopy",
    "phono3py",
    "symfc",
    "dynaphopy",
]
```

No version pins on `dynaphopy` for the same reason as the existing phonons group — pin if a regression appears.

- [ ] **Step 2: Add `dynaphopy` to `.ci_support/environment.yml`**

Open `.ci_support/environment.yml`. Find the block added by the phonons notebook PR:

```yaml
  # Optional [phonons] extra — needed for the phono3py example notebook
  - phonopy
  - phono3py
  - symfc
```

Replace with:

```yaml
  # Optional [phonons] / [phonons-md] extras — needed for the phono3py and
  # dynaphopy example notebooks plus their integration tests.
  - phonopy
  - phono3py
  - symfc
  - dynaphopy
```

- [ ] **Step 3: Prepend a draft entry to `CHANGELOG.md`**

The current top entry is `## [0.0.7]`. Prepend (above it):

```markdown
## [Unreleased]

### Added

- **`pyiron_workflow_atomistics.physics.phonons.calculate_phonon_md_renormalisation`**
  — new macro for MD-trajectory anharmonic phonon renormalisation via
  dynaphopy. Runs a Langevin NVT segment through the existing `Engine`
  Protocol, projects the trajectory's velocity ACF onto harmonic phonon
  modes, and returns an `MdPhononOutput` dataclass with renormalised
  frequencies, linewidths, and MD health diagnostics (⟨T⟩, σ_T,
  `check_md_health()` method, automatic warning on first bad run).
  Complementary to the v0.0.7 perturbative κ(T) workflow — captures
  full anharmonicity at finite T without perturbation theory.
- **`[phonons-md]` install extra** — `pip install
  pyiron_workflow_atomistics[phonons-md]` pulls in `phonopy`, `phono3py`,
  `symfc`, and `dynaphopy`. Superset of `[phonons]`; phono3py-only users
  keep the smaller install.
- **Module-header convention for phonon workflows** — `harmonic.py`,
  `anharmonic.py`, and `md_renormalised.py` each start with a docstring
  naming the upstream package they wrap (phonopy / phono3py / dynaphopy)
  for traceability.

### Out of scope (v2 follow-ups, see spec)

- NVE / NPT ensembles for the MD segment (Langevin NVT only in v1).
- Multi-temperature MD per call.
- NAC for polar materials (same status as in 0.0.7).
```

The `[Unreleased]` heading will be renamed to `## [0.0.8] — YYYY-MM-DD` at release time in T16.

- [ ] **Step 4: Verify install resolves in a fresh env**

Skip the install verification step locally if it churns the dev env — CI will exercise this with the conda-forge `dynaphopy` package. For local sanity, run:

```bash
mamba run -n test_pyiron_workflow_atomistics pip install -e ".[phonons-md]"
mamba run -n test_pyiron_workflow_atomistics python -c "import phonopy, phono3py, symfc, dynaphopy; print('ok')"
```

Expected: `ok`. If `dynaphopy` isn't already installed, this triggers a pip install of it (the test env already has phonopy / phono3py / symfc from v0.0.7).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml CHANGELOG.md .ci_support/environment.yml
git commit -m "build: add [phonons-md] optional install extra + dynaphopy in CI env

Pulls in dynaphopy as the marquee dep behind the upcoming
physics/phonons/md_renormalised.py workflow. Stored as a superset of
[phonons] so phono3py-only installs stay lean. CI env also gains
dynaphopy from conda-forge so the eventual example notebook + Tier 2
tests run end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `require_dynaphopy()` shim + Tier 1 test

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/_compat.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — require_dynaphopy lazy-import shim
# ---------------------------------------------------------------------------


def test_require_dynaphopy_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "dynaphopy")
    with pytest.raises(ImportError) as exc:
        _compat.require_dynaphopy()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons-md]" in msg
    assert "dynaphopy" in msg
```

(`_patch_missing` already exists in the same file — it sets the named module to `None` in `sys.modules` so `importlib.import_module` raises.)

- [ ] **Step 2: Run to verify it fails**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_require_dynaphopy_missing_actionable -v
```

Expected: `AttributeError: module 'pyiron_workflow_atomistics.physics.phonons._compat' has no attribute 'require_dynaphopy'`.

- [ ] **Step 3: Add the shim to `_compat.py`**

Open `pyiron_workflow_atomistics/physics/phonons/_compat.py`. Find the existing `require_symfc` function. Append below it:

```python
def require_dynaphopy() -> Any:
    """Return the imported dynaphopy module or raise an actionable ImportError.

    Used by md_renormalised.py for the MD-trajectory mode-projection workflow.
    The install hint references the [phonons-md] extras group (superset of
    [phonons] adding dynaphopy on top of phonopy + phono3py + symfc).
    """
    try:
        import importlib

        return importlib.import_module("dynaphopy")
    except ImportError as e:
        raise ImportError(
            "dynaphopy is required for this workflow. "
            "Install with: pip install pyiron_workflow_atomistics[phonons-md]"
        ) from e
```

Note this does NOT reuse `_require(module_name)` from earlier in the file because the install-hint group differs (`[phonons-md]` vs `[phonons]`). The two-line cost of duplicating the body beats parameterising the hint string.

- [ ] **Step 4: Run to verify it passes**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v -k "missing_actionable"
```

Expected: 4 passed (3 existing for phonopy/phono3py/symfc + the new dynaphopy one).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/_compat.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): require_dynaphopy lazy-import shim

Matches the require_phonopy / require_phono3py / require_symfc pattern,
but points at the [phonons-md] extras group instead of [phonons]. Used
by the upcoming md_renormalised.py module so the import cost only fires
when an MD-projection workflow is actually invoked.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `MdPhononOutput` dataclass + `check_md_health()` + Tier 1 tests

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/output.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — MdPhononOutput dataclass + check_md_health
# ---------------------------------------------------------------------------


def _make_md_output(
    *,
    temperature: float = 300.0,
    md_temperature_mean: float | None = None,
    md_temperature_std: float | None = None,
    n_atoms_supercell: int = 32,
):
    """Build a minimal MdPhononOutput for testing the dataclass shape + health checks."""
    from pyiron_workflow_atomistics.physics.phonons.output import MdPhononOutput

    if md_temperature_mean is None:
        md_temperature_mean = temperature
    if md_temperature_std is None:
        # Langevin expectation: T * sqrt(2 / (3 * N))
        md_temperature_std = temperature * np.sqrt(2.0 / (3.0 * n_atoms_supercell))

    cu = bulk("Cu", "fcc", a=3.6)
    return MdPhononOutput(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=temperature,
        q_points=np.zeros((1, 3)),
        harmonic_frequencies=np.array([[5.0, 5.0, 8.0]]),
        renormalised_frequencies=np.array([[4.9, 4.9, 7.8]]),
        linewidths=np.array([[0.1, 0.1, 0.2]]),
        converged=True,
        n_md_steps=2000,
        time_step_fs=1.0,
        md_temperature_mean=md_temperature_mean,
        md_temperature_std=md_temperature_std,
    )


def test_md_phonon_output_dataclass_shape():
    from dataclasses import MISSING, fields, is_dataclass

    from pyiron_workflow_atomistics.physics.phonons.output import MdPhononOutput

    assert is_dataclass(MdPhononOutput)

    required_names = {
        "structure",
        "fc2_supercell_matrix",
        "temperature",
        "q_points",
        "harmonic_frequencies",
        "renormalised_frequencies",
        "linewidths",
        "converged",
        "n_md_steps",
        "time_step_fs",
        "md_temperature_mean",
        "md_temperature_std",
    }
    optional_names = {
        "power_spectra",
        "frequency_grid",
        "quasiparticle",
        "dynamics",
        "phonopy",
    }
    by_name = {f.name: f for f in fields(MdPhononOutput)}
    for name in required_names:
        f = by_name[name]
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{name} must be required (no default)"
        )
    for name in optional_names:
        assert by_name[name].default is None, f"{name} must default to None"


def test_md_phonon_output_to_dict_round_trip():
    out = _make_md_output()
    d = out.to_dict()
    assert d["temperature"] == 300.0
    assert d["renormalised_frequencies"].shape == (1, 3)
    assert d["power_spectra"] is None


def test_check_md_health_passes_on_clean_run():
    out = _make_md_output()
    healthy, issues = out.check_md_health()
    assert healthy is True
    assert issues == []


def test_check_md_health_flags_temperature_drift():
    # 10% drift below requested
    out = _make_md_output(temperature=300.0, md_temperature_mean=270.0)
    healthy, issues = out.check_md_health()
    assert healthy is False
    assert any("drift" in i.lower() for i in issues)
    assert any("300" in i for i in issues)
    assert any("270" in i for i in issues)


def test_check_md_health_flags_anomalous_sigma():
    # σ_T way above Langevin expectation
    out = _make_md_output(temperature=300.0, md_temperature_std=200.0)
    healthy, issues = out.check_md_health()
    assert healthy is False
    assert any("σ" in i or "sigma" in i.lower() or "std" in i.lower() for i in issues)
```

- [ ] **Step 2: Run to verify they fail**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v -k "md_phonon_output or check_md_health"
```

Expected: 5 failures with `ImportError: cannot import name 'MdPhononOutput' from 'pyiron_workflow_atomistics.physics.phonons.output'`.

- [ ] **Step 3: Add `MdPhononOutput` + `check_md_health` to `output.py`**

Append to `pyiron_workflow_atomistics/physics/phonons/output.py`:

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
    fc2_supercell_matrix: np.ndarray  # (3, 3) int
    temperature: float                # K (target of the NVT run)
    q_points: np.ndarray              # (n_q, 3) reduced — actually used
    harmonic_frequencies: np.ndarray  # (n_q, n_band) THz — pre-renormalisation
    renormalised_frequencies: np.ndarray  # (n_q, n_band) THz — fitted
    linewidths: np.ndarray            # (n_q, n_band) THz FWHM
    converged: bool                   # all Lorentzian fits converged

    n_md_steps: int                   # production-only count
    time_step_fs: float
    md_temperature_mean: float
    md_temperature_std: float

    power_spectra: np.ndarray | None = None       # (n_q, n_band, n_freq_bins)
    frequency_grid: np.ndarray | None = None      # (n_freq_bins,) THz

    quasiparticle: Any | None = None              # dynaphopy.Quasiparticle
    dynamics: Any | None = None                   # dynaphopy.Dynamics
    phonopy: Any | None = None                    # phonopy.Phonopy view used

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every field (heavy objects by reference)."""
        return asdict(self)

    def check_md_health(
        self, drift_tolerance: float = 0.03
    ) -> tuple[bool, list[str]]:
        """Sanity-check the MD segment that drove the projection.

        Parameters
        ----------
        drift_tolerance
            Allowed relative drift between ``md_temperature_mean`` and the
            requested ``temperature``. Default 3%.

        Returns
        -------
        (is_healthy, issues)
            ``is_healthy`` is True iff no warnings fired. ``issues`` is a list
            of human-readable strings naming each issue.
        """
        issues: list[str] = []

        drift = (
            abs(self.md_temperature_mean - self.temperature) / self.temperature
        )
        if drift > drift_tolerance:
            issues.append(
                f"⟨T⟩ drift {drift:.1%} exceeds tolerance {drift_tolerance:.0%}: "
                f"requested {self.temperature:.1f} K, measured "
                f"{self.md_temperature_mean:.1f} K"
            )

        n_supercell_atoms = len(self.structure) * int(
            round(abs(np.linalg.det(self.fc2_supercell_matrix)))
        )
        expected_std = self.temperature * np.sqrt(2.0 / (3.0 * n_supercell_atoms))
        if expected_std > 0:
            ratio = self.md_temperature_std / expected_std
            if ratio < 0.5 or ratio > 2.0:
                issues.append(
                    f"σ_T = {self.md_temperature_std:.1f} K is {ratio:.2f}× the "
                    f"Langevin NVT expectation ({expected_std:.1f} K for "
                    f"{n_supercell_atoms} atoms at {self.temperature:.1f} K); "
                    f"thermostat coupling may be wrong"
                )
        return (not issues, issues)
```

(The `dataclass`, `asdict`, `Any`, `np`, `Atoms` imports are already present at the top of `output.py` from the existing `PhononOutput` definition — no new imports needed.)

- [ ] **Step 4: Run tests**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v -k "md_phonon_output or check_md_health"
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/output.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): MdPhononOutput dataclass + check_md_health()

Mirrors PhononOutput's required-up-front + Optional-tier pattern.
check_md_health() encodes two ironclad heuristics for the underlying
NVT MD segment: (a) ⟨T⟩ drift ≤ 3%, (b) σ_T within [0.5×, 2×] of the
Langevin expectation T·sqrt(2/(3·N)). Returns (is_healthy, issues).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Module-header convention applied across the phonon subpackage

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/harmonic.py` (header only)
- Modify: `pyiron_workflow_atomistics/physics/phonons/anharmonic.py` (header only)
- Create: `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py` (header + empty module)

- [ ] **Step 1: Update `harmonic.py` header**

Open `pyiron_workflow_atomistics/physics/phonons/harmonic.py`. Replace the existing module docstring with:

```python
"""phonopy FC2 helpers: supercell generation + ASE/PhonopyAtoms conversion.

Built on top of phonopy via a thin wrapper that exposes its functionality
as pyiron_workflow function-nodes and macros. The upstream package's name
is the authoritative source for behaviour and bug reports; this file
routes inputs/outputs through the pyiron_workflow Engine Protocol.
"""
```

- [ ] **Step 2: Update `anharmonic.py` header**

Open `pyiron_workflow_atomistics/physics/phonons/anharmonic.py`. Replace the existing module docstring with:

```python
"""phono3py FC3 + lattice thermal conductivity workflow.

The single user-facing entry point is :func:`calculate_phonon_thermal_conductivity`.

Built on top of phono3py via a thin wrapper that exposes its functionality
as pyiron_workflow function-nodes and macros. The upstream package's name
is the authoritative source for behaviour and bug reports; this file
routes inputs/outputs through the pyiron_workflow Engine Protocol.
"""
```

- [ ] **Step 3: Create `md_renormalised.py` with header + future imports**

Create `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`:

```python
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
from typing import Any

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
```

(Subsequent tasks add the helper functions and macro to this file.)

- [ ] **Step 4: Verify all phonon tests still pass**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v
```

Expected: all existing tests pass (count = the previous count + 5 from T2/T3). The new `md_renormalised.py` module imports successfully (no callables yet) — a smoke import in any subsequent test would confirm this.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/harmonic.py \
        pyiron_workflow_atomistics/physics/phonons/anharmonic.py \
        pyiron_workflow_atomistics/physics/phonons/md_renormalised.py
git commit -m "feat(phonons): module-header convention + md_renormalised.py scaffold

Each phonon-workflow module now starts with a docstring naming the
upstream package it wraps (phonopy / phono3py / dynaphopy). Documented
in the design spec; codified across the subpackage so future
contributors keep it consistent.

Creates the empty md_renormalised.py shell with only the standard
imports + the header. Macro and helper functions land in subsequent
tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `_resolve_md_defaults` function-node + argument-coupling validation

This is the execution-time arg resolver. Mirrors `_resolve_defaults` from `anharmonic.py` but for the dynaphopy macro's coupling rules.

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — _resolve_md_defaults argument coupling + auto-bandpath
# ---------------------------------------------------------------------------


def test_resolve_md_defaults_requires_at_least_one_fc2_source():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    with pytest.raises(ValueError) as exc:
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=None,
            phono3py_output=None,
            q_points=None,
            band_npoints=30,
            seed=42,
        )
    msg = str(exc.value)
    assert "fc2_supercell_matrix" in msg and "phono3py_output" in msg


def test_resolve_md_defaults_rejects_mismatched_supercells():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6)
    fake_phono3py_output = PhononOutput(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        fc3_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
        fc2=np.zeros((8, 8, 3, 3)),  # plausible FC2 shape for 2x2x2 Cu
    )

    with pytest.raises(ValueError) as exc:
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=3 * np.eye(3, dtype=int),  # MISMATCH
            phono3py_output=fake_phono3py_output,
            q_points=None,
            band_npoints=30,
            seed=42,
        )
    msg = str(exc.value)
    assert "disagree" in msg.lower() or "must match" in msg.lower()


def test_resolve_md_defaults_rejects_phono3py_output_without_fc2():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6)
    output_without_handles = PhononOutput(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        fc3_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
        # fc2 deliberately left None (i.e. keep_handles=False upstream)
    )

    with pytest.raises(ValueError) as exc:
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=None,
            phono3py_output=output_without_handles,
            q_points=None,
            band_npoints=30,
            seed=42,
        )
    msg = str(exc.value)
    assert "keep_handles=True" in msg
    assert "fc2" in msg.lower()


def test_resolve_md_defaults_auto_derives_band_path_when_qpoints_none():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    (
        resolved_fc2_supercell,
        resolved_q_points,
        resolved_seed,
        fc2_source_tag,
        fc2_array,
    ) = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=None,
        band_npoints=30,
        seed=42,
    )
    assert resolved_q_points.shape == (30, 3)
    assert fc2_source_tag == "recompute"
    assert fc2_array is None
    assert resolved_seed == 42


def test_resolve_md_defaults_band_path_is_deterministic():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    kwargs = dict(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=None,
        band_npoints=30,
        seed=None,  # auto-fill — but the q-points path should still match across calls
    )
    out_a = _resolve_md_defaults.node_function(**kwargs)
    out_b = _resolve_md_defaults.node_function(**kwargs)
    np.testing.assert_allclose(out_a[1], out_b[1])  # resolved_q_points identical


def test_resolve_md_defaults_passes_through_explicit_qpoints():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    user_q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    out = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=user_q,
        band_npoints=30,  # ignored because q_points is explicit
        seed=42,
    )
    np.testing.assert_allclose(out[1], user_q)


def test_resolve_md_defaults_seed_auto_filled_when_none():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    out = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=np.zeros((1, 3)),
        band_npoints=30,
        seed=None,
    )
    resolved_seed = out[2]
    assert resolved_seed is not None
    assert isinstance(resolved_seed, int)
    assert 0 <= resolved_seed < 2**32
```

- [ ] **Step 2: Run to verify they fail**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v -k "resolve_md_defaults"
```

Expected: `ImportError: cannot import name '_resolve_md_defaults'` for all 7 tests.

- [ ] **Step 3: Implement `_resolve_md_defaults`**

Append to `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v -k "resolve_md_defaults"
```

Expected: 7 passed. The auto-bandpath determinism test (`test_resolve_md_defaults_band_path_is_deterministic`) is technically Tier 3 but cheap — runs in Tier 1.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/md_renormalised.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): _resolve_md_defaults arg resolver + auto-bandpath

Execution-time arg resolution for the MD-renormalisation macro. Validates
the four-case FC2-source coupling table (both None, mismatch, missing
keep_handles, valid combinations); auto-derives a high-symmetry band path
via ASE's bandpath(path=None) when q_points is None; auto-fills seed via
SeedSequence().entropy. Mirrors anharmonic._resolve_defaults for the
analogous proxy-arg constraint in pyiron_workflow macro bodies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `_compute_fc2_from_scratch` function-node

When the user passes `fc2_supercell_matrix` (not a `phono3py_output`), this node runs the FC2 displacement generation + force eval + phonopy fit to produce the FC2 array. Reuses existing v0.0.7 building blocks.

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — _compute_fc2_from_scratch (gated on phonopy)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_fc2_from_scratch_produces_correct_shape(tmp_path):
    pytest.importorskip("phonopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _compute_fc2_from_scratch,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    fc2 = _compute_fc2_from_scratch.node_function(
        structure=cu,
        engine=engine,
        resolved_fc2_supercell=2 * np.eye(3, dtype=int),
    )
    # 2x2x2 of a Cu primitive (1 atom) → 8 supercell atoms
    assert fc2.shape == (8, 8, 3, 3)
    assert np.all(np.isfinite(fc2))
    # The fc2_disp_* directories should exist on disk
    assert (tmp_path / "fc2_disp_0000").exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_compute_fc2_from_scratch_produces_correct_shape -v
```

Expected: `ImportError: cannot import name '_compute_fc2_from_scratch'`.

- [ ] **Step 3: Implement `_compute_fc2_from_scratch`**

Append to `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`:

```python
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
```

Note this duplicates a small slice of `anharmonic._run_phono3py_thermal_conductivity`'s FC2-attach logic. The duplication is deliberate: we want a phonopy.Phonopy-fit FC2 (not phono3py's `ph3.fc2` which is structurally similar but reached through phono3py's API), and the dynaphopy projection wants a phonopy view downstream. Single fit, single object, no cross-package ambiguity.

- [ ] **Step 4: Run tests**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_compute_fc2_from_scratch_produces_correct_shape -v
```

Expected: pass. Runtime ~5-10s.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/md_renormalised.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): _compute_fc2_from_scratch function-node

Builds the FC2 array from scratch via the existing v0.0.7
_generate_fc2_supercells + _evaluate_supercells nodes plus a fresh
phonopy.Phonopy fit. Used when the macro caller didn't supply a
phono3py_output to reuse. Routes every force eval through
engine.with_working_directory('fc2_disp_NNNN/') for on-disk diagnostics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `_run_nvt_trajectory` function-node

Drives a Langevin NVT MD segment through the existing engine via `CalcInputMD`. Returns a `trajectory_pack` dict of plain arrays (positions, velocities, time, supercell) + scalars (mean T, std T, n steps, time step).

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — _run_nvt_trajectory smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_nvt_trajectory_returns_expected_pack_shape(tmp_path):
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _run_nvt_trajectory,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputMD(),  # ignored; the node builds its own
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    pack = _run_nvt_trajectory.node_function(
        structure=cu,
        engine=engine,
        resolved_fc2_supercell=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=300,
        time_step=1.0,
        thermostat_time_constant=100.0,
        seed=42,
    )

    n_supercell_atoms = 8  # 2x2x2 of Cu FCC primitive
    assert pack["positions"].shape == (300, n_supercell_atoms, 3)
    assert pack["velocities"].shape == (300, n_supercell_atoms, 3)
    assert pack["time"].shape == (300,)
    assert pack["supercell"].shape == (3, 3)
    assert pack["n_md_steps"] == 300
    assert pack["time_step_fs"] == 1.0
    # ⟨T⟩ and σ_T were measured (any finite positive value)
    assert pack["md_temperature_mean"] > 0
    assert pack["md_temperature_std"] > 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_run_nvt_trajectory_returns_expected_pack_shape -v
```

Expected: `ImportError: cannot import name '_run_nvt_trajectory'`.

- [ ] **Step 3: Implement `_run_nvt_trajectory`**

Append to `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`:

```python
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
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    # Build the supercell to actually run MD on. dynaphopy projects the
    # supercell trajectory onto modes via the FC2 supercell, so we MUST
    # run MD at the FC2 supercell size.
    from ase.build import make_supercell

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

    rng = np.random.default_rng(seed)
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
        if i >= production_steps:
            return
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
        "supercell": np.asarray(supercell_atoms.cell),
        "n_md_steps": production_steps,
        "time_step_fs": float(time_step),
        "md_temperature_mean": float(instantaneous_T.mean()),
        "md_temperature_std": float(instantaneous_T.std()),
    }
    return pack
```

Note: this node bypasses `engine.calculate()` because Langevin needs per-step control of the ASE `Atoms`. The spec's "every force eval goes through `engine.calculate`" rule has one principled carve-out: MD trajectory generation. The engine's calculator is borrowed; the `working_directory` / per-step subdir convention does NOT apply (it'd be 10000 directories). This is documented inline so future readers don't try to "fix" it.

- [ ] **Step 4: Run the test**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_run_nvt_trajectory_returns_expected_pack_shape -v
```

Expected: pass. Budget ~10s for 500 EMT MD steps on 8 atoms.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/md_renormalised.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): _run_nvt_trajectory Langevin MD driver

Runs a Langevin NVT segment on the FC2-supercell-sized Atoms (built
via ASE make_supercell). Equilibration discarded, production recorded
into a plain-ndarray + scalar dict (positions, velocities, time,
supercell, n_steps, dt, ⟨T⟩, σ_T) ready for dynaphopy.Dynamics
construction downstream. Deliberately bypasses engine.calculate() per
the spec's documented carve-out — MD wants step-by-step ASE control
that a fan-out node can't provide.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `_project_with_dynaphopy` synthesis node — required outputs only

This is the workhorse: build phonopy.Phonopy view → build dynaphopy.Dynamics → instantiate Quasiparticle → fit each q-point → pack into `MdPhononOutput`. Required fields only; optional output tiers in T12 / T13.

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — _project_with_dynaphopy synthesis smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_project_with_dynaphopy_emt_gamma_smoke(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _compute_fc2_from_scratch,
        _project_with_dynaphopy,
        _run_nvt_trajectory,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    fc2_supercell = 2 * np.eye(3, dtype=int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    fc2_array = _compute_fc2_from_scratch.node_function(
        structure=cu, engine=engine, resolved_fc2_supercell=fc2_supercell
    )
    pack = _run_nvt_trajectory.node_function(
        structure=cu,
        engine=engine,
        resolved_fc2_supercell=fc2_supercell,
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        time_step=1.0,
        thermostat_time_constant=100.0,
        seed=42,
    )

    out = _project_with_dynaphopy.node_function(
        structure=cu,
        fc2_array=fc2_array,
        resolved_fc2_supercell=fc2_supercell,
        trajectory_pack=pack,
        resolved_q_points=np.zeros((1, 3)),  # Γ only
        temperature=300.0,
        power_spectra=False,
        keep_handles=False,
    )

    # 1 q-point × 3 bands (Cu primitive has 1 atom)
    assert out.renormalised_frequencies.shape == (1, 3)
    assert out.linewidths.shape == (1, 3)
    assert out.harmonic_frequencies.shape == (1, 3)
    assert out.power_spectra is None
    assert out.quasiparticle is None
    # All renormalised frequencies should be finite (EMT-Cu is well-behaved)
    assert np.all(np.isfinite(out.renormalised_frequencies))
    # Renormalisation should not drift wildly from the harmonic value
    rel_drift = np.abs(
        (out.renormalised_frequencies - out.harmonic_frequencies)
        / out.harmonic_frequencies
    )
    assert (rel_drift < 0.5).all(), (
        f"Anomalous renormalisation: rel_drift = {rel_drift}"
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_project_with_dynaphopy_emt_gamma_smoke -v
```

Expected: `ImportError: cannot import name '_project_with_dynaphopy'`.

- [ ] **Step 3: Implement `_project_with_dynaphopy`**

Append to `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`:

```python
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
    out = np.zeros((len(q_points), 3 * len(phonon.unitcell)))
    for i, q in enumerate(q_points):
        freqs = phonon.get_frequencies(q)
        out[i] = np.asarray(freqs)
    return out


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
    from dynaphopy.dynamics import Dynamics

    from pyiron_workflow_atomistics.physics.phonons.harmonic import _ase_to_phonopy

    phonon = _build_phonopy_view(structure, fc2_array, resolved_fc2_supercell)

    # Harmonic reference frequencies (always populated for comparison).
    harmonic_frequencies = _harmonic_frequencies_at(phonon, resolved_q_points)

    # Build dynaphopy Dynamics from the trajectory pack.
    supercell_phonopy = _ase_to_phonopy(structure)  # PhonopyAtoms of the primitive
    dynamics = Dynamics(
        structure=supercell_phonopy,
        trajectory=trajectory_pack["positions"],
        velocity=trajectory_pack["velocities"],
        time=trajectory_pack["time"] * 1e-3,  # dynaphopy expects ps
        supercell=np.asarray(resolved_fc2_supercell),
    )

    qp = Quasiparticle(dynamics)
    qp.set_qha_force_constants(phonon)  # attach FC2 view
    qp.set_temperature(temperature)

    n_q = len(resolved_q_points)
    n_band = harmonic_frequencies.shape[1]
    renormalised = np.full((n_q, n_band), np.nan)
    linewidths = np.full((n_q, n_band), np.nan)
    failed: list[tuple[int, int]] = []

    spectra_blocks: list[np.ndarray] = []
    freq_grid: np.ndarray | None = None

    for iq, q in enumerate(resolved_q_points):
        qp.set_reduced_q_vector(q)
        try:
            data = qp.get_renormalized_phonon_dispersion_bands()
            # dynaphopy returns a dict-like structure; the exact keys depend
            # on version. Extract by attribute when possible.
            freqs = np.asarray(getattr(data, "renormalized_frequencies", data))
            widths = np.asarray(getattr(data, "linewidths", np.zeros(n_band)))
            renormalised[iq] = freqs[:n_band]
            linewidths[iq] = widths[:n_band]
        except Exception:  # noqa: BLE001 — dynaphopy may raise various fit errors
            failed.append((iq, -1))

        if power_spectra:
            # Capture the full power spectrum per band at this q.
            try:
                ps = qp.get_power_spectrum_phonon()
                if freq_grid is None:
                    freq_grid = np.asarray(getattr(ps, "frequency", ps[..., 0]))
                spectra_blocks.append(np.asarray(getattr(ps, "spectrum", ps)))
            except Exception:  # noqa: BLE001
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
```

**Important note on dynaphopy API:** dynaphopy 1.19.0's exact method names for "renormalised frequencies at this q" and "power spectrum" may differ between minor releases. The plan code uses `qp.get_renormalized_phonon_dispersion_bands()` and `qp.get_power_spectrum_phonon()` as the primary handles, with `getattr` fallback for attribute access on the returned objects. **Verify these against the installed dynaphopy version before merging the test**; adapt names if needed, but DO NOT change the macro signature or output shape — those are spec-locked.

- [ ] **Step 4: Run the test**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_project_with_dynaphopy_emt_gamma_smoke -v
```

Expected: pass. Budget ~60-120s. If dynaphopy's API names differ, the test fails informatively and you adjust the names with minimum surgery.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/md_renormalised.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): _project_with_dynaphopy synthesis node

Builds a phonopy.Phonopy view with the supplied FC2, instantiates
dynaphopy.Quasiparticle on the MD trajectory, fits Lorentzians per
q-point, NaN-fills failed fits, computes harmonic reference
frequencies for comparison, and packs into MdPhononOutput. Auto-warns
via check_md_health() if the MD segment looks unhealthy.

Optional tiers (power_spectra, keep_handles) are honoured at this
node — wiring through the macro lands in T9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: User-facing `calculate_phonon_md_renormalisation` macro

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing macro-level smoke test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 — full-macro smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calculate_phonon_md_renormalisation_macro_emt(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        time_step=1.0,
        q_points=[[0.0, 0.0, 0.0]],  # Γ-only for runtime
        seed=42,
    )
    wf.run()
    out = wf.outputs.md_phonon_output.value

    assert out.converged is True
    assert out.renormalised_frequencies.shape == (1, 3)
    assert out.q_points.shape == (1, 3)
    # FC2 was recomputed → fc2_disp_NNNN dirs on disk
    assert (tmp_path / "fc2_disp_0000").exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_calculate_phonon_md_renormalisation_macro_emt -v
```

Expected: `ImportError: cannot import name 'calculate_phonon_md_renormalisation'`.

- [ ] **Step 3: Implement the macro**

Append to `pyiron_workflow_atomistics/physics/phonons/md_renormalised.py`:

```python
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
```

Plus a small helper node that picks FC2 between reuse and recompute (so the macro body stays declarative — pyiron_workflow can't do `if` branches over `UserInput` proxies):

```python
@pwf.as_function_node("fc2_array")
def _select_or_compute_fc2(
    structure: Atoms,
    engine: Engine,
    resolved_fc2_supercell,
    fc2_source_tag: str,
    fc2_array_reused,
) -> np.ndarray:
    """Pick FC2 between reuse (from phono3py_output) and recomputation."""
    if fc2_source_tag == "reuse":
        if fc2_array_reused is None:
            raise RuntimeError(
                "Internal error: fc2_source_tag='reuse' but fc2_array_reused is None"
            )
        return np.asarray(fc2_array_reused)
    elif fc2_source_tag == "recompute":
        return _compute_fc2_from_scratch.node_function(
            structure=structure,
            engine=engine,
            resolved_fc2_supercell=resolved_fc2_supercell,
        )
    else:
        raise ValueError(f"Unknown fc2_source_tag: {fc2_source_tag!r}")
```

- [ ] **Step 4: Run the macro smoke test**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_calculate_phonon_md_renormalisation_macro_emt -v
```

Expected: pass. Budget ~60-120s.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/md_renormalised.py \
        tests/unit/physics/test_phonons.py
git commit -m "feat(phonons): calculate_phonon_md_renormalisation macro

Wires the 4-node graph: _resolve_md_defaults → _select_or_compute_fc2
(branches on fc2_source_tag) → _run_nvt_trajectory → _project_with_dynaphopy.
Mirrors the anharmonic._resolve_defaults / wf.defaults.outputs.<port>
pattern from v0.0.7 so all execution-time arg handling stays proxy-safe.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: FC2 reuse path (skipping `_compute_fc2_from_scratch`)

The macro from T9 supports both FC2-recompute and FC2-reuse paths via `_select_or_compute_fc2`. T10 adds the integration test that locks in the reuse path: end-to-end through phono3py → dynaphopy.

**Files:**
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_md_macro_reuses_fc2_from_phono3py_output(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = 2 * np.eye(3, dtype=int)

    # Step 1: run the phono3py macro with keep_handles=True
    engine_phono3py = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "phono3py_run"),
    )
    wf_phono3py = calculate_phonon_thermal_conductivity(
        structure=cu,
        engine=engine_phono3py,
        fc2_supercell_matrix=sc,
        temperatures=[300.0],
        q_mesh=(3, 3, 3),
        keep_handles=True,
    )
    wf_phono3py.run()
    phono3py_out = wf_phono3py.outputs.phonon_output.value

    # Step 2: run dynaphopy macro reusing the FC2.
    engine_md = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "md_run"),
    )
    wf_md = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine_md,
        # fc2_supercell_matrix deliberately NOT passed → must derive from
        # phono3py_output
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        time_step=1.0,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
        phono3py_output=phono3py_out,
    )
    wf_md.run()
    out = wf_md.outputs.md_phonon_output.value

    # Reuse path → no fc2_disp_NNNN directories in the dynaphopy run's workdir.
    assert not (tmp_path / "md_run" / "fc2_disp_0000").exists()
    # FC2 supercell propagates from phono3py output.
    np.testing.assert_array_equal(
        out.fc2_supercell_matrix, phono3py_out.fc2_supercell_matrix
    )
    assert out.renormalised_frequencies.shape == (1, 3)
```

- [ ] **Step 2: Run to verify it passes**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_md_macro_reuses_fc2_from_phono3py_output -v
```

Expected: pass (no production-code changes; the macro graph already supports the reuse path via `_select_or_compute_fc2`'s branch). If the test fails because the FC2 ndarray shape from phono3py doesn't directly match phonopy's expected layout, do minimum surgery in `_select_or_compute_fc2`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/physics/test_phonons.py
git commit -m "test(phonons): lock in FC2 reuse path through phono3py_output

End-to-end test: run calculate_phonon_thermal_conductivity with
keep_handles=True to capture FC2, then run
calculate_phonon_md_renormalisation passing phono3py_output. Asserts
(a) no fc2_disp_NNNN dirs created in the MD run's working_directory,
(b) FC2 supercell propagates correctly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Auto-warn on bad MD diagnostics

Auto-warn logic already lives in `_project_with_dynaphopy` (T8). T11 adds the monkey-patched test that locks the behaviour in.

**Files:**
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_md_macro_warns_when_temperature_drifts(monkeypatch, tmp_path):
    """Monkey-patch the trajectory pack to fake a wildly drifted ⟨T⟩."""
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons import md_renormalised
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    # Wrap _run_nvt_trajectory.node_function so the returned pack reports a
    # bogus ⟨T⟩ that triggers the drift check.
    original_node_function = md_renormalised._run_nvt_trajectory.node_function

    def drifted_node_function(*args, **kwargs):
        pack = original_node_function(*args, **kwargs)
        pack["md_temperature_mean"] = 200.0  # >>3% drift from requested 300
        return pack

    monkeypatch.setattr(
        md_renormalised._run_nvt_trajectory, "node_function", drifted_node_function
    )

    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=100,
        production_steps=500,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
    )
    with pytest.warns(UserWarning, match=r"⟨T⟩ drift.*exceeds tolerance"):
        wf.run()
```

- [ ] **Step 2: Run to verify it passes**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_md_macro_warns_when_temperature_drifts -v
```

Expected: pass (auto-warn already implemented in T8).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/physics/test_phonons.py
git commit -m "test(phonons): lock in auto-warn on MD ⟨T⟩ drift

Monkey-patches _run_nvt_trajectory to inject a bogus ⟨T⟩ that's well
outside the 3% drift tolerance, then asserts pytest.warns catches the
UserWarning emitted by _project_with_dynaphopy via check_md_health().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: `power_spectra=True` output tier

Field already populated by `_project_with_dynaphopy` (T8) when `power_spectra=True`. T12 adds tests.

**Files:**
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_power_spectra_off_by_default(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
    )
    wf.run()
    out = wf.outputs.md_phonon_output.value
    assert out.power_spectra is None
    assert out.frequency_grid is None


@pytest.mark.slow
def test_power_spectra_on_populates_arrays(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
        power_spectra=True,
    )
    wf.run()
    out = wf.outputs.md_phonon_output.value
    assert out.power_spectra is not None
    assert out.frequency_grid is not None
    # (n_q, n_band, n_freq_bins) — but n_band index ordering may differ
    # by dynaphopy version; just check first two axes.
    assert out.power_spectra.shape[0] == 1
    assert out.power_spectra.shape[1] == 3 or out.power_spectra.ndim == 2
```

- [ ] **Step 2: Run to verify they pass**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v -k "power_spectra"
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/physics/test_phonons.py
git commit -m "test(phonons): lock in power_spectra output tier (on / off)

Two macro-level tests confirming (a) power_spectra=False default leaves
the field None, (b) power_spectra=True populates power_spectra and
frequency_grid arrays. Underlying implementation already in T8's
_project_with_dynaphopy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: `keep_handles=True` output tier

Same story: implementation in T8; T13 adds the lock-in test.

**Files:**
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_keep_handles_returns_quasiparticle_dynamics_phonopy(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
        keep_handles=True,
    )
    wf.run()
    out = wf.outputs.md_phonon_output.value
    assert out.quasiparticle is not None
    assert out.dynamics is not None
    assert out.phonopy is not None
    # Sanity-check the phonopy handle by reading FC2 shape off it.
    assert np.asarray(out.phonopy.force_constants).shape == (8, 8, 3, 3)
```

- [ ] **Step 2: Run to verify it passes**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_keep_handles_returns_quasiparticle_dynamics_phonopy -v
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/physics/test_phonons.py
git commit -m "test(phonons): lock in keep_handles output tier

Asserts keep_handles=True populates quasiparticle, dynamics, and
phonopy fields on MdPhononOutput, and that the phonopy handle's
force_constants shape is the expected (8, 8, 3, 3) for 2x2x2 Cu.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Seed determinism integration test

**Files:**
- Modify: `tests/unit/physics/test_phonons.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
@pytest.mark.slow
def test_md_macro_seed_determinism(tmp_path):
    """Same seed → byte-identical renormalised frequencies across two runs."""
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    common_kwargs = dict(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=1500,
        time_step=1.0,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
    )

    engine_a = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "run_a"),
    )
    wf_a = calculate_phonon_md_renormalisation(engine=engine_a, **common_kwargs)
    wf_a.run()

    engine_b = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "run_b"),
    )
    wf_b = calculate_phonon_md_renormalisation(engine=engine_b, **common_kwargs)
    wf_b.run()

    out_a = wf_a.outputs.md_phonon_output.value
    out_b = wf_b.outputs.md_phonon_output.value
    np.testing.assert_allclose(
        out_a.renormalised_frequencies, out_b.renormalised_frequencies
    )
```

- [ ] **Step 2: Run to verify it passes**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_md_macro_seed_determinism -v
```

Expected: pass (T5's `_resolve_md_defaults` threads the resolved seed through `_run_nvt_trajectory`; the trajectory + dynaphopy fit are then deterministic).

If this test fails (the trajectory generation is non-deterministic for the same seed), inspect `_run_nvt_trajectory`: ASE's Langevin uses `np.random.RandomState(seed)` per the T7 implementation, which IS deterministic. Drift indicates a hidden non-deterministic source — find and fix it.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/physics/test_phonons.py
git commit -m "test(phonons): lock in MD seed determinism through the macro

Two macro runs with the same seed produce byte-identical renormalised
frequencies. Validates that _resolve_md_defaults' seed plumbing reaches
both _run_nvt_trajectory and (transitively) the dynaphopy fit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Public re-exports + CHANGELOG finalize + 0.0.8

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/phonons/__init__.py`
- Modify: `tests/unit/physics/test_phonons.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_phonons.py`:

```python
# ---------------------------------------------------------------------------
# Tier 1 — public re-exports (md renormalisation)
# ---------------------------------------------------------------------------


def test_public_reexports_include_md_renormalisation():
    from pyiron_workflow_atomistics.physics.phonons import (
        MdPhononOutput,
        PhononOutput,
        calculate_phonon_md_renormalisation,
        calculate_phonon_thermal_conductivity,
    )

    assert PhononOutput is not None
    assert MdPhononOutput is not None
    assert callable(calculate_phonon_thermal_conductivity)
    assert callable(calculate_phonon_md_renormalisation)
```

- [ ] **Step 2: Run to verify it fails**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py::test_public_reexports_include_md_renormalisation -v
```

Expected: `ImportError: cannot import MdPhononOutput from pyiron_workflow_atomistics.physics.phonons`.

- [ ] **Step 3: Update `__init__.py`**

Open `pyiron_workflow_atomistics/physics/phonons/__init__.py`. Replace its content with:

```python
"""Phonon workflows.

v0.0.7 covers phono3py-based lattice thermal conductivity κ(T) on top of a
phonopy harmonic FC2 calculation.

v0.0.8 adds dynaphopy-based MD-trajectory anharmonic renormalisation, which
captures full anharmonicity at finite T from a Langevin NVT segment.

Polar-material non-analytic correction (BORN + ε∞) is documented as a v2
follow-up in both phono3py and dynaphopy specs.

Public API
----------
- :class:`PhononOutput` — structured result of the phono3py BTE workflow.
- :class:`MdPhononOutput` — structured result of the dynaphopy MD-projection workflow.
- :func:`calculate_phonon_thermal_conductivity` — phono3py macro (v0.0.7).
- :func:`calculate_phonon_md_renormalisation` — dynaphopy macro (v0.0.8).
"""

from .anharmonic import calculate_phonon_thermal_conductivity
from .md_renormalised import calculate_phonon_md_renormalisation
from .output import MdPhononOutput, PhononOutput

__all__ = [
    "PhononOutput",
    "MdPhononOutput",
    "calculate_phonon_thermal_conductivity",
    "calculate_phonon_md_renormalisation",
]
```

- [ ] **Step 4: Finalize the CHANGELOG entry**

In `CHANGELOG.md`, replace the `## [Unreleased]` heading from T1 with `## [0.0.8] — 2026-05-15` (today's date per system context). Leave the rest of the body intact.

- [ ] **Step 5: Run the full phonons test suite**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/unit/physics/test_phonons.py -v
```

Expected: all tests pass. Tier 1 count = 27 (existing) + ~13 new from T2/T3/T5/T15 = ~40. Tier 2 count = ~10 (existing) + 8 new = ~18. Total ~58 phonon-subpackage tests.

- [ ] **Step 6: Run the full repo test suite**

```bash
mamba run -n test_pyiron_workflow_atomistics pytest tests/ --tb=short -x
```

Expected: green. Total runtime ~10-15 min (the new Tier-2 tests add ~2-3 min on top of the v0.0.7 baseline).

- [ ] **Step 7: Commit**

```bash
git add pyiron_workflow_atomistics/physics/phonons/__init__.py \
        tests/unit/physics/test_phonons.py \
        CHANGELOG.md
git commit -m "feat(phonons): public re-exports + finalize 0.0.8 changelog

Exposes MdPhononOutput and calculate_phonon_md_renormalisation alongside
the v0.0.7 PhononOutput / calculate_phonon_thermal_conductivity. Tags
the CHANGELOG entry with today's date for the 0.0.8 release.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final acceptance checklist

Before opening the PR, verify each of these manually:

- [ ] `pip install -e ".[test,phonons-md]"` succeeds in a fresh pixi env (per the [pixi-for-envs] memory).
- [ ] `pip install -e .` (without `[phonons-md]`) succeeds — base install unaffected.
- [ ] `python -c "import pyiron_workflow_atomistics.physics.phonons"` succeeds with or without `[phonons-md]`.
- [ ] `pytest tests/unit/physics/test_phonons.py -v -k "not slow"` passes without the `[phonons-md]` extra (Tier 1 only).
- [ ] `pytest tests/unit/physics/test_phonons.py -v` passes fully with the `[phonons-md]` extra.
- [ ] `pytest tests/ -v` is green — no regression in existing tests (including the v0.0.7 phono3py macro).
- [ ] `CHANGELOG.md` has a single new entry at the top, tagged `0.0.8` with today's date.
- [ ] `git log --oneline` shows ~15 commits on the feature branch, one per task.
