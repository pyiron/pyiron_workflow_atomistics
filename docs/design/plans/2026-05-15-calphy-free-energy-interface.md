# Calphy free-energy interface — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `physics/free_energy/` subpackage with six user-facing function-nodes — `free_energy`, `reversible_scaling_temperature`, `reversible_scaling_pressure`, `melting_temperature`, `alchemy`, `composition_scaling` — each returning a typed `FreeEnergyOutput` dataclass. Each node wraps `calphy` and consumes a `LammpsEngine` (only its `command` field) plus a separate `LammpsPotential` dataclass.

**Architecture:** Six public function-nodes share two private helpers in `_calphy_adapter.py`: `_build_calphy_calculation` (per-mode kwarg fan-out → `calphy.input.Calculation`) and `_run_calphy_job` (dispatches `calphy.kernel.setup_calculation` + `run_calculation`, reads `report.yaml` back as a dict). Calphy + `pyiron_workflow_lammps` are an opt-in `[free-energy]` extra behind lazy imports inside `_compat.py`. Calphy's `script_mode=True` is hardcoded so calphy runs LAMMPS as a subprocess, matching how `LammpsEngine` itself runs LAMMPS.

**Tech Stack:** `pyiron_workflow` (`@pwf.as_function_node`), `ase.Atoms`, `calphy>=1.5.6`, `pyiron_workflow_lammps`, numpy, pytest.

**Spec:** `docs/design/specs/2026-05-15-calphy-free-energy-interface-design.md`

**Conventions cross-checked against the codebase:**
- `physics/phonons/` is the canonical reference for "lazy-imported physics subpackage with an `_compat.py` shim and an optional extra". Read `physics/phonons/_compat.py` and `physics/phonons/__init__.py` before Task 2.
- Tests live in `tests/unit/physics/test_<topic>.py`; conftest fixtures `simple_atoms`, `fcc_al_atoms` are available (see `tests/conftest.py`).
- Tier-2 integration tests gate on `pytest.importorskip("calphy")` AND `shutil.which("lmp")`; mark as `@pytest.mark.slow` when budget > 60 s.
- LAMMPS data files are written via `ase.io.write(path, atoms, format="lammps-data")`; calphy reads them by setting `Calculation.lattice` to the file path and `Calculation.file_format="lammps-data"`.
- Calphy's element-ordering rule (see `calphy/input.py:309-318`) requires `Calculation.element` to match the `pair_coeff` element list; we derive elements from `dict.fromkeys(atoms.get_chemical_symbols())` exactly as `LammpsEngine.get_lammps_element_order` already does.

**One deliberate deviation from the spec:** the spec listed `_load_rs_curve` as a Tier-1 helper. In practice, calphy writes reversible-scaling artefacts as `temperature_sweep.dat` and `free_energy.dat` whose exact filenames have drifted across calphy versions. To avoid pinning calphy's internals, the plan reads them through `np.loadtxt` on the file pattern calphy currently uses (`*free_energy*.dat`) and explicitly tests against pinned files; if calphy renames them, a single helper update fixes all nodes.

---

## File structure

```
pyiron_workflow_atomistics/physics/free_energy/
├── __init__.py             # Task 18 — public re-exports
├── _compat.py              # Task 2 — _require_calphy(), _require_lammps_engine()
├── inputs.py               # Task 3 — LammpsPotential dataclass
├── outputs.py              # Task 4 — FreeEnergyOutput dataclass
├── _calphy_adapter.py      # Tasks 5, 6, 7, 8, 9, 10 — adapter helpers
└── calphy.py               # Tasks 11–17 — six public function-nodes

tests/unit/physics/
└── test_free_energy.py     # All Tier 1, 2, 3 tests

tests/resources/free_energy/
└── Cu01.eam.alloy          # Task 12 — pinned test potential (vendored from calphy/examples/potentials)
```

---

### Task 1: Add `[free-energy]` optional extra and CHANGELOG stub

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]` block)
- Modify: `CHANGELOG.md` (prepend new top entry)

- [ ] **Step 1: Add the extra to `pyproject.toml`**

Open `pyproject.toml`. Find the existing `[project.optional-dependencies]` block (around line 43). Append a new entry **after** `phonons-md`:

```toml
free-energy = [
    "calphy>=1.5.6",
    "pyiron_workflow_lammps",
]
```

Block becomes:

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
free-energy = [
    "calphy>=1.5.6",
    "pyiron_workflow_lammps",
]
```

No version pin on `pyiron_workflow_lammps` — the calphy adapter only reads `engine.command` and the `dataclasses.fields` introspection it does is API-stable.

- [ ] **Step 2: Prepend a draft entry to `CHANGELOG.md`**

Current top entry is `## [0.0.8] — 2026-05-15`. Prepend (above it):

```markdown
## [Unreleased]

### Added

- **`pyiron_workflow_atomistics.physics.free_energy`** — new subpackage
  for free-energy workflows via `calphy`. Six public function-nodes:
  `free_energy`, `reversible_scaling_temperature`,
  `reversible_scaling_pressure`, `melting_temperature`, `alchemy`,
  `composition_scaling`. Each returns a typed `FreeEnergyOutput`
  dataclass and consumes a minimal `LammpsEngine` (only its `command`
  field is read) plus a dedicated `LammpsPotential` dataclass.
- **`[free-energy]` install extra** — `pip install
  pyiron_workflow_atomistics[free-energy]` pulls in `calphy>=1.5.6` and
  `pyiron_workflow_lammps`. Base install unaffected; lazy imports keep
  non-free-energy users from paying for the extra.

### Out of scope (v2 follow-ups)

- Free-energy extraction from `phonopy` (QHA) and `dynaphopy` surfaced
  as additional nodes in this same subpackage.
- SLURM/SGE scheduler passthrough (calphy supports it natively; v1
  pins `scheduler='local'`).

```

The `[Unreleased]` heading will be renamed to `## [0.0.9] — YYYY-MM-DD` at release time.

- [ ] **Step 3: Verify the install resolves**

```bash
cd /home/liger/pyiron_workflow_atomistics
pip install -e ".[test,free-energy]"
```

Expected: install completes; `python -c "import calphy, pyiron_workflow_lammps; print('ok')"` prints `ok`.

If `calphy` wheels are missing for your platform, log it and proceed — Tier-1 tests run without calphy, and Tier-2 tests skip via `pytest.importorskip`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "$(cat <<'EOF'
build: add [free-energy] optional install extra

Pulls in calphy>=1.5.6 and pyiron_workflow_lammps as opt-in deps for
the upcoming physics/free_energy subpackage. Base install unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Lazy-import shims in `_compat.py`

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/__init__.py`
- Create: `pyiron_workflow_atomistics/physics/free_energy/_compat.py`
- Create: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_free_energy.py`:

```python
"""Tests for pyiron_workflow_atomistics.physics.free_energy."""

from __future__ import annotations

import sys

import pytest


def test_require_calphy_raises_actionable_when_missing(monkeypatch):
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    monkeypatch.setitem(sys.modules, "calphy", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        _compat._require_calphy()
    assert "pip install 'pyiron_workflow_atomistics[free-energy]'" in str(exc.value)


def test_require_lammps_engine_raises_actionable_when_missing(monkeypatch):
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    monkeypatch.setitem(sys.modules, "pyiron_workflow_lammps", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        _compat._require_lammps_engine()
    assert "pip install 'pyiron_workflow_atomistics[free-energy]'" in str(exc.value)


def test_require_calphy_returns_module_when_present():
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    calphy = pytest.importorskip("calphy")
    assert _compat._require_calphy() is calphy
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/liger/pyiron_workflow_atomistics
pytest tests/unit/physics/test_free_energy.py -v
```

Expected: `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.physics.free_energy'`.

- [ ] **Step 3: Create the package directory and `__init__.py`**

```bash
mkdir -p pyiron_workflow_atomistics/physics/free_energy
```

Create `pyiron_workflow_atomistics/physics/free_energy/__init__.py`:

```python
"""calphy-backed free-energy workflows.

Public re-exports are added in Task 18 once all components exist. This
file is intentionally tiny until then so importing the subpackage
doesn't fail when the [free-energy] extra is not installed.
"""
```

- [ ] **Step 4: Implement `_compat.py`**

Create `pyiron_workflow_atomistics/physics/free_energy/_compat.py`:

```python
"""Lazy-import shims for the optional [free-energy] extra.

Mirrors the pattern in physics/phonons/_compat.py: every public node
calls one of these helpers at the top of its body so the error message
points the user at the install line, not a bare ``ImportError``.
"""

from __future__ import annotations

_INSTALL_HINT = (
    "pip install 'pyiron_workflow_atomistics[free-energy]'"
)


def _require_calphy():
    """Return the imported ``calphy`` module or raise an actionable error."""
    try:
        import calphy
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"calphy is required for free-energy workflows but is not "
            f"installed.\nInstall with: {_INSTALL_HINT}"
        ) from exc
    if calphy is None:  # monkeypatched in tests
        raise ModuleNotFoundError(
            f"calphy is required for free-energy workflows but is not "
            f"installed.\nInstall with: {_INSTALL_HINT}"
        )
    return calphy


def _require_lammps_engine():
    """Return ``LammpsEngine`` or raise an actionable error."""
    try:
        from pyiron_workflow_lammps.engine import LammpsEngine
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"pyiron_workflow_lammps is required for free-energy workflows "
            f"but is not installed.\nInstall with: {_INSTALL_HINT}"
        ) from exc
    import sys
    if sys.modules.get("pyiron_workflow_lammps") is None:
        raise ModuleNotFoundError(
            f"pyiron_workflow_lammps is required for free-energy workflows "
            f"but is not installed.\nInstall with: {_INSTALL_HINT}"
        )
    return LammpsEngine
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v
```

Expected: all three tests pass (the `_require_calphy_returns_module_when_present` test will be skipped if calphy is not installed).

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/ tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): lazy-import shims for calphy / pyiron_workflow_lammps

Adds physics/free_energy/_compat.py with _require_calphy() and
_require_lammps_engine(). Both raise ModuleNotFoundError with the
exact pip install line when the [free-energy] extra is missing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `LammpsPotential` input dataclass

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/inputs.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_free_energy.py`:

```python
# ---------------------------------------------------------------------------
# LammpsPotential
# ---------------------------------------------------------------------------


def test_lammps_potential_required_fields():
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /path/to/Cu.eam.alloy Cu")
    assert pot.pair_style == "eam/alloy"
    assert pot.pair_coeff == "* * /path/to/Cu.eam.alloy Cu"
    assert pot.potential_file is None


def test_lammps_potential_optional_file():
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(
        pair_style="pace",
        pair_coeff="* * /path/to/pot.yace Cu",
        potential_file="/path/to/extra.txt",
    )
    assert pot.potential_file == "/path/to/extra.txt"


def test_lammps_potential_picklable():
    import pickle
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /path/to/Cu.eam.alloy Cu")
    restored = pickle.loads(pickle.dumps(pot))
    assert restored == pot
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k LammpsPotential
```

Expected: `ModuleNotFoundError: ... .inputs`.

- [ ] **Step 3: Implement `inputs.py`**

Create `pyiron_workflow_atomistics/physics/free_energy/inputs.py`:

```python
"""Physics-level input dataclasses for calphy free-energy workflows.

Only the potential lives here. The structure is `ase.Atoms` (no dataclass
needed), and the LAMMPS launcher comes from ``LammpsEngine.command``,
parsed by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LammpsPotential:
    """LAMMPS interatomic potential, passed verbatim to calphy.

    Attributes
    ----------
    pair_style
        LAMMPS ``pair_style`` line, e.g. ``"eam/alloy"``, ``"pace"``,
        ``"grace"``.
    pair_coeff
        LAMMPS ``pair_coeff`` line, e.g.
        ``"* * /path/to/Cu01.eam.alloy Cu"``. Element ordering must
        match the structure's chemical-symbol first-occurrence order.
    potential_file
        Optional auxiliary potential file path (some potentials require
        one); passed to ``calphy.input.Calculation.potential_file``.
    """

    pair_style: str
    pair_coeff: str
    potential_file: str | None = None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k LammpsPotential
```

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/inputs.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): LammpsPotential input dataclass

A simple dataclass for pair_style + pair_coeff + optional
potential_file. Passed verbatim to calphy.input.Calculation; this
package does not parse or rewrite the strings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `FreeEnergyOutput` dataclass

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/outputs.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/physics/test_free_energy.py`:

```python
# ---------------------------------------------------------------------------
# FreeEnergyOutput
# ---------------------------------------------------------------------------


def test_free_energy_output_required_fields():
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="fe",
        reference_phase="solid",
        free_energy=-3.5,
        free_energy_error=0.01,
        temperature=300.0,
        pressure=0.0,
        n_atoms=108,
        elements=["Cu"],
        simfolder="/tmp/fe",
        report={"results": {"free_energy": -3.5}},
    )
    assert out.mode == "fe"
    assert out.free_energy == -3.5
    assert out.temperature_array is None
    assert out.melting_temperature is None


def test_free_energy_output_optional_fields_default_to_none():
    from dataclasses import fields, MISSING
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    optional_names = {
        "temperature_array",
        "free_energy_array",
        "pressure_array",
        "melting_temperature",
        "melting_temperature_error",
        "composition_path",
        "einstein_free_energy",
    }
    for f in fields(FreeEnergyOutput):
        if f.name in optional_names:
            assert f.default is None, f"{f.name} should default to None"


def test_free_energy_output_to_dict_round_trip():
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="ts",
        reference_phase="solid",
        free_energy=-3.5,
        free_energy_error=0.01,
        temperature=300.0,
        pressure=0.0,
        n_atoms=108,
        elements=["Cu"],
        simfolder="/tmp/ts",
        report={"results": {"free_energy": -3.5}},
    )
    d = out.to_dict()
    assert d["mode"] == "ts"
    assert d["temperature_array"] is None


def test_free_energy_output_picklable():
    import pickle
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="fe",
        reference_phase="solid",
        free_energy=-3.5,
        free_energy_error=0.01,
        temperature=300.0,
        pressure=0.0,
        n_atoms=108,
        elements=["Cu"],
        simfolder="/tmp/fe",
        report={},
    )
    restored = pickle.loads(pickle.dumps(out))
    assert restored.free_energy == -3.5
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k FreeEnergyOutput
```

Expected: `ModuleNotFoundError: ... .outputs`.

- [ ] **Step 3: Implement `outputs.py`**

Create `pyiron_workflow_atomistics/physics/free_energy/outputs.py`:

```python
"""Structured result of a calphy free-energy calculation.

Same shape regardless of which mode produced it; per-mode arrays
(temperature_array, free_energy_array, melting_temperature, ...) are
None when not applicable. Pickleable — holds plain Python + numpy
types only, no Phase references, no LAMMPS handles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class FreeEnergyOutput:
    """Result of one calphy free-energy calculation.

    Units
    -----
    ``free_energy`` / ``free_energy_error`` / ``einstein_free_energy``:
    eV/atom (calphy native).
    ``temperature``: K.
    ``pressure``: bar (calphy native — differs from
    :class:`pyiron_workflow_atomistics.engine.CalcInputMD.pressure`
    which is Pa; do not mix).
    """

    mode: Literal["fe", "ts", "tscale", "pscale",
                  "melting_temperature", "alchemy", "composition_scaling"]
    reference_phase: Literal["solid", "liquid", "both"]
    free_energy: float
    free_energy_error: float
    temperature: float
    pressure: float
    n_atoms: int
    elements: list[str]
    simfolder: str
    report: dict[str, Any]

    # mode-specific; None when not applicable
    temperature_array: np.ndarray | None = None
    free_energy_array: np.ndarray | None = None
    pressure_array: np.ndarray | None = None
    melting_temperature: float | None = None
    melting_temperature_error: float | None = None
    composition_path: list[dict[str, int]] | None = None
    einstein_free_energy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of the dataclass fields."""
        return asdict(self)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k FreeEnergyOutput
```

Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/outputs.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): FreeEnergyOutput typed result dataclass

One dataclass per mode would proliferate; one shared dataclass with
mode-specific optional fields keeps post-processing generic. Pickle-
able by construction (plain Python + numpy types only).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `_split_lammps_command` LAMMPS-launcher parser

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _split_lammps_command
# ---------------------------------------------------------------------------


import pytest


@pytest.mark.parametrize("cmd, expected", [
    ("lmp", ("lmp", None, 1)),
    ("lmp -in in.lmp -log log.lammps", ("lmp", None, 1)),
    ("mpirun -np 4 lmp", ("lmp", "mpirun", 4)),
    ("mpiexec -n 8 lmp -in in.lmp -log log.lammps", ("lmp", "mpiexec", 8)),
    ("srun -n 16 lmp", ("lmp", "srun", 16)),
    ("mpirun --bind-to none -np 2 lmp",
     ("lmp", "mpirun --bind-to none", 2)),
    ("/opt/lammps/bin/lmp_mpi -in in.lmp",
     ("/opt/lammps/bin/lmp_mpi", None, 1)),
])
def test_split_lammps_command_valid(cmd, expected):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _split_lammps_command,
    )

    assert _split_lammps_command(cmd) == expected


@pytest.mark.parametrize("cmd", [
    "mpirun -np 4 lmp -unknown-flag x",
    "lmp -partition 2x2",
    "lmp -screen none",
])
def test_split_lammps_command_rejects_unknown_tokens(cmd):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _split_lammps_command,
    )

    with pytest.raises(ValueError, match="Unrecognized tokens"):
        _split_lammps_command(cmd)


def test_split_lammps_command_rejects_empty():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _split_lammps_command,
    )

    with pytest.raises(ValueError):
        _split_lammps_command("")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k split_lammps_command
```

Expected: `ModuleNotFoundError: ... ._calphy_adapter`.

- [ ] **Step 3: Implement `_calphy_adapter.py` with the parser**

Create `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`:

```python
"""Internal helpers connecting pyiron_workflow_atomistics to calphy.

Public callers should not import from this module — use the function
nodes in ``physics/free_energy/calphy.py``. Everything here is
considered private API.
"""

from __future__ import annotations

import shlex

_LAUNCHERS = ("mpirun", "mpiexec", "srun")
_LAUNCHER_CORE_FLAGS = ("-np", "-n")
_BINARY_TAIL_FLAGS = ("-in", "-log")


def _split_lammps_command(cmd: str) -> tuple[str, str | None, int]:
    """Parse a LAMMPS launcher command into (binary, mpi_executable, cores).

    Accepts:
      "lmp"                                          -> ("lmp", None, 1)
      "lmp -in in.lmp -log log.lammps"               -> ("lmp", None, 1)
      "mpirun -np 4 lmp"                             -> ("lmp", "mpirun", 4)
      "mpirun --bind-to none -np 2 lmp"              -> ("lmp", "mpirun --bind-to none", 2)

    Rejects any tokens other than recognised launcher flags, `-np`/`-n`,
    and the trailing `-in <file>` / `-log <file>` pair (which calphy
    overwrites with its own paths anyway).
    """
    if not cmd or not cmd.strip():
        raise ValueError("Empty LAMMPS command")

    tokens = shlex.split(cmd)
    i = 0

    mpi_parts: list[str] = []
    cores = 1
    has_launcher = tokens[0] in _LAUNCHERS

    if has_launcher:
        mpi_parts.append(tokens[i])
        i += 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in _LAUNCHER_CORE_FLAGS:
                if i + 1 >= len(tokens):
                    raise ValueError(f"Missing core count after {tok!r}")
                try:
                    cores = int(tokens[i + 1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid core count after {tok!r}: {tokens[i + 1]!r}"
                    ) from exc
                i += 2
                continue
            if tok.startswith("-"):
                # generic launcher flag with or without value
                mpi_parts.append(tok)
                # Heuristic: consume a value if the next token doesn't start with '-'.
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    # Only consume if it's NOT the LAMMPS binary. Binaries
                    # are named "lmp", "lmp_*", or end with "/lmp" or
                    # similar; flag values are arbitrary strings. The
                    # convention this parser supports is that launcher
                    # flag values come before the binary, so we treat
                    # the FIRST non-dash, non-binary-looking token as the
                    # value and break to look for the binary.
                    candidate = tokens[i + 1]
                    if _looks_like_lammps_binary(candidate):
                        i += 1
                        break
                    mpi_parts.append(candidate)
                    i += 2
                    continue
                i += 1
                continue
            # First non-launcher, non-flag token is the binary
            break

    if i >= len(tokens):
        raise ValueError(f"No LAMMPS binary in command: {cmd!r}")

    binary = tokens[i]
    if not _looks_like_lammps_binary(binary):
        raise ValueError(
            f"Token {binary!r} does not look like a LAMMPS binary "
            f"(expected name containing 'lmp' or 'lammps')"
        )
    i += 1

    # Tail may contain only `-in <file>` and `-log <file>`; calphy will
    # replace these with its own paths. Anything else is rejected.
    unknown: list[str] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok in _BINARY_TAIL_FLAGS:
            if i + 1 >= len(tokens):
                unknown.append(tok)
                break
            i += 2
            continue
        unknown.append(tok)
        i += 1
    if unknown:
        raise ValueError(
            f"Unrecognized tokens in LammpsEngine.command: {unknown}. "
            f"The calphy adapter only accepts launcher + binary + "
            f"optional -in/-log; everything else is dropped silently "
            f"by calphy and the adapter refuses rather than mislead."
        )

    return binary, " ".join(mpi_parts) if mpi_parts else None, cores


def _looks_like_lammps_binary(token: str) -> bool:
    """Heuristic: a LAMMPS binary's last path component contains 'lmp' or 'lammps'."""
    name = token.rsplit("/", 1)[-1].lower()
    return "lmp" in name or "lammps" in name
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k split_lammps_command
```

Expected: all parametrised cases pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): LAMMPS launcher command parser

_split_lammps_command splits LammpsEngine.command into (binary,
mpi_executable, cores). Handles bare lmp, mpirun/mpiexec/srun with
-np/-n, and arbitrary launcher flags before the binary. Rejects any
trailing flags other than -in/-log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `_validate_engine_only_command`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _validate_engine_only_command
# ---------------------------------------------------------------------------


lammps_engine = pytest.importorskip("pyiron_workflow_lammps.engine")


def _make_minimal_engine():
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_lammps.engine import LammpsEngine

    return LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")


def test_validate_engine_only_command_passes_for_minimal_engine():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    _validate_engine_only_command(eng)  # should not raise


def test_validate_engine_only_command_rejects_raw_script():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.raw_script = "run 1000"
    with pytest.raises(ValueError, match=r"raw_script"):
        _validate_engine_only_command(eng)


def test_validate_engine_only_command_rejects_path_to_model():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.path_to_model = "/real/model"
    with pytest.raises(ValueError, match=r"path_to_model"):
        _validate_engine_only_command(eng)


def test_validate_engine_only_command_rejects_pair_style():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.input_script_pair_style = "eam/alloy"
    with pytest.raises(ValueError, match=r"input_script_pair_style"):
        _validate_engine_only_command(eng)


def test_validate_engine_only_command_command_can_be_changed():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.command = "mpirun -np 8 lmp"
    _validate_engine_only_command(eng)  # should not raise


def test_validate_engine_only_command_working_directory_carveout():
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_engine_only_command,
    )

    eng = _make_minimal_engine()
    eng.working_directory = "/somewhere/else"
    _validate_engine_only_command(eng)  # working_directory is in the carve-out set
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k validate_engine_only_command
```

Expected: `ImportError` or `AttributeError` — function not defined.

- [ ] **Step 3: Implement the validator**

Append to `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`:

```python
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiron_workflow_lammps.engine import LammpsEngine


_ENGINE_CARVE_OUTS = frozenset({
    "EngineInput",        # required to construct; ignored by calphy
    "mode",               # init=False, derived from EngineInput
    "working_directory",  # adapter sets its own simfolder
    "command",            # the one field we actually use
})


def _validate_engine_only_command(engine: "LammpsEngine") -> None:
    """Refuse LammpsEngine fields that the calphy adapter cannot honor.

    Only ``engine.command`` is consumed. Every other field that has been
    changed from its dataclass default is rejected, because calphy
    generates its own LAMMPS input from the supplied LammpsPotential
    and silently ignores everything else on the engine.
    """
    overridden: list[str] = []
    for f in fields(engine):
        if f.name in _ENGINE_CARVE_OUTS:
            continue
        default = _resolve_default(f)
        current = getattr(engine, f.name)
        if not _equals_default(current, default):
            overridden.append(f.name)

    if overridden:
        raise ValueError(
            f"calphy adapter only reads LammpsEngine.command. The "
            f"following non-default fields were set: {sorted(overridden)}. "
            f"calphy generates its own LAMMPS input; setting these "
            f"silently has no effect on the free-energy result. Pass "
            f"the potential via `potential=LammpsPotential(...)` and "
            f"construct a minimal engine:\n"
            f"  LammpsEngine(\n"
            f"      EngineInput=CalcInputStatic(),\n"
            f"      command='mpirun -np 4 lmp',\n"
            f"  )"
        )


def _resolve_default(f):
    """Return a dataclass field's default value, evaluating default_factory."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return MISSING


def _equals_default(current, default) -> bool:
    """Tolerant equality: MISSING means no default → cannot judge → accept."""
    if default is MISSING:
        return True
    return current == default
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k validate_engine_only_command
```

Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): _validate_engine_only_command guard

Iterates dataclass fields on LammpsEngine; rejects anything non-
default except the carve-outs (EngineInput, mode, working_directory,
command). The error message includes the offending field names and
a minimal-engine code example.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `_validate_structure`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _validate_structure
# ---------------------------------------------------------------------------


def test_validate_structure_accepts_cubic_bulk(fcc_al_atoms):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    _validate_structure(fcc_al_atoms)  # should not raise


def test_validate_structure_rejects_empty():
    from ase import Atoms
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    with pytest.raises(ValueError, match=r"empty"):
        _validate_structure(Atoms())


def test_validate_structure_rejects_mixed_pbc(fcc_al_atoms):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    a = fcc_al_atoms.copy()
    a.pbc = (True, True, False)
    with pytest.raises(ValueError, match=r"PBC"):
        _validate_structure(a)


def test_validate_structure_rejects_zero_volume(fcc_al_atoms):
    from ase import Atoms
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _validate_structure,
    )

    a = Atoms("Cu", positions=[[0, 0, 0]], cell=[[0, 0, 0]] * 3, pbc=True)
    with pytest.raises(ValueError, match=r"non-positive volume"):
        _validate_structure(a)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k validate_structure
```

Expected: `ImportError` — `_validate_structure` not defined.

- [ ] **Step 3: Implement the validator**

Append to `_calphy_adapter.py`:

```python
from ase import Atoms


def _validate_structure(structure: Atoms) -> None:
    """Refuse structures calphy cannot consume meaningfully.

    calphy assumes a fully periodic 3D supercell with positive volume
    and at least one atom. Anything else either crashes calphy mid-run
    or silently produces meaningless free energies (open boundaries
    don't have a well-defined Frenkel-Ladd reference).
    """
    if len(structure) == 0:
        raise ValueError("structure is empty (zero atoms)")
    pbc = tuple(bool(p) for p in structure.pbc)
    if pbc != (True, True, True):
        raise ValueError(
            f"calphy free-energy workflows require fully periodic 3D "
            f"PBC; got pbc={pbc}"
        )
    if structure.get_volume() <= 0.0:
        raise ValueError(
            f"structure has non-positive volume ({structure.get_volume()}); "
            f"calphy will refuse to integrate against it"
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k validate_structure
```

Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): _validate_structure boundary checks

Rejects empty atoms, non-3D PBC, and degenerate cells before calphy
gets the chance to crash mid-run. Catches the failure where the user
sees the clearest message.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `_build_calphy_calculation` — `fe` mode skeleton + data-file writer

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _build_calphy_calculation — fe mode
# ---------------------------------------------------------------------------


def test_build_calphy_calculation_fe_basic(tmp_path, fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _build_calphy_calculation,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    eng = _make_minimal_engine()
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff="* * /tmp/Al.eam.alloy Al",
    )
    calc = _build_calphy_calculation(
        mode="fe",
        structure=fcc_al_atoms,
        potential=pot,
        lammps_engine=eng,
        working_directory=str(tmp_path),
        temperature=300.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
        n_iterations=1,
        npt=True,
        equilibration_control="nose-hoover",
    )
    assert calc.mode == "fe"
    assert calc.element == ["Al"]
    assert calc.pair_style == ["eam/alloy"]
    assert calc.pair_coeff == ["* * /tmp/Al.eam.alloy Al"]
    assert calc.script_mode is True
    assert calc.lammps_executable == "lmp"
    assert calc.mpi_executable is None
    assert calc.reference_phase == "solid"
    assert calc.npt is True
    # data file must exist in working_directory
    assert (tmp_path / "lammps.data").exists()
    assert str(tmp_path / "lammps.data") in calc.lattice
    assert calc.file_format == "lammps-data"


def test_build_calphy_calculation_fe_passes_mpi_command(tmp_path, fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _build_calphy_calculation,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    eng = _make_minimal_engine()
    eng.command = "mpirun -np 8 lmp"
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    calc = _build_calphy_calculation(
        mode="fe",
        structure=fcc_al_atoms,
        potential=pot,
        lammps_engine=eng,
        working_directory=str(tmp_path),
        temperature=300.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
        n_iterations=1,
        npt=True,
        equilibration_control="nose-hoover",
    )
    assert calc.lammps_executable == "lmp"
    assert calc.mpi_executable == "mpirun"
    # cores is captured on queue.cores
    assert calc.queue.cores == 8


def test_build_calphy_calculation_writes_data_file_matches_structure(
    tmp_path, fcc_al_atoms
):
    pytest.importorskip("calphy")
    from ase.io import read as ase_read
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _build_calphy_calculation,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    eng = _make_minimal_engine()
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    _build_calphy_calculation(
        mode="fe",
        structure=fcc_al_atoms,
        potential=pot,
        lammps_engine=eng,
        working_directory=str(tmp_path),
        temperature=300.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
        n_iterations=1,
        npt=True,
        equilibration_control="nose-hoover",
    )
    round_trip = ase_read(
        str(tmp_path / "lammps.data"),
        format="lammps-data",
        style="atomic",
    )
    import numpy as np
    np.testing.assert_allclose(round_trip.get_positions(),
                               fcc_al_atoms.get_positions(),
                               atol=1e-8)
    np.testing.assert_allclose(round_trip.get_cell(),
                               fcc_al_atoms.get_cell(),
                               atol=1e-8)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k build_calphy_calculation
```

Expected: `ImportError` — function not defined.

- [ ] **Step 3: Implement the builder (fe-only for now)**

Append to `_calphy_adapter.py`:

```python
import os
from typing import Any, Literal

from ase.data import atomic_masses, atomic_numbers
from ase.io import write as ase_write


def _atoms_element_order(structure: Atoms) -> list[str]:
    """Same first-occurrence ordering rule LammpsEngine uses."""
    return list(dict.fromkeys(structure.get_chemical_symbols()))


def _atoms_to_lammps_data(structure: Atoms, path: str,
                          element_order: list[str]) -> None:
    """Write a LAMMPS data file with type ordering matching element_order."""
    # ase >=3.23 honors `specorder` for lammps-data writes; older versions
    # silently use alphabetical order. We require ase==3.28.0 (see
    # pyproject.toml) so specorder is guaranteed available.
    ase_write(path, structure, format="lammps-data",
              specorder=element_order, atom_style="atomic")


def _build_calphy_calculation(
    *,
    mode: Literal["fe", "ts", "tscale", "pscale",
                  "melting_temperature", "alchemy", "composition_scaling"],
    structure: Atoms,
    potential,                 # LammpsPotential
    lammps_engine,             # LammpsEngine
    working_directory: str,
    # mode-generic kwargs (each public node forwards the subset it needs)
    temperature: float | None = None,
    temperature_range: tuple[float, float] | None = None,
    pressure: float = 0.0,
    pressure_range: tuple[float, float] | None = None,
    reference_phase: Literal["solid", "liquid"] | None = None,
    temperature_guess: float | None = None,
    melting_step: int = 200,
    melting_max_attempts: int = 5,
    pair_style_target: str | None = None,
    pair_coeff_target: str | None = None,
    output_chemical_composition: dict[str, int] | None = None,
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
):
    """Build a ``calphy.input.Calculation`` from the user kwargs.

    Returns the validated Pydantic model, ready to feed to
    :func:`calphy.kernel.setup_calculation`. Writes the LAMMPS data
    file to ``{working_directory}/lammps.data`` as a side effect.
    """
    calphy = __import_calphy()

    element_order = _atoms_element_order(structure)
    mass = [float(atomic_masses[atomic_numbers[s]]) for s in element_order]

    data_path = os.path.join(working_directory, "lammps.data")
    os.makedirs(working_directory, exist_ok=True)
    _atoms_to_lammps_data(structure, data_path, element_order)

    binary, mpi, cores = _split_lammps_command(lammps_engine.command)

    kwargs: dict[str, Any] = {
        "mode": mode,
        "element": element_order,
        "mass": mass,
        "lattice": data_path,
        "file_format": "lammps-data",
        "pair_style": potential.pair_style,
        "pair_coeff": potential.pair_coeff,
        "potential_file": potential.potential_file,
        "n_equilibration_steps": n_equilibration_steps,
        "n_switching_steps": n_switching_steps,
        "n_iterations": n_iterations,
        "npt": npt,
        "equilibration_control": (
            "nose_hoover" if equilibration_control == "nose-hoover"
            else equilibration_control
        ),
        "script_mode": True,
        "lammps_executable": binary,
        "mpi_executable": mpi,
        "queue": {
            "scheduler": "local",
            "cores": cores,
        },
    }

    # Per-mode kwarg routing
    if mode == "fe":
        if temperature is None:
            raise ValueError("free_energy requires `temperature`")
        if reference_phase is None:
            raise ValueError("free_energy requires `reference_phase`")
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = reference_phase
    elif mode == "ts":
        if temperature_range is None or len(temperature_range) != 2:
            raise ValueError(
                "reversible_scaling_temperature requires "
                "`temperature_range=(lo, hi)`"
            )
        if reference_phase is None:
            raise ValueError("`reference_phase` is required")
        kwargs["temperature"] = [float(temperature_range[0]),
                                 float(temperature_range[1])]
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = reference_phase
    elif mode == "pscale":
        if pressure_range is None or len(pressure_range) != 2:
            raise ValueError(
                "reversible_scaling_pressure requires "
                "`pressure_range=(lo, hi)`"
            )
        if temperature is None or reference_phase is None:
            raise ValueError("`temperature` and `reference_phase` are required")
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = [float(pressure_range[0]),
                              float(pressure_range[1])]
        kwargs["reference_phase"] = reference_phase
    elif mode == "melting_temperature":
        if temperature_guess is not None:
            if temperature_guess <= 0:
                raise ValueError("`temperature_guess` must be positive")
            kwargs["temperature"] = float(temperature_guess)
        kwargs["pressure"] = float(pressure)
        kwargs["melting_temperature"] = {
            "step": int(melting_step),
            "attempts": int(melting_max_attempts),
        }
    elif mode == "alchemy":
        if (temperature is None or pair_style_target is None
                or pair_coeff_target is None):
            raise ValueError(
                "alchemy requires `temperature`, `pair_style_target`, "
                "and `pair_coeff_target`"
            )
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = "solid"
        kwargs["pair_style"] = [potential.pair_style, pair_style_target]
        kwargs["pair_coeff"] = [potential.pair_coeff, pair_coeff_target]
    elif mode == "composition_scaling":
        if temperature is None or output_chemical_composition is None:
            raise ValueError(
                "composition_scaling requires `temperature` and "
                "`output_chemical_composition`"
            )
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = "solid"
        kwargs["composition_scaling"] = {
            "output_chemical_composition": [dict(output_chemical_composition)],
        }
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return calphy.input.Calculation(**kwargs)


def __import_calphy():
    """Import calphy lazily so the adapter module is importable without it."""
    from pyiron_workflow_atomistics.physics.free_energy._compat import (
        _require_calphy,
    )

    return _require_calphy()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k build_calphy_calculation
```

Expected: all three tests pass (skipped without calphy installed).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): _build_calphy_calculation per-mode kwarg fan-out

One builder for all six modes. Writes the structure as a lammps-data
file with element ordering matching pair_coeff, parses the engine
command into lammps_executable/mpi_executable/cores, and routes each
mode's kwargs onto calphy.input.Calculation. Tests cover fe; other
modes are exercised in Tasks 13-17.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `_run_calphy_job` and `_pack_free_energy_output`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _pack_free_energy_output
# ---------------------------------------------------------------------------


def test_pack_free_energy_output_fe_minimal(fcc_al_atoms, tmp_path):
    from types import SimpleNamespace
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _pack_free_energy_output,
    )

    fake_report = {
        "results": {
            "free_energy": -3.5,
            "error": 0.01,
            "einstein_crystal": -3.7,
        },
        "input": {
            "temperature": 300.0,
            "pressure": 0.0,
        },
    }
    fake_job = SimpleNamespace(simfolder=str(tmp_path))
    out = _pack_free_energy_output(
        mode="fe",
        job=fake_job,
        report=fake_report,
        simfolder=str(tmp_path),
        structure=fcc_al_atoms,
        reference_phase="solid",
        temperature=300.0,
        pressure=0.0,
    )
    assert out.mode == "fe"
    assert out.free_energy == -3.5
    assert out.free_energy_error == 0.01
    assert out.einstein_free_energy == -3.7
    assert out.temperature == 300.0
    assert out.pressure == 0.0
    assert out.n_atoms == len(fcc_al_atoms)
    assert out.elements == ["Al"]
    assert out.simfolder == str(tmp_path)
    assert out.report is fake_report


def test_pack_free_energy_output_melting_temperature(fcc_al_atoms, tmp_path):
    from types import SimpleNamespace
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _pack_free_energy_output,
    )

    fake_job = SimpleNamespace(simfolder=str(tmp_path), tm=1357.0, dtm=15.0)
    fake_report = {"results": {"free_energy": 0.0, "error": 0.0}}
    out = _pack_free_energy_output(
        mode="melting_temperature",
        job=fake_job,
        report=fake_report,
        simfolder=str(tmp_path),
        structure=fcc_al_atoms,
        reference_phase="both",
        temperature=1357.0,
        pressure=0.0,
    )
    assert out.melting_temperature == 1357.0
    assert out.melting_temperature_error == 15.0


# ---------------------------------------------------------------------------
# _run_calphy_job — mocked
# ---------------------------------------------------------------------------


def test_run_calphy_job_dispatches_and_reads_report(monkeypatch, tmp_path):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.physics.free_energy import _calphy_adapter

    # Fake job with a simfolder containing a report.yaml
    import yaml
    sim = tmp_path / "sim"
    sim.mkdir()
    fake_report = {"results": {"free_energy": -3.5, "error": 0.01}}
    (sim / "report.yaml").write_text(yaml.safe_dump(fake_report))

    from types import SimpleNamespace
    fake_job = SimpleNamespace(simfolder=str(sim),
                               calc=SimpleNamespace(mode="fe"))

    captured = {}

    def fake_setup(calc):
        captured["setup"] = True
        return fake_job

    def fake_run(job):
        captured["run"] = True
        return job

    monkeypatch.setattr(_calphy_adapter, "_setup_calculation", fake_setup)
    monkeypatch.setattr(_calphy_adapter, "_run_calculation", fake_run)

    fake_calc = SimpleNamespace(mode="fe")
    job, report = _calphy_adapter._run_calphy_job(fake_calc)
    assert captured == {"setup": True, "run": True}
    assert report == fake_report
    assert job is fake_job
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k "pack_free_energy_output or run_calphy_job"
```

Expected: `AttributeError` — functions not defined.

- [ ] **Step 3: Implement the runner and packer**

Append to `_calphy_adapter.py`:

```python
def _setup_calculation(calc):
    """Indirection seam so tests can monkeypatch without importing calphy."""
    calphy = __import_calphy()
    return calphy.kernel.setup_calculation(calc)


def _run_calculation(job):
    """Same indirection seam for ``calphy.kernel.run_calculation``."""
    calphy = __import_calphy()
    return calphy.kernel.run_calculation(job)


def _run_calphy_job(calc):
    """Dispatch a built ``Calculation`` through calphy.

    Returns ``(job, report)`` where ``report`` is the parsed
    ``report.yaml`` from ``job.simfolder``. Reading the file rather
    than scraping live attributes survives calphy minor-version
    changes that move fields around.
    """
    import yaml as _yaml
    job = _setup_calculation(calc)
    job = _run_calculation(job)
    report_path = os.path.join(job.simfolder, "report.yaml")
    try:
        with open(report_path) as f:
            report = _yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"calphy did not produce a report.yaml in {job.simfolder}; "
            f"the run may have failed silently. Inspect that directory "
            f"for partial artefacts."
        ) from exc
    return job, report


def _pack_free_energy_output(
    *,
    mode: str,
    job,
    report: dict,
    simfolder: str,
    structure: Atoms,
    reference_phase: str,
    temperature: float,
    pressure: float,
):
    """Build a :class:`FreeEnergyOutput` from a finished calphy job."""
    from pyiron_workflow_atomistics.physics.free_energy.outputs import (
        FreeEnergyOutput,
    )

    results = (report or {}).get("results", {})

    out = FreeEnergyOutput(
        mode=mode,
        reference_phase=reference_phase,
        free_energy=float(results.get("free_energy", float("nan"))),
        free_energy_error=float(results.get("error", float("nan"))),
        temperature=float(temperature),
        pressure=float(pressure),
        n_atoms=len(structure),
        elements=_atoms_element_order(structure),
        simfolder=os.path.abspath(simfolder),
        report=report or {},
    )

    if mode == "fe":
        einstein = results.get("einstein_crystal")
        if einstein is not None:
            out.einstein_free_energy = float(einstein)
    elif mode in ("ts", "tscale"):
        try:
            t_arr, f_arr = _load_rs_curve(simfolder)
            out.temperature_array = t_arr
            out.free_energy_array = f_arr
        except FileNotFoundError:
            # calphy already wrote `report.yaml` so the run succeeded;
            # the curve file is simply absent. Leave the arrays None.
            pass
    elif mode == "pscale":
        try:
            p_arr, f_arr = _load_rs_curve(simfolder, axis="pressure")
            out.pressure_array = p_arr
            out.free_energy_array = f_arr
        except FileNotFoundError:
            pass
    elif mode == "melting_temperature":
        out.melting_temperature = (
            float(job.tm) if getattr(job, "tm", None) is not None else None
        )
        out.melting_temperature_error = (
            float(job.dtm) if getattr(job, "dtm", None) is not None else None
        )
    elif mode == "composition_scaling":
        # report["input"]["composition_scaling"] holds the dict-list calphy used
        try:
            out.composition_path = list(
                (report.get("input", {})
                 .get("composition_scaling", {})
                 .get("output_chemical_composition", []))
            )
        except Exception:
            out.composition_path = None

    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k "pack_free_energy_output or run_calphy_job"
```

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): _run_calphy_job + _pack_free_energy_output

_run_calphy_job dispatches calphy.kernel.setup_calculation and
run_calculation through monkeypatch-friendly indirections, then
reads report.yaml from job.simfolder. _pack_free_energy_output
builds a FreeEnergyOutput from the report + structure, including
per-mode optional fields (einstein_crystal, melting_temperature,
composition_path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `_load_rs_curve` reversible-scaling curve reader

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _load_rs_curve
# ---------------------------------------------------------------------------


def test_load_rs_curve_temperature(tmp_path):
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _load_rs_curve,
    )

    # calphy writes "temperature_sweep.dat" with cols [T, F]
    data = np.array([[100.0, -3.50],
                     [200.0, -3.55],
                     [300.0, -3.60]])
    np.savetxt(tmp_path / "temperature_sweep.dat", data)

    t, f = _load_rs_curve(str(tmp_path))
    np.testing.assert_allclose(t, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(f, [-3.50, -3.55, -3.60])


def test_load_rs_curve_pressure(tmp_path):
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _load_rs_curve,
    )

    data = np.array([[0.0, -3.50],
                     [5000.0, -3.45],
                     [10000.0, -3.40]])
    np.savetxt(tmp_path / "pressure_sweep.dat", data)

    p, f = _load_rs_curve(str(tmp_path), axis="pressure")
    np.testing.assert_allclose(p, [0.0, 5000.0, 10000.0])
    np.testing.assert_allclose(f, [-3.50, -3.45, -3.40])


def test_load_rs_curve_missing_raises(tmp_path):
    from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
        _load_rs_curve,
    )

    with pytest.raises(FileNotFoundError):
        _load_rs_curve(str(tmp_path))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k load_rs_curve
```

Expected: `AttributeError` — function not defined.

- [ ] **Step 3: Implement the loader**

Append to `_calphy_adapter.py`:

```python
import glob

import numpy as np


def _load_rs_curve(
    simfolder: str,
    axis: Literal["temperature", "pressure"] = "temperature",
) -> tuple[np.ndarray, np.ndarray]:
    """Read calphy's reversible-scaling sweep file.

    Calphy writes ``temperature_sweep.dat`` for ``ts``/``tscale`` and
    ``pressure_sweep.dat`` for ``pscale``. Both are two-column whitespace-
    separated: independent variable, then free energy.

    Returns ``(x_array, free_energy_array)`` as numpy arrays.

    Raises
    ------
    FileNotFoundError
        If the expected sweep file is not present in ``simfolder`` —
        either calphy didn't run, or its filename convention changed.
    """
    expected = f"{axis}_sweep.dat"
    path = os.path.join(simfolder, expected)
    if not os.path.exists(path):
        # Tolerate calphy's older convention `*free_energy*.dat`
        candidates = glob.glob(os.path.join(simfolder, "*sweep*.dat"))
        if not candidates:
            raise FileNotFoundError(
                f"No reversible-scaling sweep file in {simfolder} "
                f"(looked for {expected} and *sweep*.dat)"
            )
        path = candidates[0]
    data = np.loadtxt(path)
    if data.ndim == 1:  # single row
        data = data[None, :]
    return data[:, 0], data[:, 1]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k load_rs_curve
```

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/_calphy_adapter.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): _load_rs_curve reversible-scaling reader

Reads calphy's temperature_sweep.dat / pressure_sweep.dat (two-column
whitespace files). Falls back to a glob for forward-compat with
calphy's older naming. FileNotFoundError if nothing matches — the
caller treats this as 'no curve data, leave arrays None'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Public node `free_energy` + Tier 1 unit tests

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/calphy.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# free_energy public node — Tier 1 (no calphy / no LAMMPS)
# ---------------------------------------------------------------------------


def test_free_energy_node_raises_when_calphy_missing(monkeypatch, fcc_al_atoms):
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    monkeypatch.setitem(sys.modules, "calphy", None)
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ModuleNotFoundError, match=r"pip install"):
        free_energy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            reference_phase="solid",
        )


def test_free_energy_node_rejects_non_default_engine_field(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    eng.raw_script = "run 1000"
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"raw_script"):
        free_energy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            reference_phase="solid",
        )


def test_free_energy_node_rejects_non_periodic_structure(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    s = fcc_al_atoms.copy()
    s.pbc = (True, True, False)
    with pytest.raises(ValueError, match=r"PBC"):
        free_energy.node_function(
            structure=s,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            reference_phase="solid",
        )


def test_free_energy_node_restores_cwd_on_error(monkeypatch, tmp_path,
                                                fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy import _calphy_adapter
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")

    def boom(_calc):
        raise RuntimeError("calphy exploded mid-run")

    monkeypatch.setattr(_calphy_adapter, "_run_calphy_job", boom)
    monkeypatch.chdir(tmp_path)
    cwd_before = os.getcwd()
    with pytest.raises(RuntimeError, match="calphy exploded"):
        free_energy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            working_directory=str(tmp_path),
            temperature=300.0,
            reference_phase="solid",
        )
    assert os.getcwd() == cwd_before
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k free_energy_node
```

Expected: `ModuleNotFoundError: ... .calphy` (the public-nodes module).

- [ ] **Step 3: Implement the `free_energy` public node**

Create `pyiron_workflow_atomistics/physics/free_energy/calphy.py`:

```python
"""Public function-nodes — one per calphy mode.

Each node:
  1. Asserts the [free-energy] extra is installed via _require_*.
  2. Validates the engine has only its `command` set, and the structure
     is a 3D fully-periodic supercell.
  3. Creates ``working_directory/subdir/`` and chdirs into it.
  4. Builds a calphy.input.Calculation, runs it, packs a FreeEnergyOutput.
  5. Restores the previous cwd in a try/finally.

All calphy and pyiron_workflow_lammps imports happen inside node bodies
so importing this subpackage works without the [free-energy] extra.
"""

from __future__ import annotations

import os
from typing import Literal

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
    _build_calphy_calculation,
    _pack_free_energy_output,
    _run_calphy_job,
    _validate_engine_only_command,
    _validate_structure,
)
from pyiron_workflow_atomistics.physics.free_energy._compat import (
    _require_calphy,
    _require_lammps_engine,
)
from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput


def _run_one(
    *,
    mode: str,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str,
    subdir: str,
    reference_phase: str,
    temperature: float,
    pressure: float,
    builder_kwargs: dict,
) -> FreeEnergyOutput:
    """Shared body for every public node: validate → chdir → run → pack."""
    _require_calphy()
    _require_lammps_engine()
    _validate_engine_only_command(lammps_engine)
    _validate_structure(structure)

    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(simfolder)
        calc = _build_calphy_calculation(
            mode=mode,
            structure=structure,
            potential=potential,
            lammps_engine=lammps_engine,
            working_directory=simfolder,
            **builder_kwargs,
        )
        job, report = _run_calphy_job(calc)
        return _pack_free_energy_output(
            mode=mode,
            job=job,
            report=report,
            simfolder=simfolder,
            structure=structure,
            reference_phase=reference_phase,
            temperature=temperature,
            pressure=pressure,
        )
    finally:
        os.chdir(prev_cwd)


@pwf.as_function_node("free_energy_output")
def free_energy(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "free_energy",
    temperature: float,
    pressure: float = 0.0,
    reference_phase: Literal["solid", "liquid"],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Helmholtz/Gibbs free energy at one (T, P) via Frenkel-Ladd / UF reference.

    Pressure is in **bar** (calphy native). Temperature in K. Free energy
    returned in eV/atom.
    """
    return _run_one(
        mode="fe",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase=reference_phase,
        temperature=temperature,
        pressure=pressure,
        builder_kwargs=dict(
            temperature=temperature,
            pressure=pressure,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k free_energy_node
```

Expected: all four tests pass (or skip when calphy/pyiron_workflow_lammps are missing).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/calphy.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): public free_energy function-node (mode='fe')

Shared _run_one helper centralises validation → chdir → build → run
→ pack so the other five public nodes (added in Tasks 13-17) are
one-line wrappers around it. Tier 1 tests cover missing-dep,
non-default engine fields, non-periodic structure, and cwd restore
on mid-run exception.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Tier 2 smoke test for `free_energy` (FCC Cu EAM)

**Files:**
- Create: `tests/resources/free_energy/Cu01.eam.alloy` (vendored from calphy/examples/potentials)
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Vendor the test potential**

```bash
mkdir -p /home/liger/pyiron_workflow_atomistics/tests/resources/free_energy
cp /home/liger/calphy/examples/potentials/Cu01.eam.alloy \
   /home/liger/pyiron_workflow_atomistics/tests/resources/free_energy/Cu01.eam.alloy
```

This is ~760 KB and license-compatible (potential ships under the same ASL as calphy itself); it lives only in the test tree.

- [ ] **Step 2: Write the failing Tier-2 test**

Append to `tests/unit/physics/test_free_energy.py`:

```python
# ---------------------------------------------------------------------------
# Tier 2 integration — needs calphy + lmp binary
# ---------------------------------------------------------------------------

import shutil
from pathlib import Path

LAMMPS_BIN = shutil.which("lmp") or shutil.which("lmp_mpi")
RESOURCES = Path(__file__).resolve().parent.parent.parent / "resources" / "free_energy"

requires_lammps = pytest.mark.skipif(
    LAMMPS_BIN is None or not (RESOURCES / "Cu01.eam.alloy").exists(),
    reason="needs lmp binary on PATH and tests/resources/free_energy/Cu01.eam.alloy",
)
requires_calphy = pytest.mark.skipif(
    pytest.importorskip.__module__ is None,
    reason="calphy not installed",
)


@requires_lammps
def test_free_energy_fcc_cu_smoke(tmp_path):
    pytest.importorskip("calphy")
    from ase.build import bulk
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import free_energy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    cu = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff=f"* * {RESOURCES / 'Cu01.eam.alloy'} Cu",
    )
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command=LAMMPS_BIN)

    out = free_energy.node_function(
        structure=cu,
        lammps_engine=eng,
        potential=pot,
        working_directory=str(tmp_path),
        subdir="fe",
        temperature=100.0,
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
    )
    assert out.mode == "fe"
    assert out.reference_phase == "solid"
    assert out.n_atoms == 108
    assert out.elements == ["Cu"]
    assert -4.5 < out.free_energy < -3.0  # Cu EAM, eV/atom @ 100 K
    assert out.free_energy_error >= 0
    assert os.path.isabs(out.simfolder)
    assert (tmp_path / "fe" / "report.yaml").exists()
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/unit/physics/test_free_energy.py::test_free_energy_fcc_cu_smoke -v
```

Expected: PASS if a `lmp` binary is on `PATH`; SKIPPED otherwise. Budget ~30 s.

- [ ] **Step 4: Commit**

```bash
git add tests/resources/free_energy/Cu01.eam.alloy tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
test(free-energy): Tier 2 smoke test for free_energy fcc-Cu EAM

End-to-end run of the free_energy function-node against a real
LAMMPS binary + Cu01.eam.alloy. Asserts the result shape, n_atoms
captured, simfolder absolute, and free energy in the expected range
for EAM Cu at 100 K. Skipped when lmp is not on PATH.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Public node `reversible_scaling_temperature`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/calphy.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# reversible_scaling_temperature
# ---------------------------------------------------------------------------


def test_reversible_scaling_temperature_validates_tuple_shape(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        reversible_scaling_temperature,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"temperature_range"):
        reversible_scaling_temperature.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature_range=300.0,  # scalar — must be 2-tuple
            reference_phase="solid",
        )


@requires_lammps
def test_reversible_scaling_temperature_returns_curve(tmp_path):
    pytest.importorskip("calphy")
    from ase.build import bulk
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        reversible_scaling_temperature,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    cu = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff=f"* * {RESOURCES / 'Cu01.eam.alloy'} Cu",
    )
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command=LAMMPS_BIN)

    out = reversible_scaling_temperature.node_function(
        structure=cu,
        lammps_engine=eng,
        potential=pot,
        working_directory=str(tmp_path),
        subdir="ts",
        temperature_range=(100.0, 300.0),
        pressure=0.0,
        reference_phase="solid",
        n_equilibration_steps=2000,
        n_switching_steps=2000,
    )
    assert out.mode == "ts"
    assert out.temperature_array is not None
    assert out.free_energy_array is not None
    assert out.temperature_array.shape == out.free_energy_array.shape
    import numpy as np
    assert np.all(np.diff(out.temperature_array) >= 0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k reversible_scaling_temperature
```

Expected: `ImportError` — node not defined.

- [ ] **Step 3: Implement `reversible_scaling_temperature`**

Append to `pyiron_workflow_atomistics/physics/free_energy/calphy.py`:

```python
@pwf.as_function_node("free_energy_output")
def reversible_scaling_temperature(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "reversible_scaling_temperature",
    temperature_range: tuple[float, float],
    pressure: float = 0.0,
    reference_phase: Literal["solid", "liquid"],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Free energy along an isobar by reversible scaling in temperature.

    ``temperature_range`` is (lo, hi) in K. Pressure is in bar. The
    ``FreeEnergyOutput.temperature_array`` and ``free_energy_array``
    fields are populated with the integrated curve.
    """
    if (temperature_range is None
            or not hasattr(temperature_range, "__len__")
            or len(temperature_range) != 2):
        raise ValueError(
            "reversible_scaling_temperature requires "
            "`temperature_range=(lo, hi)` (length-2 tuple)"
        )
    return _run_one(
        mode="ts",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase=reference_phase,
        temperature=float(temperature_range[0]),
        pressure=pressure,
        builder_kwargs=dict(
            temperature_range=temperature_range,
            pressure=pressure,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k reversible_scaling_temperature
```

Expected: validation test passes; Tier-2 smoke passes or skips.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/calphy.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): reversible_scaling_temperature public node

Wraps calphy mode='ts'. Tuple-shape validation is done up-front so
the user sees a clear ValueError rather than a calphy Pydantic
error. Returns FreeEnergyOutput with temperature_array and
free_energy_array populated from temperature_sweep.dat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Public node `reversible_scaling_pressure`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/calphy.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# reversible_scaling_pressure
# ---------------------------------------------------------------------------


def test_reversible_scaling_pressure_validates_tuple_shape(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        reversible_scaling_pressure,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"pressure_range"):
        reversible_scaling_pressure.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            pressure_range=1000.0,  # scalar — must be 2-tuple
            reference_phase="solid",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k reversible_scaling_pressure
```

Expected: `ImportError` — node not defined.

- [ ] **Step 3: Implement `reversible_scaling_pressure`**

Append to `calphy.py`:

```python
@pwf.as_function_node("free_energy_output")
def reversible_scaling_pressure(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "reversible_scaling_pressure",
    temperature: float,
    pressure_range: tuple[float, float],
    reference_phase: Literal["solid", "liquid"],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Free energy along an isotherm by reversible scaling in pressure.

    ``pressure_range`` is (lo, hi) in bar (calphy native).
    """
    if (pressure_range is None
            or not hasattr(pressure_range, "__len__")
            or len(pressure_range) != 2):
        raise ValueError(
            "reversible_scaling_pressure requires "
            "`pressure_range=(lo, hi)` (length-2 tuple)"
        )
    return _run_one(
        mode="pscale",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase=reference_phase,
        temperature=temperature,
        pressure=float(pressure_range[0]),
        builder_kwargs=dict(
            temperature=temperature,
            pressure_range=pressure_range,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k reversible_scaling_pressure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/calphy.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): reversible_scaling_pressure public node

Wraps calphy mode='pscale'. Tuple-shape validation up-front;
pressure_range in bar, calphy native.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Public node `melting_temperature`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/calphy.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# melting_temperature
# ---------------------------------------------------------------------------


def test_melting_temperature_validates_positive_guess(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import melting_temperature
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/Al.eam.alloy Al")
    with pytest.raises(ValueError, match=r"positive"):
        melting_temperature.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature_guess=-100.0,
        )


@requires_lammps
@pytest.mark.slow
def test_melting_temperature_runs(tmp_path):
    pytest.importorskip("calphy")
    from ase.build import bulk
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import melting_temperature
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    cu = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    pot = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff=f"* * {RESOURCES / 'Cu01.eam.alloy'} Cu",
    )
    eng = LammpsEngine(EngineInput=CalcInputStatic(), command=LAMMPS_BIN)
    out = melting_temperature.node_function(
        structure=cu,
        lammps_engine=eng,
        potential=pot,
        working_directory=str(tmp_path),
        temperature_guess=1300.0,
        step=400,
        max_attempts=3,
        n_equilibration_steps=2000,
        n_switching_steps=2000,
    )
    assert out.mode == "melting_temperature"
    assert out.reference_phase == "both"
    assert 800.0 < out.melting_temperature < 2000.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k melting_temperature
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `melting_temperature`**

Append to `calphy.py`:

```python
@pwf.as_function_node("free_energy_output")
def melting_temperature(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "melting_temperature",
    temperature_guess: float | None = None,
    pressure: float = 0.0,
    step: int = 200,
    max_attempts: int = 5,
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Automated solid+liquid free-energy crossover via calphy's MeltingTemp.

    ``temperature_guess`` is calphy's starting temperature in K (if
    ``None``, calphy guesses from ``mendeleev``). ``step`` (K) and
    ``max_attempts`` route to ``Calculation.melting_temperature``.
    Result has ``reference_phase="both"`` and populates
    ``melting_temperature`` + ``melting_temperature_error``.
    """
    if temperature_guess is not None and temperature_guess <= 0:
        raise ValueError(
            f"`temperature_guess` must be positive, got {temperature_guess}"
        )
    return _run_one(
        mode="melting_temperature",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase="both",
        temperature=temperature_guess or 0.0,
        pressure=pressure,
        builder_kwargs=dict(
            temperature_guess=temperature_guess,
            pressure=pressure,
            melting_step=step,
            melting_max_attempts=max_attempts,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k melting_temperature
```

Expected: validation test passes; slow Tier-2 test passes (~minutes) or skips.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/calphy.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): melting_temperature public node

Wraps calphy's MeltingTemp routine. temperature_guess routes to
Calculation.temperature; step / max_attempts route to
Calculation.melting_temperature.step / .attempts. Result has
reference_phase='both' and populates melting_temperature and
melting_temperature_error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Public node `alchemy`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/calphy.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# alchemy
# ---------------------------------------------------------------------------


def test_alchemy_requires_target_potential_strings(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import alchemy
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/A.eam.alloy Al")
    with pytest.raises(ValueError, match=r"pair_style_target"):
        alchemy.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            # missing pair_style_target / pair_coeff_target
            pair_style_target=None,
            pair_coeff_target=None,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k alchemy
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `alchemy`**

Append to `calphy.py`:

```python
@pwf.as_function_node("free_energy_output")
def alchemy(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "alchemy",
    temperature: float,
    pressure: float = 0.0,
    pair_style_target: str,
    pair_coeff_target: str,
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Alchemical free-energy difference between two potentials.

    The starting potential is the supplied ``potential``; the target
    potential is supplied as raw ``pair_style_target`` /
    ``pair_coeff_target`` strings. v1 takes two separate strings rather
    than two LammpsPotential dataclasses — modelling a "two-engines"
    macro felt like overkill given alchemy is rarely used. If the use
    case grows, refactor in a follow-up.
    """
    if not pair_style_target or not pair_coeff_target:
        raise ValueError(
            "alchemy requires both `pair_style_target` and "
            "`pair_coeff_target` (raw LAMMPS strings for the target "
            "potential)"
        )
    return _run_one(
        mode="alchemy",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase="solid",
        temperature=temperature,
        pressure=pressure,
        builder_kwargs=dict(
            temperature=temperature,
            pressure=pressure,
            pair_style_target=pair_style_target,
            pair_coeff_target=pair_coeff_target,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k alchemy
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/calphy.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): alchemy public node

Wraps calphy mode='alchemy'. Source potential is the supplied
LammpsPotential; target is supplied as raw pair_style_target /
pair_coeff_target strings — modelling a two-engines macro felt like
overkill for a rarely-used mode.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Public node `composition_scaling`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/calphy.py`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# composition_scaling
# ---------------------------------------------------------------------------


def test_composition_scaling_requires_output_composition(fcc_al_atoms):
    pytest.importorskip("calphy")
    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.calphy import (
        composition_scaling,
    )
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
    from pyiron_workflow_lammps.engine import LammpsEngine

    eng = LammpsEngine(EngineInput=CalcInputStatic(), command="lmp")
    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /tmp/AB.eam.alloy A B")
    with pytest.raises(ValueError, match=r"output_chemical_composition"):
        composition_scaling.node_function(
            structure=fcc_al_atoms,
            lammps_engine=eng,
            potential=pot,
            temperature=300.0,
            output_chemical_composition=None,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k composition_scaling
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `composition_scaling`**

Append to `calphy.py`:

```python
@pwf.as_function_node("free_energy_output")
def composition_scaling(
    *,
    structure: Atoms,
    lammps_engine,
    potential: LammpsPotential,
    working_directory: str = ".",
    subdir: str = "composition_scaling",
    temperature: float,
    pressure: float = 0.0,
    output_chemical_composition: dict[str, int],
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
) -> FreeEnergyOutput:
    """Free-energy integration over composition between two stoichiometries.

    The starting composition is read off ``structure``; the target
    composition is given as ``output_chemical_composition`` (a dict of
    element-symbol → atom-count). calphy interpolates by alchemically
    converting atoms.
    """
    if not output_chemical_composition:
        raise ValueError(
            "composition_scaling requires "
            "`output_chemical_composition={'A': n_a, 'B': n_b, ...}` "
            "(target atom counts per element)"
        )
    return _run_one(
        mode="composition_scaling",
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        working_directory=working_directory,
        subdir=subdir,
        reference_phase="solid",
        temperature=temperature,
        pressure=pressure,
        builder_kwargs=dict(
            temperature=temperature,
            pressure=pressure,
            output_chemical_composition=output_chemical_composition,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            npt=npt,
            equilibration_control=equilibration_control,
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k composition_scaling
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/calphy.py tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): composition_scaling public node

Wraps calphy mode='composition_scaling'. Target composition is a
dict of element → atom-count; FreeEnergyOutput.composition_path is
populated from calphy's input record so callers can post-process.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Public re-exports + import smoke + CHANGELOG finalize

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/__init__.py`
- Modify: `CHANGELOG.md`
- Modify: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


def test_subpackage_public_api_re_exports():
    import pyiron_workflow_atomistics.physics.free_energy as fe

    public = {
        "FreeEnergyOutput",
        "LammpsPotential",
        "free_energy",
        "reversible_scaling_temperature",
        "reversible_scaling_pressure",
        "melting_temperature",
        "alchemy",
        "composition_scaling",
    }
    missing = public - set(dir(fe))
    assert not missing, f"missing public exports: {missing}"


def test_subpackage_imports_without_calphy(monkeypatch):
    """Importing the subpackage must NOT trigger a calphy import."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "calphy", None)
    # Drop any cached module so import is fresh
    sys.modules.pop("pyiron_workflow_atomistics.physics.free_energy", None)
    sys.modules.pop(
        "pyiron_workflow_atomistics.physics.free_energy._calphy_adapter", None,
    )
    sys.modules.pop(
        "pyiron_workflow_atomistics.physics.free_energy.calphy", None,
    )
    importlib.import_module(
        "pyiron_workflow_atomistics.physics.free_energy"
    )  # should not raise
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k "public_api_re_exports or imports_without_calphy"
```

Expected: `AssertionError: missing public exports: ...`.

- [ ] **Step 3: Update `__init__.py`**

Replace the contents of
`pyiron_workflow_atomistics/physics/free_energy/__init__.py` with:

```python
"""calphy-backed free-energy workflows.

Public API
----------

Dataclasses:
    LammpsPotential  - pair_style + pair_coeff + optional potential_file
    FreeEnergyOutput - typed result of every node

Function-nodes (one per calphy mode):
    free_energy                       - mode='fe'
    reversible_scaling_temperature    - mode='ts'
    reversible_scaling_pressure       - mode='pscale'
    melting_temperature               - mode='melting_temperature'
    alchemy                           - mode='alchemy'
    composition_scaling               - mode='composition_scaling'

All node-and-adapter imports defer ``calphy`` and ``pyiron_workflow_lammps``
imports to node-body call time, so importing this subpackage does not
require the ``[free-energy]`` extra.
"""

from pyiron_workflow_atomistics.physics.free_energy.calphy import (
    alchemy,
    composition_scaling,
    free_energy,
    melting_temperature,
    reversible_scaling_pressure,
    reversible_scaling_temperature,
)
from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

__all__ = [
    "FreeEnergyOutput",
    "LammpsPotential",
    "alchemy",
    "composition_scaling",
    "free_energy",
    "melting_temperature",
    "reversible_scaling_pressure",
    "reversible_scaling_temperature",
]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/physics/test_free_energy.py -v -k "public_api_re_exports or imports_without_calphy"
```

Expected: both pass.

- [ ] **Step 5: Finalize CHANGELOG**

Rename `## [Unreleased]` to `## [0.0.9] — YYYY-MM-DD` (substituting today's date) and keep the bullets from Task 1.

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/unit/physics/test_free_energy.py -v
```

Expected: every Tier-1 test passes; Tier-2 tests pass or skip with a clear reason.

- [ ] **Step 7: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/__init__.py CHANGELOG.md tests/unit/physics/test_free_energy.py
git commit -m "$(cat <<'EOF'
feat(free-energy): public re-exports + CHANGELOG 0.0.9

Subpackage __init__.py re-exports the six function-nodes plus
LammpsPotential and FreeEnergyOutput. Lazy imports preserved: the
subpackage imports without calphy installed (Tier 1 tests prove it).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final acceptance checklist

Before marking the implementation complete, run through this list:

- [ ] `pytest tests/unit/physics/test_free_energy.py -v` — every Tier-1 test passes; Tier-2 tests pass or skip with a clear reason.
- [ ] `python -c "import pyiron_workflow_atomistics.physics.free_energy as fe; print(fe.__all__)"` — prints the eight public names from Task 18.
- [ ] `python -c "import sys; sys.modules['calphy']=None; import pyiron_workflow_atomistics.physics.free_energy; print('ok')"` — prints `ok` (the subpackage imports even without calphy).
- [ ] `pip install -e ".[test,free-energy]"` succeeds; `python -c "import calphy, pyiron_workflow_lammps; print('ok')"` prints `ok`.
- [ ] Tier 2: `pytest tests/unit/physics/test_free_energy.py::test_free_energy_fcc_cu_smoke -v` passes locally with a `lmp` binary on `PATH`; budget ~30 s.
- [ ] Tier 2 slow: `pytest tests/unit/physics/test_free_energy.py::test_melting_temperature_runs -v -m slow` passes locally; budget a few minutes.
- [ ] Hand-test that a user with a real EAM potential can run `free_energy.node_function(...)` end-to-end and get a sensible `FreeEnergyOutput`.
- [ ] `git log --oneline | head -20` shows one commit per task with the prefixed format (`feat(free-energy):`, `test(free-energy):`, `build:`, `docs(free-energy):`).
- [ ] Spec follow-ups section unchanged — no scope creep beyond v1.

---

## Spec coverage map

| Spec section | Implementation task(s) |
|---|---|
| File structure (`physics/free_energy/{...}`) | Task 2 creates the package; Tasks 3-11 fill it |
| `LammpsPotential` dataclass | Task 3 |
| `FreeEnergyOutput` dataclass | Task 4 |
| `_validate_engine_only_command` | Task 6 |
| `_split_lammps_command` | Task 5 |
| `_build_calphy_calculation` (all six modes) | Task 8 (fe) + Tasks 13, 14, 15, 16, 17 (one mode-branch each, all share the dispatch table) |
| `_run_calphy_job` | Task 9 |
| `_load_rs_curve` | Task 10 |
| `_pack_free_energy_output` | Task 9 |
| `script_mode=True` hardcoded | Task 8 (in `_build_calphy_calculation`) |
| Public re-exports (`__init__.py`) | Task 18 |
| Failure modes table | Tier-1 tests in Tasks 2, 6, 7, 11, 13, 14, 15, 16, 17 |
| Tier 1 unit tests | Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18 |
| Tier 2 integration tests | Tasks 12, 13, 15 |
| Tier 3 determinism / regression | Task 12 (simfolder absolute), Task 11 (cwd restore on exception), Task 8 (data file round-trip) |
| `[free-energy]` optional extra | Task 1 |
| CHANGELOG `[Unreleased]` → `0.0.9` | Tasks 1 and 18 |
| Follow-ups section (incl. phonopy/dynaphopy TODO) | Already in the spec; no code change |
