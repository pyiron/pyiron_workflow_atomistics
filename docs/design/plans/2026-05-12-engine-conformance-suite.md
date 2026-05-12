# Engine Conformance Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pyiron_workflow_atomistics.testing.EngineConformanceTests` — a reusable pytest mixin that downstream engine packages (`pyiron_workflow_lammps`, `pyiron_workflow_vasp`) subclass to verify their `Engine` Protocol implementation against a single source-of-truth bar.

**Architecture:** Purely additive change to atomistics. New sub-package `pyiron_workflow_atomistics/testing/` holding the mixin; one in-tree subclass exercising `ASEEngine` proves the mixin is correct; top-level `__init__.py` exposes the sub-package; bump version + CHANGELOG.md + tag-and-release to PyPI.

**Tech Stack:** Python 3.11 (Python ≥3.9 supported), `typing.Protocol`, `dataclasses`, pytest. Existing repo conventions: ruff/black, versioneer, conda-forge env via `.ci_support/`, shared pyiron CI workflows.

**Spec:** `docs/design/specs/2026-05-12-engine-conformance-suite-design.md`.

**Branch:** `design-engine-conformance-suite` (already pushed to `origin`).

**Working directory:** `/home/liger/pyiron_workflow_atomistics`.

**Python interpreter / pytest binary:** `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python` and `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest`.

---

## File structure (what this PR touches)

| Path | Action | Responsibility |
|---|---|---|
| `pyiron_workflow_atomistics/testing/__init__.py` | NEW | Re-export `EngineConformanceTests` so consumers `from pyiron_workflow_atomistics.testing import EngineConformanceTests`. |
| `pyiron_workflow_atomistics/testing/engine_conformance.py` | NEW | The mixin class with 5 `test_*` methods exercising every Protocol contract clause. |
| `pyiron_workflow_atomistics/__init__.py` | MODIFY | One added line: `from . import testing` so the sub-package is reachable via attribute access on the top-level module. |
| `tests/unit/engine/test_ase_conformance.py` | NEW | In-tree `EngineConformanceTests` subclass with an `ASEEngine(EMT)` factory — proves the mixin is correct. |
| `CHANGELOG.md` | NEW | Top-of-file `0.0.5` section describing the conformance suite addition + the downstream-engine-migration cycle it enables. |
| `docs/design/plans/2026-05-12-engine-conformance-suite.md` | NEW | This plan file. Committed first. |

No file in `pyiron_workflow_atomistics/engine/` is modified — the Protocol contract is already final from the cleanup PR.

---

## Task 1: Commit this plan first

**Files:**
- Create: `docs/design/plans/2026-05-12-engine-conformance-suite.md` (this file)

- [ ] **Step 1: Verify file is staged**

```bash
cd /home/liger/pyiron_workflow_atomistics
git status --short docs/design/plans/2026-05-12-engine-conformance-suite.md
```

Expected: `?? docs/design/plans/2026-05-12-engine-conformance-suite.md` (untracked, ready to add).

- [ ] **Step 2: Commit**

```bash
git add docs/design/plans/2026-05-12-engine-conformance-suite.md
git commit -m "docs(plan): conformance suite implementation plan

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Push**

```bash
git push origin design-engine-conformance-suite
```

Expected: branch tip on remote moves to the new commit.

---

## Task 2: Scaffold the `testing/` sub-package (empty)

**Files:**
- Create: `pyiron_workflow_atomistics/testing/__init__.py`
- Create: `pyiron_workflow_atomistics/testing/engine_conformance.py`
- Modify: `pyiron_workflow_atomistics/__init__.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/unit/test_testing_namespace.py`:

```python
"""Smoke test: pyiron_workflow_atomistics.testing namespace exists."""


def test_testing_subpackage_importable():
    import pyiron_workflow_atomistics
    import pyiron_workflow_atomistics.testing

    # Public access must work via attribute (top-level __init__.py registers it)
    assert hasattr(pyiron_workflow_atomistics, "testing")


def test_engine_conformance_tests_importable():
    from pyiron_workflow_atomistics.testing import EngineConformanceTests

    assert EngineConformanceTests is not None
```

- [ ] **Step 2: Run the smoke test to verify it fails**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_testing_namespace.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.testing'`.

- [ ] **Step 3: Create the package directory with empty `__init__.py`**

Create `pyiron_workflow_atomistics/testing/__init__.py`:

```python
"""Public test helpers for downstream Engine implementations.

The :class:`EngineConformanceTests` mixin is the single source of truth
for the :class:`pyiron_workflow_atomistics.engine.Engine` Protocol
contract. Downstream packages subclass it with their own engine factory
and run pytest; every contract clause is exercised by a named method.
"""

from .engine_conformance import EngineConformanceTests

__all__ = ["EngineConformanceTests"]
```

- [ ] **Step 4: Create the (currently empty) conformance module**

Create `pyiron_workflow_atomistics/testing/engine_conformance.py`:

```python
"""Pytest mixin verifying Engine Protocol conformance.

Subclass with::

    class TestMyEngineConformance(EngineConformanceTests):
        @staticmethod
        def engine_factory(tmp_path):
            return MyEngine(EngineInput=CalcInputStatic(),
                            working_directory=str(tmp_path))

The class needs no `__init__`; pytest discovers methods directly.
"""

from __future__ import annotations


class EngineConformanceTests:
    """Subclass and override ``engine_factory`` (required).

    Optionally override ``test_structure_factory`` to swap the default
    4-atom Cu FCC bulk used by the run() smoke test.
    """
```

- [ ] **Step 5: Register the sub-package on the top-level module**

Edit `pyiron_workflow_atomistics/__init__.py` — append after the existing `__all__` assignment:

```python
from . import testing  # noqa: F401  -- exposes pyiron_workflow_atomistics.testing
```

- [ ] **Step 6: Re-run the smoke test to verify it passes**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_testing_namespace.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Run the existing engine tests to confirm no regression**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine -q --no-header
```

Expected: 15 passed (matching the existing baseline from the cleanup PR).

- [ ] **Step 8: Commit**

```bash
git add pyiron_workflow_atomistics/testing/ pyiron_workflow_atomistics/__init__.py tests/unit/test_testing_namespace.py
git commit -m "feat(testing): scaffold testing/ sub-package with empty conformance mixin

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `test_satisfies_engine_protocol` + in-tree ASEEngine subclass

**Files:**
- Modify: `pyiron_workflow_atomistics/testing/engine_conformance.py`
- Create: `tests/unit/engine/test_ase_conformance.py`

- [ ] **Step 1: Write the failing test**

Append to `pyiron_workflow_atomistics/testing/engine_conformance.py`:

```python
from dataclasses import is_dataclass

from pyiron_workflow_atomistics.engine import Engine


class EngineConformanceTests:
    """Subclass and override ``engine_factory`` (required).

    Optionally override ``test_structure_factory`` to swap the default
    4-atom Cu FCC bulk used by the run() smoke test.
    """

    def test_satisfies_engine_protocol(self, tmp_path):
        """Engine instances must satisfy the runtime_checkable Protocol
        and be @dataclass-decorated so dataclasses.replace works in
        with_working_directory."""
        eng = type(self).engine_factory(tmp_path)
        assert isinstance(eng, Engine), (
            f"{type(eng).__name__} does not satisfy the Engine Protocol "
            "(missing working_directory, get_calculate_fn, or "
            "with_working_directory)"
        )
        assert is_dataclass(eng), (
            f"{type(eng).__name__} must be a @dataclass for the "
            "dataclasses.replace()-based with_working_directory pattern"
        )
```

Replace the prior empty class body with the implementation above.

- [ ] **Step 2: Create the in-tree ASEEngine subclass**

Create `tests/unit/engine/test_ase_conformance.py`:

```python
"""In-tree conformance test: prove EngineConformanceTests is correct
by running it against the canonical ASEEngine (EMT calculator)."""

from __future__ import annotations

from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.testing import EngineConformanceTests


class TestASEEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        return ASEEngine(
            EngineInput=CalcInputStatic(),
            calculator=EMT(),
            working_directory=str(tmp_path),
        )
```

- [ ] **Step 3: Run the new test to verify it passes for ASEEngine**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_ase_conformance.py -v
```

Expected: 1 passed (the `test_satisfies_engine_protocol` method picked up via inheritance).

- [ ] **Step 4: Sanity-check it fails for a non-engine**

This is the negative case — verifies the mixin actually catches violations. Add a temporary local test (do NOT commit):

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from pyiron_workflow_atomistics.testing import EngineConformanceTests
class Bad:
    working_directory = None  # no get_calculate_fn, no with_working_directory

class FakeTest(EngineConformanceTests):
    @staticmethod
    def engine_factory(_): return Bad()

try:
    FakeTest().test_satisfies_engine_protocol('/tmp')
except AssertionError as e:
    print('OK — mixin rejected non-engine:', str(e)[:80])
"
```

Expected: prints `OK — mixin rejected non-engine: <ClassName 'Bad' does not satisfy …>`.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/testing/engine_conformance.py tests/unit/engine/test_ase_conformance.py
git commit -m "feat(testing): add Protocol-satisfaction conformance check + ASE in-tree subclass

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add `test_with_working_directory_is_pure`

**Files:**
- Modify: `pyiron_workflow_atomistics/testing/engine_conformance.py`

- [ ] **Step 1: Add the method**

Append a new method to `EngineConformanceTests` in `pyiron_workflow_atomistics/testing/engine_conformance.py`:

```python
import os


class EngineConformanceTests:
    ...

    def test_with_working_directory_is_pure(self, tmp_path):
        """Engine.with_working_directory(subdir) must return a NEW instance
        whose working_directory is os.path.join(self.working_directory, subdir),
        without mutating the parent."""
        eng = type(self).engine_factory(tmp_path)
        parent_wd = eng.working_directory

        sub = eng.with_working_directory("subdir_a")

        # Parent unchanged
        assert eng.working_directory == parent_wd, (
            "with_working_directory mutated the parent engine's working_directory"
        )
        # Child path composed correctly
        assert sub.working_directory == os.path.join(parent_wd, "subdir_a"), (
            f"with_working_directory composed an unexpected path: "
            f"{sub.working_directory!r} != {os.path.join(parent_wd, 'subdir_a')!r}"
        )
        # New instance — not the same object
        assert sub is not eng, (
            "with_working_directory returned self instead of a copy"
        )
        # Same dataclass type
        assert type(sub) is type(eng)
```

Make sure `import os` is at the top of the file (one shared import block).

- [ ] **Step 2: Run the ASE conformance test**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_ase_conformance.py -v
```

Expected: 2 passed (now includes `test_with_working_directory_is_pure`).

- [ ] **Step 3: Commit**

```bash
git add pyiron_workflow_atomistics/testing/engine_conformance.py
git commit -m "feat(testing): conformance test for with_working_directory purity

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `test_pickleable`

**Files:**
- Modify: `pyiron_workflow_atomistics/testing/engine_conformance.py`

- [ ] **Step 1: Add the method**

Append to `EngineConformanceTests`:

```python
import pickle


class EngineConformanceTests:
    ...

    def test_pickleable(self, tmp_path):
        """Engines must survive pickle round-trip — workflows are
        checkpointed to disk and may be resubmitted to SLURM."""
        eng = type(self).engine_factory(tmp_path)
        roundtrip = pickle.loads(pickle.dumps(eng))
        assert roundtrip.working_directory == eng.working_directory, (
            "Pickle round-trip lost or corrupted working_directory"
        )
        assert type(roundtrip) is type(eng)
```

Add `import pickle` to the top-of-file import block.

- [ ] **Step 2: Run**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_ase_conformance.py -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add pyiron_workflow_atomistics/testing/engine_conformance.py
git commit -m "feat(testing): conformance test for pickle round-trip

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Add `test_get_calculate_fn_signature`

**Files:**
- Modify: `pyiron_workflow_atomistics/testing/engine_conformance.py`

- [ ] **Step 1: Add the default structure factory + method**

First, add the structure-factory class attribute + helper to `EngineConformanceTests`:

```python
from typing import Callable

from ase import Atoms
from ase.build import bulk


class EngineConformanceTests:
    """Subclass and override ``engine_factory`` (required).

    Optionally override ``test_structure_factory`` (a callable returning
    an :class:`ase.Atoms` instance) to swap the default 4-atom Cu FCC
    bulk used by the run() smoke test.
    """

    # Optional override
    test_structure_factory: Callable | None = None

    def _structure(self) -> Atoms:
        factory = type(self).test_structure_factory
        if factory is None:
            return bulk("Cu", "fcc", a=3.6, cubic=True)
        return factory()
    ...
```

Then add the contract test:

```python
class EngineConformanceTests:
    ...

    def test_get_calculate_fn_signature(self, tmp_path):
        """get_calculate_fn(structure) must return (callable, kwargs)
        with `structure` NOT in the returned kwargs dict — the caller
        invokes the callable as fn(structure=structure, **kwargs)."""
        eng = type(self).engine_factory(tmp_path)
        result = eng.get_calculate_fn(self._structure())

        assert isinstance(result, tuple) and len(result) == 2, (
            "get_calculate_fn must return a (callable, kwargs) tuple"
        )
        fn, kwargs = result
        assert callable(fn), "First element of get_calculate_fn return must be callable"
        assert isinstance(kwargs, dict), (
            "Second element of get_calculate_fn return must be a dict"
        )
        assert "structure" not in kwargs, (
            "structure must NOT appear in the kwargs dict — the caller "
            "supplies it positionally as fn(structure=..., **kwargs)"
        )
```

- [ ] **Step 2: Run**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_ase_conformance.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add pyiron_workflow_atomistics/testing/engine_conformance.py
git commit -m "feat(testing): conformance test for get_calculate_fn signature

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add `test_run_returns_engine_output` (the smoke test)

**Files:**
- Modify: `pyiron_workflow_atomistics/testing/engine_conformance.py`

- [ ] **Step 1: Add the method**

Append to `EngineConformanceTests`:

```python
from pyiron_workflow_atomistics.engine import EngineOutput, run


class EngineConformanceTests:
    ...

    def test_run_returns_engine_output(self, tmp_path):
        """Single-point smoke test — verifies the calc callable returns
        a real EngineOutput dataclass with the three required fields
        populated. Does NOT assert any physics, just the shape."""
        eng = type(self).engine_factory(tmp_path)
        out = run.node_function(structure=self._structure(), engine=eng)

        assert isinstance(out, EngineOutput), (
            f"run() must return EngineOutput, got {type(out).__name__}"
        )
        assert out.final_structure is not None, "EngineOutput.final_structure must be populated"
        assert out.final_energy is not None, "EngineOutput.final_energy must be populated"
        assert isinstance(out.converged, bool), (
            f"EngineOutput.converged must be bool, got {type(out.converged).__name__}"
        )
```

Important: this method calls `run.node_function(...)` (the `pwf.as_function_node`-decorated function's underlying callable). The mixin must NOT trigger a pyiron_workflow graph — it's a unit test of the engine, not the workflow runner.

- [ ] **Step 2: Run the full suite against ASEEngine**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine/test_ase_conformance.py -v
```

Expected: 5 passed. The smoke run uses EMT on Cu4 — sub-second on any CPU.

- [ ] **Step 3: Run the entire engine test directory to confirm no regression**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/engine -q --no-header
```

Expected: 20 passed (15 baseline + 5 conformance). Anything else means investigate.

- [ ] **Step 4: Commit**

```bash
git add pyiron_workflow_atomistics/testing/engine_conformance.py
git commit -m "feat(testing): conformance test for run() returning EngineOutput

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Write CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create the file**

Create `CHANGELOG.md` at the repo root:

```markdown
# Changelog

All notable changes to `pyiron_workflow_atomistics` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package follows [PEP 440](https://peps.python.org/pep-0440/) versioning
via `versioneer`.

## [0.0.5] — 2026-05-12

### Added

- **`pyiron_workflow_atomistics.testing.EngineConformanceTests`** — a reusable
  pytest mixin that verifies an `Engine` Protocol implementation against
  every contract clause: Protocol satisfaction (`isinstance`),
  `@dataclass`-ness, `with_working_directory` purity, pickle round-trip,
  `get_calculate_fn` signature, and a single-point `run()` smoke. Downstream
  engine packages subclass it with their own `engine_factory(tmp_path)`
  staticmethod and run pytest.
- In-tree `tests/unit/engine/test_ase_conformance.py` subclasses the mixin
  with an `ASEEngine(EMT)` factory, proving the suite is correct against
  the canonical engine.

### Enables

- `pyiron_workflow_lammps` migration onto the new Protocol-based Engine
  contract (see its `docs/design/specs/2026-05-12-engine-protocol-migration-design.md`).
- `pyiron_workflow_vasp` greenfield engine implementation (see its
  `docs/design/specs/2026-05-12-vasp-engine-design.md`).

### Unchanged

- All existing public symbols. The Engine Protocol shape, `EngineOutput`
  dataclass, `run` / `subengine` / `subdir_path` nodes, and every
  `engine` / `structure` / `physics` / `analysis` API is identical to 0.0.4.

## [0.0.4] — pre-2026-05-12

See git history for the cleanup-and-reorganise PR (#30, #31, #32, #33).
```

- [ ] **Step 2: Verify it renders**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import pathlib
text = pathlib.Path('CHANGELOG.md').read_text()
assert '[0.0.5]' in text
assert 'EngineConformanceTests' in text
print('CHANGELOG.md OK,', len(text), 'chars')
"
```

Expected: `CHANGELOG.md OK, NNN chars`.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): 0.0.5 — conformance test suite

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Final lint + tests + push

**Files:**
- (no edits — just verification)

- [ ] **Step 1: Run ruff**

```bash
cd /home/liger/pyiron_workflow_atomistics
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check pyiron_workflow_atomistics/ tests/
```

Expected: `All checks passed!`. If anything trips, run `ruff check --fix pyiron_workflow_atomistics/ tests/` then re-check.

- [ ] **Step 2: Run ruff import-sort**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check --select I pyiron_workflow_atomistics/ tests/
```

Expected: `All checks passed!`.

- [ ] **Step 3: Run black**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m black --check pyiron_workflow_atomistics/ tests/
```

Expected: `All done! ✨ … N files would be left unchanged.`. If anything wants reformatting, run `black pyiron_workflow_atomistics/ tests/` then re-stage, amend the last commit (`git commit --amend --no-edit`), or commit as a small `style:` follow-up.

- [ ] **Step 4: Run the full unit test suite excluding `slow`**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header -m "not slow"
```

Expected: 89 passed + 3 pre-existing version-baseline failures (the `test_version_*` and `test_tests.py::test_version` cases that have been red on `origin/main` since before this branch — they're not regressions). Anything else is the implementer's responsibility to investigate.

- [ ] **Step 5: Push the branch**

```bash
git push origin design-engine-conformance-suite
```

Expected: branch tip on remote updated. If divergent, the implementer should `git pull --rebase origin design-engine-conformance-suite` and re-push.

---

## Task 10: Open the PR

**Files:**
- (no edits)

- [ ] **Step 1: Open PR via gh CLI**

```bash
gh pr create \
  --title "feat(testing): EngineConformanceTests pytest mixin (v0.0.5)" \
  --body "$(cat <<'EOF'
## Summary

Adds \`pyiron_workflow_atomistics.testing.EngineConformanceTests\` — a reusable pytest mixin that downstream engine packages subclass to verify their \`Engine\` Protocol implementation against a single source-of-truth bar.

## Why now

The Engine Protocol is \`@runtime_checkable\`, but \`isinstance(eng, Engine)\` only verifies the *shape* of the class — not that \`with_working_directory\` is pure, not that the engine pickles, not that \`get_calculate_fn\` honours its signature contract, not that \`run()\` returns a real \`EngineOutput\`. Each downstream (\`pyiron_workflow_lammps\`, \`pyiron_workflow_vasp\`) currently has to reinvent this harness — contract drift is inevitable.

## What's in this PR

- New \`pyiron_workflow_atomistics/testing/engine_conformance.py\`: a 5-method mixin (one method per contract clause).
- New \`pyiron_workflow_atomistics/testing/__init__.py\`: re-exports the mixin.
- One added line in the top-level \`__init__.py\`: \`from . import testing\`.
- New \`tests/unit/engine/test_ase_conformance.py\`: in-tree subclass proving the mixin is correct against \`ASEEngine(EMT)\`.
- New \`CHANGELOG.md\` documenting the 0.0.5 addition.

## What's NOT in this PR

- No existing symbol moved, renamed, or deleted. Purely additive.
- No MD-mode conformance assertions (deferred — neither LAMMPS nor VASP migrations target MD this cycle).
- No numerical accuracy assertions (downstream concern — each engine pins its own goldens).

## Release sequence

After merge:

1. Tag the merge commit: \`git tag pyiron_workflow_atomistics-0.0.5 <sha> && git push origin pyiron_workflow_atomistics-0.0.5\`
2. The pyiron \`pyproject-release.yml\` shared workflow auto-publishes to PyPI on the tag.
3. Only after 0.0.5 is on PyPI do the LAMMPS and VASP migration PRs open (they pin \`pyiron-workflow-atomistics==0.0.5\`).

## Test plan

- [x] \`pytest tests/unit -m "not slow"\` — 89 passed + 3 pre-existing version-baseline failures.
- [x] \`pytest tests/unit/engine/test_ase_conformance.py -v\` — 5 passed.
- [x] \`ruff check\` + \`ruff check --select I\` + \`black --check\` all green.
- [ ] CI runs against ubuntu-3.10/3.11/3.12 + macos-3.12 — all engine tests pass on every platform.

## Spec

\`docs/design/specs/2026-05-12-engine-conformance-suite-design.md\`. Plan: \`docs/design/plans/2026-05-12-engine-conformance-suite.md\`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 2: Verify CI status**

After ~3 minutes:

```bash
gh pr checks $(gh pr view --json number --jq .number)
```

Expected: all checks (ruff-check, ruff-sort-imports, black, build-docs, build-notebooks, coverage, pip-check, pypi-release, unit-tests × 4 platforms, benchmark-tests) PASS. Pre-existing version-test failures only.

If anything fails: read the log, fix the smallest possible thing, push a follow-up commit. Don't squash-merge until green.

---

## Task 11: Post-merge release (not part of the PR; runs after merge)

**Files:**
- (no edits)

- [ ] **Step 1: Squash-merge the PR**

```bash
gh pr merge $(gh pr view --json number --jq .number) --squash \
  --subject "feat(testing): EngineConformanceTests pytest mixin (v0.0.5) (#$(gh pr view --json number --jq .number))" \
  --body "Adds pyiron_workflow_atomistics.testing.EngineConformanceTests, the reusable pytest mixin used by pyiron_workflow_lammps and pyiron_workflow_vasp to verify Engine Protocol conformance. Purely additive — no existing symbol moves or breaks. Ships as v0.0.5; tag-and-publish follows.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

Expected: PR state becomes `MERGED`; squash commit SHA reported.

- [ ] **Step 2: Tag the merge commit**

```bash
cd /home/liger/pyiron_workflow_atomistics
git checkout main
git pull origin main
MERGE_SHA=$(git rev-parse HEAD)
git tag pyiron_workflow_atomistics-0.0.5 "$MERGE_SHA"
git push origin pyiron_workflow_atomistics-0.0.5
echo "Tagged $MERGE_SHA as pyiron_workflow_atomistics-0.0.5"
```

Expected: tag created locally + pushed to origin.

- [ ] **Step 3: Verify the PyPI release workflow fires**

```bash
gh run list --workflow=pyproject-release.yml --limit 3
```

Expected: a new run triggered by the tag push, in `in_progress` / `queued` state. Wait ~2 minutes then re-check; expected status `success`.

- [ ] **Step 4: Verify the PyPI page**

After the release workflow succeeds:

```bash
curl -s https://pypi.org/pypi/pyiron-workflow-atomistics/json | \
  /home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c \
  "import json, sys; d = json.load(sys.stdin); print('version:', d['info']['version'])"
```

Expected: `version: 0.0.5`.

If the workflow is gated on a trusted-publisher manual approval, surface the URL to the maintainer instead of waiting.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Plan task |
|---|---|
| `pyiron_workflow_atomistics/testing/engine_conformance.py` module exists | Task 2 (scaffold) + Tasks 3–7 (build mixin) |
| `pyiron_workflow_atomistics/testing/__init__.py` re-exports the mixin | Task 2 |
| Top-level `__init__.py` registers `from . import testing` | Task 2 |
| `test_satisfies_engine_protocol` | Task 3 |
| `test_with_working_directory_is_pure` | Task 4 |
| `test_pickleable` | Task 5 |
| `test_get_calculate_fn_signature` | Task 6 |
| `test_run_returns_engine_output` | Task 7 |
| In-tree `ASEEngine` subclass proving the mixin | Task 3 (created) + Tasks 4–7 (gain methods via inheritance) |
| Version bump 0.0.4 → 0.0.5 via versioneer | Task 11 (the tag IS the version bump per versioneer) |
| `CHANGELOG.md` with 0.0.5 section | Task 8 |
| Tag `pyiron_workflow_atomistics-0.0.5` after merge | Task 11 |
| Auto-publish via `pyproject-release.yml` on tag | Task 11 |
| All existing tests still pass (no regression) | Task 2 Step 7, Task 7 Step 3, Task 9 Step 4 |

Every spec requirement maps to a task. No gaps.

**Placeholder scan:** No "TBD", "TODO", or "implement later" tokens. Every code step has the actual code; every command step has the actual command + expected output.

**Type consistency:** `EngineConformanceTests` class shape (`engine_factory` / `test_structure_factory` class attrs, 5 `test_*` methods) is consistent across Tasks 2–7. Method names exactly match the spec.

**Identifier consistency:** `bulk("Cu","fcc",a=3.6,cubic=True)` is the default structure in Tasks 6–7; same expression on the same line everywhere. `ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path))` is the factory shape in Tasks 3 and 7 — identical.

---

## Execution Handoff

**Plan complete and saved to `docs/design/plans/2026-05-12-engine-conformance-suite.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
