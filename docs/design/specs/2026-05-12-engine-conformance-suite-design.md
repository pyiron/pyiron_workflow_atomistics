# Engine conformance test suite

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-12 |
| Repo | `pyiron/pyiron_workflow_atomistics` |
| Parent design | [Cleanup & reorganisation](2026-05-12-pyiron-workflow-atomistics-cleanup-design.md) |
| Companion specs | `pyiron_workflow_lammps/docs/design/specs/2026-05-12-engine-protocol-migration-design.md`, `pyiron_workflow_vasp/docs/design/specs/2026-05-12-vasp-engine-design.md` |

## Problem

The Engine Protocol (`pyiron_workflow_atomistics.engine.protocol.Engine`) is `@runtime_checkable`, but `isinstance(eng, Engine)` only verifies the *shape* of the class — it does not exercise the contract:

- `with_working_directory` must return a new instance, not mutate self.
- The engine must round-trip through `pickle.dumps/loads` so workflows can be checkpointed.
- `get_calculate_fn(structure)` must return `(callable, kwargs)` and `structure` must not appear in the kwargs.
- The callable must return an `EngineOutput` dataclass instance.

Each downstream engine package (`pyiron_workflow_lammps`, `pyiron_workflow_vasp`, and any future backend) currently has to reinvent this harness. The result is contract drift — different engines pass different bars, regressions don't get caught uniformly, and PR review on each downstream relies on visual inspection of test code rather than verification against a single source of truth.

## Approach

Add a `pyiron_workflow_atomistics.testing.engine_conformance` module containing a `EngineConformanceTests` mixin. Downstream engine packages subclass it with a `engine_factory(tmp_path)` static method (and optionally a `test_structure_factory`) and run pytest. Every contract clause is exercised by a single named test method.

The suite is intentionally minimal at this revision:

- One test per Protocol attribute / method shape (`isinstance`, `is_dataclass`, `get_calculate_fn` return type).
- One test for `with_working_directory` purity (new instance, original untouched, path composed correctly).
- One test for pickle round-trip.
- One smoke `run()` test that asserts the returned object is an `EngineOutput` with the three required fields populated.

What is **out of scope** for this revision:

- MD-mode assertions. The Protocol allows `CalcInputMD` but neither LAMMPS nor VASP migrations target MD this cycle.
- Per-property numerical accuracy. The conformance suite verifies shape; numerical correctness is each downstream's concern (see the regression-verification gate in the LAMMPS migration spec).
- Property-test-style generative inputs. Just deterministic single-point on `bulk("Cu","fcc",a=3.6,cubic=True)` by default; downstreams can override via `test_structure_factory`.

## Components

```
pyiron_workflow_atomistics/
└── testing/
    ├── __init__.py          # re-exports EngineConformanceTests
    └── engine_conformance.py
tests/unit/engine/
└── test_ase_conformance.py  # ASEEngine subclass — proves the suite is correct
```

### `EngineConformanceTests` mixin shape

```python
class EngineConformanceTests:
    # Required override
    engine_factory: Callable                       # (tmp_path) -> Engine
    # Optional override (default: 4-atom Cu FCC)
    test_structure_factory: Callable | None = None

    def test_satisfies_engine_protocol(self, tmp_path): ...
    def test_with_working_directory_is_pure(self, tmp_path): ...
    def test_pickleable(self, tmp_path): ...
    def test_get_calculate_fn_signature(self, tmp_path): ...
    def test_run_returns_engine_output(self, tmp_path): ...
```

Subclassing pattern (the same shape both downstreams will use):

```python
class TestMyEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        return MyEngine(
            EngineInput=CalcInputStatic(),
            working_directory=str(tmp_path),
            ...
        )
```

## Public API change

`pyiron_workflow_atomistics/__init__.py` exposes the new sub-package:

```python
from . import testing   # noqa: F401  -- new
```

so consumers can import as `pyiron_workflow_atomistics.testing.EngineConformanceTests`.

## Versioning + release

This change is purely additive — no existing symbol moves or breaks — so it ships as a patch bump:

- `pyproject.toml` version: bumped via `versioneer` tag-based versioning (next tag = `pyiron_workflow_atomistics-0.0.5`).
- `CHANGELOG.md` (new): single section describing the addition + the downstream-engine migration cycle it enables.

### Release sequence

1. PR with the conformance suite + an `ASEEngine` conformance test (proving the suite is correct in-tree) + CHANGELOG entry merges to `main`.
2. Tag the merge commit `pyiron_workflow_atomistics-0.0.5`. `git tag pyiron_workflow_atomistics-0.0.5 <sha> && git push origin pyiron_workflow_atomistics-0.0.5`.
3. The pyiron `pyproject-release.yml` shared workflow auto-publishes `pyiron-workflow-atomistics==0.0.5` to PyPI on the tag push.
4. **Only after 0.0.5 is on PyPI** do the LAMMPS and VASP PRs open. Their pyproject pins `pyiron-workflow-atomistics==0.0.5`; the CI `pip-check` job would otherwise reject the install.

If trusted-publisher / manual approval gates anything, that's flagged during the atomistics PR's merge review.

## Risk register

1. **MD-mode contract not enforced**: a future engine with broken MD shape could still pass conformance. Acceptable — once a downstream needs MD-mode coverage, the suite gains a sibling `EngineMDConformanceTests` mixin then.
2. **`test_run_returns_engine_output` requires a working backend in CI**: LAMMPS will install `lammps` from conda-forge; VASP will substitute a fixture-based mock command. Each downstream owns its own CI footprint decision.
3. **Pickle test fails on engines that hold non-pickleable state** (e.g. an MLIP backend with a live `torch.nn.Module` reference): documented in the Protocol contract as a hard requirement; engines that need this must implement `__getstate__`/`__setstate__`.

## Out of scope

- Numerical accuracy of `run()`.
- Multi-structure / parallel execution tests.
- MD-mode conformance.
- Cross-engine integration tests (running the same physics macro against multiple engines back-to-back).
