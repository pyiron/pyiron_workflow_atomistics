# Cleanup and reorganise `pyiron_workflow_atomistics` — Design

| Field | Value |
|---|---|
| Date | 2026-05-12 |
| Owner | @ligerzero-ai |
| Status | Draft — awaiting user approval |
| Scope | `pyiron_workflow_atomistics` only (no `pyiron_workflow_vasp` changes this pass) |
| Branch | `cleanup-and-reorganise` (off `origin/main` at `39f006e`) |
| Follow-up | A separate PR will add the `Task`/`Workload` abstraction (approach 2) once this lands |

## Goal

Make `pyiron_workflow_atomistics` match this vision:

- A **generic `Engine` interface** that physics workflows talk to uniformly at every level of the graph.
- Engines receive **physics-level input dataclasses** (force tolerance, temperature, ensemble) — no code-specific jargon like `EDIFFG` or `units metal`.
- **Physics workflows organised by topic** into modules: `bulk`, `surface`, `point_defect`, `grain_boundary`.
- **Generic structure-manipulation functions** in their own module (`structure/`), reused freely by physics workflows.
- Only **node functions genuinely useful to end users** are exposed; kwargs-plumbing scaffolding is hidden.
- Layout and naming are **legible to LLM agents**: predictable import paths, curated `__init__.py`, typed APIs, examples in every public docstring.

## Conflicts with the goal in the current codebase

1. **`Engine` is half a protocol, half a dataclass.** `dataclass_storage.py:31` decorates `Engine` with `@dataclass` *and* gives it an explicit `__init__`; no `abstractmethod`, no `Protocol`. The real contract (`get_calculate_fn(structure) → (fn, kwargs)`, `working_directory`, `.copy()`) is duck-typed by every consumer.

2. **Code-specific jargon leaked into "physics" dataclasses.** `CalcInputMD` docstring references *"LAMMPS units style"* (`dataclass_storage.py:65`) even though no LAMMPS engine exists. `EngineOutput` is a `PrintableClass` with class-level `None` attributes carrying five near-identical stress fields.

3. **Triple duplication inside `engine_ase/`.** `ase.py`, `ase_calculator.py`, and `ase_engine.py` each redefine `fillin_default_calckwargs`, `generate_kwargs_variant`, `generate_kwargs_variants`, and a copy of `extract_values`/`calculate_structure_node`. `ase_calculator.py` is canonical; the other two are stale earlier versions.

4. **Kwargs-plumbing nodes leak into the user API.** `calculator.py` exports `fillin_default_calckwargs`, `generate_kwargs_variant`, `generate_kwargs_variants`, `add_arg_to_kwargs_list`, `validate_calculation_inputs`, `convert_EngineOutput_to_output_dict` — none of which a user composing a physics workflow needs.

5. **Module boundaries unenforced.** Every package `__init__.py` is empty (`structure_manipulator/`, `surface/`, `gb/`, `bulk_defect/`). `bulk_defect/vacancy.py` and `bulk_defect/point.py` define the *same* three functions; `duplicate_engine` is defined in 3 places.

6. **Engines and physics workflows are tangled.** `bulk.py:200` reaches into engine internals via `get_calc_fn_calc_fn_kwargs_from_calculation_engine` and pokes `engine.working_directory` directly. `surface/surface_study.py:138` instantiates `_calculate_if_not_present_` that bypasses the workflow graph.

7. **Repository debris.** Checked-in `pyiron_workflow_atomistics_test/` venv, `pyiron_module_template/` stub, tracked `.pytest_cache/` and `__pycache__/`, root-level `working_surface_example.py`, parallel `hcp_gb_generator/` sub-project. `gb/hcp_generator.py` duplicates that sub-project's purpose.

8. **Dependency reality check.** Installing this into the same env as `pyiron_workflow_vasp` upgrades `pyiron_workflow 0.13.3 → 0.15.2`, `pymatgen 2023.10.11 → 2025.6.14`, etc. — out-of-scope for this PR, but worth noting for the VASP integration follow-up.

## Approach: surgical cleanup, same architecture

Keep the existing `Engine.get_calculate_fn` contract — it's a fine pattern — but formalise it as a `typing.Protocol`, dedupe `engine_ase/` to one file, replace `EngineOutput` with a real `@dataclass`, hide kwargs plumbing under `_internal/`, populate every `__init__.py` with a curated public API, and split modules so each has one obvious responsibility.

Approach 2 (extract a `Task`/`Workload` abstraction so physics workflows never touch engine internals) is deferred to a follow-up PR once this lands.

## Target module layout

```
pyiron_workflow_atomistics/
├── engine/                       # Engine protocol + concrete engines
│   ├── __init__.py               # exports: Engine, EngineOutput, run, CalcInputStatic,
│   │                             # CalcInputMinimize, CalcInputMD, ASEEngine
│   ├── protocol.py               # Engine Protocol + EngineOutput dataclass + run() node
│   ├── inputs.py                 # CalcInputStatic / CalcInputMinimize / CalcInputMD
│   └── ase.py                    # ASEEngine (all engine_ase/* merged)
│
├── structure/                    # Generic structure manipulation (engine-agnostic)
│   ├── __init__.py
│   ├── build.py                  # get_bulk, create_surface_slab
│   ├── transform.py              # add_vacuum, create_supercell*, rattle
│   └── defects.py                # create_vacancy, substitutional_swap, oct/tet site finders
│
├── physics/                      # Physics workflows (one module per topic)
│   ├── __init__.py               # empty by design — users import per topic
│   ├── bulk.py                   # eos_volume_scan, optimise_cubic_lattice_parameter, equation_of_state
│   ├── surface.py                # calculate_surface_energy
│   ├── point_defect.py           # get_vacancy_formation_energy, get_substitutional_formation_energy
│   └── grain_boundary.py         # pure_gb_study, cleavage_study, segregation_study
│
├── analysis/                     # Featurisation, post-processing, derived quantities
│   ├── __init__.py
│   ├── featurisers.py
│   ├── gb_plane.py               # find_gb_plane, plot_gb_plane
│   └── quantities.py             # get_per_atom_quantity
│
└── _internal/                    # NOT exported; private plumbing
    ├── kwargs_helpers.py
    ├── dataclass_helpers.py
    └── workdir.py
```

## Engine protocol and dataclasses

### `Engine` Protocol

```python
# engine/protocol.py
from typing import Any, Callable, Protocol, runtime_checkable
from ase import Atoms

@runtime_checkable
class Engine(Protocol):
    """An engine computes properties of a structure.

    Implementations: ASEEngine (this package), VaspEngine (pyiron_workflow_vasp,
    follow-up), LammpsEngine (future), etc.

    Engines MUST be pickleable so workflows can be checkpointed/submitted to
    SLURM. Engines MUST be copyable via .with_working_directory() without
    mutating self. These requirements are documented but not enforced via
    __reduce__; relying on duck typing keeps the contract simple.
    """

    working_directory: str

    def get_calculate_fn(
        self, structure: Atoms
    ) -> tuple[Callable[..., "EngineOutput"], dict[str, Any]]: ...

    def with_working_directory(self, subdir: str) -> "Engine":
        """Return a *copy* of this engine whose working_directory is joined
        with ``subdir``. Pure — never mutates self. Replaces the
        free-floating `duplicate_engine` function currently triple-defined."""
```

### `EngineOutput` dataclass

```python
@dataclass
class EngineOutput:
    final_structure: Atoms
    final_energy: float
    converged: bool

    final_forces: np.ndarray | None = None
    final_stress: np.ndarray | None = None            # shape (3, 3)
    final_stress_voigt: np.ndarray | None = None      # shape (6,) — kept for convenience
    final_volume: float | None = None
    final_magmoms: np.ndarray | None = None

    energies: list[float] | None = None
    forces: list[np.ndarray] | None = None
    stresses: list[np.ndarray] | None = None
    structures: list[Atoms] | None = None
    n_ionic_steps: int | None = None

    def to_dict(self) -> dict[str, Any]: ...
```

Both stress representations are kept (per user decision). Engines that natively produce only Voigt fill `final_stress_voigt` and leave `final_stress` derived (or vice versa).

### `EngineInput` dataclasses

Existing `CalcInputStatic` / `CalcInputMinimize` / `CalcInputMD` keep their structure; cleanup is jargon-strip only.

| Change | Reason |
|---|---|
| Strip *"LAMMPS units style"* from `CalcInputMD` docstring | Vestigial; no LAMMPS engine exists |
| `time_step` units `ps → fs` | ASE and LAMMPS native is fs; saves a bug-prone ×1000 |
| Drop `delta_temp`, `delta_press` fields | Unused; would need a "ramp" mode |
| Rename `temperature_damping_timescale → thermostat_time_constant` | Less verbose, clearer physics meaning |

### Single execution entry point

```python
# engine/protocol.py
@pwf.as_function_node("engine_output")
def run(structure: Atoms, engine: Engine) -> EngineOutput:
    """Execute `engine` on `structure`. The one node every physics workflow
    uses to compute things."""
    fn, kwargs = engine.get_calculate_fn(structure)
    return fn(structure=structure, **kwargs)
```

Replaces `calculator.calculate_structure_node` (which had a dual-mode duck-typed interface). The `validate_calculation_inputs` node is deleted alongside.

## Public API per module

Rule: every public symbol re-exports from a top-level subpackage `__init__.py`. Anything else is internal.

### `engine/`

```python
from .protocol import Engine, EngineOutput, run
from .inputs   import CalcInputStatic, CalcInputMinimize, CalcInputMD
from .ase      import ASEEngine

__all__ = [
    "Engine", "EngineOutput", "run",
    "CalcInputStatic", "CalcInputMinimize", "CalcInputMD",
    "ASEEngine",
]
```

### `structure/`

```python
from .build     import get_bulk, create_surface_slab
from .transform import add_vacuum, create_supercell, create_supercell_with_min_dimensions, rattle
from .defects   import create_vacancy, substitutional_swap, find_octahedral_sites, find_tetrahedral_sites

__all__ = [
    "get_bulk", "create_surface_slab",
    "add_vacuum", "create_supercell", "create_supercell_with_min_dimensions", "rattle",
    "create_vacancy", "substitutional_swap", "find_octahedral_sites", "find_tetrahedral_sites",
]
```

### `physics/` — topical imports only

```python
# physics/__init__.py is empty by design. Users write:
from pyiron_workflow_atomistics.physics.bulk            import eos_volume_scan, optimise_cubic_lattice_parameter
from pyiron_workflow_atomistics.physics.surface         import calculate_surface_energy
from pyiron_workflow_atomistics.physics.point_defect    import get_vacancy_formation_energy
from pyiron_workflow_atomistics.physics.grain_boundary  import pure_gb_study
```

Per module:
- `physics.bulk`: `eos_volume_scan`, `optimise_cubic_lattice_parameter`, `equation_of_state`
- `physics.surface`: `calculate_surface_energy`
- `physics.point_defect`: `get_vacancy_formation_energy`, `get_substitutional_formation_energy` (new)
- `physics.grain_boundary`: `pure_gb_study`, `cleavage_study`, `segregation_study`

### `analysis/`

```python
from .featurisers import (
    voronoi_site_featuriser,         # was voronoiSiteFeaturiser
    distance_matrix_site_featuriser, # was distanceMatrixSiteFeaturiser
    soap_site_featuriser,            # was soapSiteFeaturiser
    summarize_cosine_groups,
    pca_whiten,
)
from .gb_plane   import find_gb_plane, plot_gb_plane
from .quantities import get_per_atom_quantity
```

Featuriser names rename camelCase → snake_case as part of this PR (we're touching every public API anyway; PEP-8 conformance comes for free). Internal helpers and non-public symbols keep their existing names.

### Deleted / moved-to-`_internal/`

| Current symbol | Fate |
|---|---|
| `calculator.fillin_default_calckwargs` | `_internal/kwargs_helpers.py` |
| `calculator.generate_kwargs_variant` / `_variants` | `_internal/kwargs_helpers.py` |
| `calculator.add_arg_to_kwargs_list` | **deleted** (one inlinable call site) |
| `calculator.validate_calculation_inputs` | **deleted** (single-path engine API) |
| `calculator.convert_EngineOutput_to_output_dict` | **deleted** (use `EngineOutput.to_dict()`) |
| `calculator.extract_output_values_from_EngineOutput` | **deleted** (`.energies`, `.forces` accessible directly) |
| `calculator.extract_values_from_dict` | **deleted** (one-liner) |
| `utils.duplicate_engine` (×3 definitions) | **deleted** → `Engine.with_working_directory()` |
| `utils.add_string` | **deleted** (`+` operator) |
| `utils.convert_structure` | **deleted** (one-liner) |
| `utils.get_calc_fn_calc_fn_kwargs_from_calculation_engine` | **deleted** (use `engine.get_calculate_fn`) |
| `utils.modify_dataclass` / `_multi` / `modify_dict` | `_internal/dataclass_helpers.py` |
| `utils.get_subdirpaths` / `get_working_subdir_kwargs` | `_internal/workdir.py` |
| `utils.get_per_atom_quantity` | `analysis/quantities.py` |
| `bulk_defect/point.py` (duplicate of `vacancy.py`) | **deleted** |
| `engine_ase/ase.py` (stale duplicate) | **deleted** |
| `engine_ase/ase_engine.py` (stale duplicate) | **deleted** |
| `dataclass_storage.BuildBulkStructure_Input` | **deleted** (unused) |
| `dataclass_storage.PrintableClass` | **deleted** (replaced by `@dataclass`) |

**Public node count: ~35 → ~22** — every survivor is something a user puts in their graph.

### LLM-agent affordances

- `__all__` in every public `__init__.py`.
- Numpy-style docstrings with **examples** on every public symbol.
- Module-level docstring at the top of each `__init__.py`: what the module does, the 2–4 entry points, where internals live.
- `py.typed` marker so LSP/Pyright resolves types for consumers.

## Migration plan — 6 commits on `cleanup-and-reorganise`

Each commit leaves the repo importable and the then-current test suite green. Order minimises cross-file churn.

### Commit 1 — `chore: repo hygiene`

Pure deletion + `.gitignore`. No source code touched.

- Delete `pyiron_workflow_atomistics_test/` (checked-in venv, ~7 MB).
- Delete `pyiron_module_template/` (empty template).
- Delete `working_surface_example.py` (root-level dev stray).
- Delete tracked `__pycache__/`, `.pytest_cache/` (top-level and inside `hcp_gb_generator/`).
- `.gitignore` adds: `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `*.venv*/`, `pyiron_workflow_atomistics_test/`.
- `hcp_gb_generator/` stays untouched (out of scope; flagged for a follow-up PR).

### Commit 2 — `refactor(engine): protocol, EngineOutput dataclass, consolidate engine_ase`

The "blast radius" commit. Atomic — no compatibility shims. Every importer flips to the new `engine.*` paths in this commit so later commits don't need transitional plumbing.

- Create `engine/protocol.py` with `Engine` Protocol, `EngineOutput` dataclass, `run()` node.
- Create `engine/inputs.py` from `dataclass_storage.py` (jargon-strip, drop unused fields, rename `temperature_damping_timescale → thermostat_time_constant`, time-step units → fs).
- Create `engine/ase.py`:
  - Canonical `ase_calc_structure` and `ase_md_calc_structure` from `ase_calculator.py`.
  - Returns the new `EngineOutput` dataclass (drop `final_results: dict`, drop duplicate stress reps, fill both `final_stress` (3,3) and `final_stress_voigt` (6,)).
  - `ASEEngine` gains `with_working_directory()` (typically a one-liner using `dataclasses.replace`).
- Update every physics module (`bulk.py`, `surface/surface_study.py`, `bulk_defect/*.py`, `gb/*.py`) to import from `engine.protocol` / `engine.ase` / `engine.inputs` and to use `engine.run` instead of `calculator.calculate_structure_node`. Files stay in their current physical locations (commit 3 moves them).
- Delete `engine_ase/__init__.py`, `engine_ase/ase.py`, `engine_ase/ase_calculator.py`, `engine_ase/ase_engine.py`.
- Delete `dataclass_storage.py` and `calculator.py` (their public surface is now in `engine/`).
- `engine/__init__.py` populated with the 8-symbol curated API.

This commit ends with the engine layer in its final form. The rest of the cleanup is plumbing relocation around it.

### Commit 3 — `refactor(physics+structure): module reshuffle`

Pure relocation. Mostly `git mv` plus a few extractions. No behaviour change; only paths and a handful of public-symbol renames.

- `structure_manipulator/tools.py` → `structure/transform.py` (rename `rattle_structure` → `rattle`).
- `structure_manipulator/interstitials.py` → `structure/defects.py`; `substitutional_swap_one_site` → `substitutional_swap`.
- `bulk_defect/vacancy.py:create_vacancy_structure` → `structure/defects.py:create_vacancy`.
- New `structure/build.py`: `get_bulk_structure` (renamed `get_bulk`) and `create_surface` (renamed `create_surface_slab`).
- `bulk.py` → `physics/bulk.py` (workflow macros only).
- `surface/surface_study.py` → `physics/surface.py`; delete `surface/builder.py` (its helpers moved to `structure/build.py`).
- `bulk_defect/{point.py, vacancy.py}` consolidated → `physics/point_defect.py`.
- `gb/{gb_study.py, cleavage.py, segregation.py, optimiser.py, utils.py, hcp_generator.py}` → `physics/grain_boundary.py`. `gb/gb_code/` (the `lz-GB-code` searcher/constructor wrappers) moves intact as the sub-package `physics/_grain_boundary_code/` — leading underscore signals "internal to grain_boundary"; not re-exported from `physics/__init__.py` or `physics/grain_boundary.py`.
- `gb/analysis.py` → `analysis/gb_plane.py`.
- `featurisers.py` → `analysis/featurisers.py` (camelCase → snake_case rename of the three public featuriser functions, per the API table above).
- `utils.py:get_per_atom_quantity` → `analysis/quantities.py`.

All imports were already updated to point to the new `engine.*` modules in commit 2, so this commit's diff is overwhelmingly `git mv`. The renames (`rattle_structure → rattle`, `substitutional_swap_one_site → substitutional_swap`, `get_bulk_structure → get_bulk`, `create_surface → create_surface_slab`, the featurisers) are applied in-place during the move.

### Commit 4 — `refactor(engine): replace duplicate_engine with Engine.with_working_directory`

- Add `with_working_directory()` to `ASEEngine`.
- Replace every call site of `duplicate_engine(engine, subdir)` in `physics/*.py` with `engine.with_working_directory(subdir)`.
- Delete `utils.duplicate_engine` and the two redundant copies.
- Replace open-coded `os.makedirs(working_directory, exist_ok=True)` + `engine.working_directory` poke in `bulk.evaluate_structures` with `engine.with_working_directory(...)` calls.

### Commit 5 — `refactor(internal): hide kwargs plumbing under _internal/`

- Create `_internal/kwargs_helpers.py` (single canonical `fillin_default_calckwargs`, `generate_kwargs_variant(s)`).
- Create `_internal/dataclass_helpers.py` (`modify_dataclass`, `modify_dataclass_multi`, `modify_dict`).
- Create `_internal/workdir.py` (`get_subdirpaths`, `get_working_subdir_kwargs`).
- Delete top-level `utils.py` (its public symbols already migrated: `get_per_atom_quantity` → `analysis/quantities.py` in commit 3; `duplicate_engine` deleted in commit 4; the remaining helpers move to `_internal/` in this commit, the rest of the file's symbols — `add_string`, `convert_structure`, `get_calc_fn_calc_fn_kwargs_from_calculation_engine` — are deleted outright).
- (`calculator.py` and `dataclass_storage.py` were already removed in commit 2.)
- Inline call sites: any remaining users of the moved helpers are updated to import from `pyiron_workflow_atomistics._internal.*`.

### Commit 6 — `refactor(api): curate __init__.py exports, fix tests/notebooks`

- Populate every public `__init__.py` with the `__all__` lists from the "Public API per module" section.
- Add `py.typed` marker at package root.
- **Tests**: remove tests for deleted symbols (`add_arg_to_kwargs_list`, `validate_calculation_inputs`, `add_string`, `convert_structure`, `get_calc_fn_calc_fn_kwargs_from_calculation_engine`, `convert_EngineOutput_to_output_dict`, `extract_values_from_dict`, `extract_output_values_from_EngineOutput`). Update import paths on the rest. Add new unit tests for `physics/surface.py` and `physics/point_defect.py` (no current coverage), Engine-protocol conformance (including a `pickle.dumps`/`pickle.loads` round-trip on `ASEEngine(EMT(), ...)`), and the notebook execution integration test.
- **Notebooks**: mechanical port of imports + full re-execution (10-min timeout each, ASE/EAM calculators per the coverage table). Outputs committed (per user decision).

### Estimated diff size

| Category | + LOC | − LOC |
|---|---|---|
| Engine consolidation (commit 2) | ~500 | ~1300 |
| Physics/structure reshuffle (commit 3) | ~50 | ~50 |
| `duplicate_engine` cleanup (commit 4) | ~30 | ~80 |
| `_internal/` move (commit 5) | ~80 | ~400 |
| `__init__.py` curation + tests + notebooks (commit 6) | ~300 | ~600 |
| **Total** | **~960** | **~2430** |

Net **−1470 LOC** for the same functionality + a much cleaner contract.

## Testing

Tests get the same physical layout as the source.

```
tests/
├── conftest.py                       # kept; reusable fixtures unchanged
├── unit/
│   ├── engine/{test_protocol, test_inputs, test_ase}.py
│   ├── structure/{test_build, test_transform, test_defects}.py
│   ├── physics/{test_bulk, test_surface, test_point_defect, test_grain_boundary}.py
│   └── analysis/{test_featurisers, test_gb_plane}.py
├── integration/
│   ├── test_notebook_execution.py    # NEW — see below
│   └── test_readme.py                # kept
└── benchmark/test_benchmark.py       # kept
```

Net test diff: removes ~600 lines of plumbing-symbol tests; adds ~250 lines of new physics tests, Engine-protocol conformance, and the notebook integration test.

### Notebook execution

One integration test drives every notebook via `nbclient` (already implicitly available through `jupyter`). 10-minute timeout per notebook (per user decision). Outputs **committed** after a clean re-run.

```python
# tests/integration/test_notebook_execution.py
import pathlib
import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.slow
def test_notebook_runs(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3").execute()
```

The CI workflow (`.github/workflows/push-pull.yml`) calls a shared pyiron-org reusable workflow and is **not modified** in this PR.

### Notebook coverage — one per physics interface

The Fe-based notebooks keep using the checked-in `notebooks/Al-Fe.eam.fs` EAM potential (per user decision). EMT is used only as a fallback for new demos whose physics doesn't constrain the element.

| Physics macro | Notebook | Calculator |
|---|---|---|
| `physics.bulk.eos_volume_scan` | `eos.ipynb` (merge of `equations_of_state.ipynb` + `equations_of_state_ase.ipynb`) | EAM / Fe |
| `physics.bulk.optimise_cubic_lattice_parameter` | `optimise_lattice_parameter.ipynb` (extracted) | EAM / Fe |
| `physics.bulk.equation_of_state` | covered in `eos.ipynb` | — |
| `physics.surface.calculate_surface_energy` | `surface_energy.ipynb` (ported) | EAM / Fe |
| `physics.point_defect.get_vacancy_formation_energy` | `vacancy_formation_energy.ipynb` (ported) | EAM / Fe |
| `physics.point_defect.get_substitutional_formation_energy` | **NEW** `substitutional_formation_energy.ipynb` (dilute Ni-in-Cu) | EMT |
| `physics.grain_boundary.pure_gb_study` | `pure_grain_boundary_study.ipynb` (ported) | EAM / Fe |
| `physics.grain_boundary.cleavage_study` | **NEW** `gb_cleavage.ipynb` (extracted from `pure_gb_study` for standalone clarity) | EAM / Fe |
| `physics.grain_boundary.segregation_study` | `grain_boundary_segregation.ipynb` (ported) | EAM / Fe |

Also ported: `structure_optimisation.ipynb` (basic `engine.run`), `bulk_solution_energy.ipynb`. Deleted: `notebook_blank.ipynb` (literal stub).

Final count: 9 notebooks (2 new, 2 merged/deleted).

## Repository hygiene checklist

- Add `py.typed` marker at `pyiron_workflow_atomistics/py.typed` (empty file).
- Strip large notebook outputs *not* enforced — re-executed notebooks commit with fresh outputs; matplotlib figures stay deterministic via fixed `np.random.seed(0)` and non-stochastic EMT/EAM.
- Keep `notebooks/Al-Fe.eam.fs` checked in (per user decision).

## Risk register

1. **Notebook re-execution may surface issues** with the new `engine.run` calling convention. Mitigated by running them locally as part of commit 6, before merge.
2. **`hcp_gb_generator/` sub-project stays untouched.** `physics/grain_boundary.py` imports from `lz-GB-code` (pinned external package, unchanged).
3. **Downstream consumers (e.g. `pyiron_workflow_vasp`)** will break against the new layout. Deprecation shims declined per user input; affected repos pin to a pre-cleanup commit until they catch up.
4. **`pyiron_workflow 0.13.3 → 0.15.2` API drift** not audited. If `@as_function_node` / `@as_macro_node` / `wf.<child>.outputs` semantics changed, that surfaces in commit 6's test run. Easy to fix incrementally.
5. **Pickle-safety of `ASEEngine`** is documented but not enforced; an `ase.calculators.Calculator` instance held by the engine may not always pickle cleanly (e.g. some MLIP backends hold non-pickleable handles). A conformance test will assert pickle round-trip with `EMT()`; backends that fail can be flagged in their own `__getstate__`/`__setstate__`.

## Out of scope

- `pyiron_workflow_vasp` changes (separate PR after this lands).
- The `Task` / `Workload` abstraction (approach 2, follow-up PR).
- Splitting `hcp_gb_generator/` into its own repo.
- Updating the shared `pyiron/actions` CI workflow.
- Re-pinning `pyiron_workflow_vasp` to match this package's dependencies.
