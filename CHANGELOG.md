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
