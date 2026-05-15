# Changelog

All notable changes to `pyiron_workflow_atomistics` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package follows [PEP 440](https://peps.python.org/pep-0440/) versioning
via `versioneer`.

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

## [0.0.7] — 2026-05-14

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

## [0.0.6] — 2026-05-13

### Changed (breaking)

- **Renamed `pyiron_workflow_atomistics.engine.run` → `.calculate`.** The
  function-node previously called `run` collided with `Workflow.run`
  (pyiron_workflow's workflow-execution method), so every
  `wf.X = run(structure, engine=…)` site needed an explicit
  `label="X"` workaround to avoid

      AttributeError: run is an attribute or method of the
      <class 'pyiron_workflow.workflow.Workflow'> class, and cannot
      be used as a child label.

  Renaming to `calculate` removes the collision; the `label=…` argument
  is no longer required and the new name aligns with the existing
  `Engine.get_calculate_fn` / `calc_fn` terminology elsewhere in the
  contract. Migration:

      sed -i 's|\bfrom pyiron_workflow_atomistics.engine import \(.*\)\brun\b|from pyiron_workflow_atomistics.engine import \1calculate|g' your_files.py
      sed -i 's|\brun\.node_function\b|calculate.node_function|g' your_files.py
      # then drop any `, label="X"` you'd previously added to work
      # around the Workflow.run collision.

  Affects every `physics/*.py` macro internally, the
  `EngineConformanceTests` mixin's `test_run_returns_engine_output`
  method (now uses `calculate.node_function`), all 10 example
  notebooks, and downstream engine repos pinning
  `pyiron-workflow-atomistics==0.0.6+`.

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
