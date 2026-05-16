# Changelog

All notable changes to `pyiron_workflow_atomistics` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package follows [PEP 440](https://peps.python.org/pep-0440/) versioning
via `versioneer`.

## [0.0.11] — 2026-05-16

### Fixed

- **`physics.grand_canonical_gb` GB-energy formula.** The ported
  ``gb_energy()`` followed GRIP's ``(E_total − n_mask·E_coh) / area``
  formulation, which only works when the calculator provides per-atom
  energy decomposition (LAMMPS-native). Under any ASE engine the full
  slab energy was summed against a partial atom mask, producing
  non-physical negative excess that clamped to 100 J/m² every iteration.
  Now matches ``physics.grain_boundary.get_GB_energy``:

      Egb = (E_total − N·E_coh) / (2·area) · 16.021766    [J/m²]

  where ``N`` is the **total** atom count in the slab and the factor of 2
  accounts for the two equivalent GB planes in a periodic bicrystal.
  The private ``_count_atoms_in_gb_region`` helper is removed.
- The ``n_gb_atoms`` parameter of ``gb_energy()`` is renamed to
  ``n_atoms`` to reflect the new semantics. The signature change is
  internal to ``_grand_canonical_gb_code`` (not part of the public
  ``physics.grand_canonical_gb`` API), so consumers of ``gco_search``
  see no breakage.

## [0.0.10] — 2026-05-16

### Added

- **`pyiron_workflow_atomistics.physics.grand_canonical_gb`** — new
  subpackage for grand-canonical optimization (GCO) of grain-boundary
  phases. Port of the algorithm from Chen, Heo, Wood, Asta, Frolov,
  *Nat. Commun.* **15**, 7049 (2024) (upstream:
  https://github.com/enze-chen/grip). Two public function-nodes:
  - ``gco_search(minimize_engine, lower_slab, upper_slab, e_cohesive,
    config, n_iters, md_engine=None, seed=0, dlat=0.0)``
    — sequential per-seed GCO loop returning a kept-structure DataFrame
    plus the corresponding list of ``ase.Atoms``. Parallelism across
    seeds is composed by the caller via ``for_node``.
  - ``build_bicrystal_slabs(crystal, symbol, a, upper_dirs, lower_dirs,
    c=0.0, cutoff=35.0, size_z=15)`` — convenience slab builder for fcc/bcc/hcp/
    dc/sc.
- **`GCOConfig`** dataclass — all algorithmic knobs (geometry,
  sampling, MD, dedup). Pre-loop validation rejects inconsistent
  configs; warnings on sketchy ones.

### Out of scope (v2 follow-ups)

- LAMMPS-engine MD integration (waits on ``pyiron_workflow_lammps``
  shipping ``CalcInputMD`` support; ``gco_search`` will work unchanged).
- Multi-component composition sampling (GRIP ``inter_w`` / ``inter_s``).
- ``plot_gco.py`` equivalent (existing matplotlib idioms in
  ``physics.grain_boundary`` cover n-vs-Egb plots).
- Aggressive ``extra=True`` dedup pass.

## [0.0.9] — 2026-05-15

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

## [0.0.8] — 2026-05-15

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
