# Changelog

All notable changes to `pyiron_workflow_atomistics` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package follows [PEP 440](https://peps.python.org/pep-0440/) versioning
via `versioneer`.

## [Unreleased]

### Fixed

- **MD-renormalised FC2 was fitted at the wrong displacement distance.**
  ``calculate_phonon_md_renormalisation`` generated its FC2 supercells with
  phono3py (0.03 Å, ``is_diagonal=False``) but re-derived the dataset with
  phonopy's defaults (0.01 Å, ``is_diagonal=True``), so on high-symmetry
  cells the fitted force constants were silently scaled by 3 (frequencies
  by √3) and on lower-symmetry cells the supercell-count guard raised. The
  displacement kwargs are now passed to both sides, the macro exposes
  ``displacement_distance`` / ``is_plusminus`` so callers can set them, and
  the guard compares the displaced positions element-wise rather than only
  the supercell count (#100).
  ``anharmonic_free_energy_dynaphopy`` and
  ``anharmonic_free_energy_dynaphopy_tdi`` expose the same two kwargs and
  forward them to the FC2 fit.
- **calphy free-energy nodes work with calphy 2.3.** The calphy adapter
  built its calculations with ``script_mode``, which calphy 2 removed, and
  the real-LAMMPS tests of ``physics.free_energy.calphy`` failed on each
  calphy release checked (1.5.6, 1.8.5 and 2.3.0). It now dispatches each mode to calphy's own
  ``routine_*`` functions as ``calphy.queuekernel`` does (``Alchemy`` for
  ``composition_scaling``, ``MeltingTemp.calculate_tm`` for
  ``melting_temperature``), backs the melting calculation with the input
  file ``MeltingTemp`` re-reads, and reads the melting temperature from
  ``calc_tm`` / ``tmerr``.
- **Nose-Hoover equilibration.** ``equilibration_control="nose-hoover"``
  (the default) was passed to calphy as ``"nose_hoover"``, which calphy
  treats as Berendsen, so every run equilibrated with Berendsen. It is
  now passed through verbatim.
- **Launcher handling for calphy 2.** calphy 2 builds the parallel launch
  line itself as ``<launcher> -np <cores> lmp``. Engine commands with
  launcher flags (``mpirun --bind-to none -np 2 lmp``), with ``srun``, or
  with a launcher but no core count are now rejected with a
  ``ValueError`` instead of failing inside calphy or silently running
  serially. Arguments after the binary (e.g. ``lmp -sf omp`` or
  ``lmp -k on g 1 -sf kk -pk kokkos``) are passed to every LAMMPS run
  through calphy's ``md.cmdargs``; ``-in`` / ``-log`` / ``-screen`` are
  dropped (calphy sets them) and multi-partition, restart-conversion,
  ``-skiprun`` and ``-help`` switches are rejected.
- ``LammpsPotential.potential_file`` is rejected with a ``ValueError``:
  calphy 2 no longer reads it, so the file was silently ignored.
- A relative potential file in ``pair_coeff`` (or ``pair_coeff_target``)
  is resolved against the caller's directory. calphy resolved it against
  the node's simfolder, so LAMMPS could not find the file.
- ``composition_scaling`` rejects a target composition equal to that of
  the input structure (calphy failed with an ``IndexError``), and fills
  ``composition_path`` from the calculation, since calphy 2's
  ``report.yaml`` no longer echoes the composition block. The target may
  now contain elements the structure lacks (pure Cu → Cu107Ni1); they
  become extra atom types after the structure's own elements, which
  ``pair_coeff`` must map in that order. Its tests run real Cu107Ni1 →
  Cu106Ni2 and Cu108 → Cu107Ni1 transformations with known answers.
- ``melting_temperature`` reports its working directory as ``simfolder``
  (no longer an empty extra folder) and records the solid / liquid sweep
  dissipation verdicts in ``report``, as calphy only logs a warning when
  a sweep was irreversible. When calphy's bracket search returns to a
  bracket it already tried (or a previous run left its folders behind),
  the folder-exists ``ValueError`` is re-raised as a ``RuntimeError``
  that names the cause and what to change.

### Changed

- **The ``[free-energy]`` extra requires calphy 2.3.**
  ``calphy>=1.5.6`` becomes ``calphy>=2.3.0,<2.4.0`` and
  ``pyiron_workflow_lammps`` becomes ``>=0.2.0,<0.3.0`` (0.1.x pins
  ``pyiron_workflow`` 0.15 and cannot be installed alongside this
  package). The extra did not work with any calphy release before, so no
  working setup breaks. Release this as a 0.3.x patch:
  ``pyiron_workflow_lammps`` 0.2.0 requires
  ``pyiron-workflow-atomistics<0.4.0``, so under a 0.4.0 the extra could
  not be resolved until a ``pyiron_workflow_lammps`` release raises that
  cap. calphy is distributed under the Academic Software Licence v1.0,
  which permits non-commercial academic use only; the README now says so
  where the extra is described, and names the LAMMPS build and packages
  calphy needs.
- The adapter no longer lists calphy's ``tscale`` mode, which no node
  exposed and the calculation builder rejected.

## [0.3.0] — 2026-10-22

### Changed

- **BREAKING: `structuretoolkit`, `pyscal3` and `scikit-learn` are no longer
  required dependencies.** They move to a new ``[analysis]`` extra, so
  packages that only use ``pyiron_workflow_atomistics.engine`` (e.g.
  ``pyiron_workflow_vasp``) no longer install them. They are needed only by
  the CNA / Voronoi descriptors in ``analysis.structure_descriptors``
  (``cna_fractions``, ``analyse_reference_structure``, ``classify_solid``,
  ``voronoi_max_mean``), the ``physics.melting`` workflows and the void
  analysis in ``structure.defects`` (``calculate_voids``,
  ``filter_and_cluster_atoms``). Install them with::

      pip install "pyiron_workflow_atomistics[analysis]"

  Every module still imports without them and all public names are
  unchanged; calling one of those functions without the extra raises a
  ``ModuleNotFoundError`` that names the extra (the ``physics.melting``
  workflows check before their first engine calculation). The conda-forge
  environments used for CI, docs and binder keep installing them.
- **BREAKING: `tqdm` is no longer a required dependency.** It only draws
  progress bars for the grain-boundary CSL search and moves to the ``[gb]``
  extra; the search runs without progress bars if it is missing.
- Importing ``pyiron_workflow_atomistics.analysis`` (and with it
  ``physics.bulk``, ``physics.elastic``, ``physics.grain_boundary`` and
  ``physics.free_energy``) no longer imports ``structuretoolkit``.

### Added

- **Core-install CI.** The clean-install workflow also installs the wheel
  without optional extras, checks the ``[analysis]`` packages are not pulled
  in, and runs the unit tests (tests needing an extra are skipped).

## [0.2.4] — 2026-10-22

### Added

- **Python 3.14 support.** ``requires-python`` is widened to
  ``>=3.11, <3.15`` and the 3.14 classifier is added. Every runtime
  dependency has CPython 3.14 wheels on PyPI and the pinned conda-forge
  environment solves for ``python=3.14``; the clean-install workflow and
  the upstream unit-test matrix now also run on 3.14.

## [0.2.3] — 2026-10-22

### Fixed

- **Installed `__version__`.** Built wheels shipped versioneer's
  git-querying ``_version.py`` template, so an installed package reported
  ``0+unknown``. ``versionfile_build`` is now set, so the wheel carries a
  static ``_version.py`` with the real release version, and a test checks
  that installed builds report it.
- **pyscal3 4 support.** pyscal3 4.0 removed the ``System`` class, which
  broke the void analysis in ``structure/defects.py``. It is ported to the
  pyscal3 4 API around ``ase.Atoms`` (void sites as placeholder species
  ``X``, per-atom data in ``structure.arrays``, results written as extended
  XYZ; the ``write`` argument of ``calculate_voids`` is now honoured).
  Pins move to ``pyscal3==4.1.0`` and ``structuretoolkit==0.0.48``.

### Changed

- **Relaxed release caps.** The released ``numpy`` and ``scipy`` ranges
  now admit numpy 2.5 and scipy 1.18 (``numpy<2.6.0``, ``scipy<1.19.0``);
  the ``==`` pins stay on the newest Python 3.11-compatible releases. The
  calendar-versioned ``pymatgen`` is capped at the next year rather than
  the next month.
- Lower bounds raised to match ``structuretoolkit`` 0.0.48
  (``ase>=3.24.0``, ``numpy>=1.23.5``) and ``pyscal3>=4.1.0``; the
  ``matplotlib`` lower bound is now listed under its conda name
  ``matplotlib-base`` so the release script actually applies it.

### Added

- **Clean-install CI.** A workflow builds the wheel, installs it with
  freely resolved dependencies into a fresh environment and runs the test
  suite out of tree.

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
