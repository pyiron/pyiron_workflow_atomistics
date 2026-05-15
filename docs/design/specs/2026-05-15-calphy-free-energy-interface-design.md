# Calphy free-energy interface

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-15 |
| Repo | `pyiron/pyiron_workflow_atomistics` |
| Scope | `calphy` ≥1.5.6: `fe`, `ts`, `tscale`, `pscale`, `melting_temperature`, `alchemy`, `composition_scaling` |
| Out of scope (v1) | SLURM/SGE schedulers, calphy's `script_mode=False` Python-library path, user-supplied custom LAMMPS scripts, free-energy extraction from `phonopy`/`dynaphopy` (see Follow-ups) |

## Problem

`pyiron_workflow_atomistics` covers total-energy and finite-temperature force-derived quantities (`bulk.eos_volume_scan`, `physics/phonons/*`, defects, surfaces, grain boundaries) but has no free-energy workflow. The headline thermodynamic outputs — Helmholtz/Gibbs free energy at a state point, F(T) along an isobar, and melting temperature — are exactly the missing piece for closing on quantitative phase diagrams from MLIP/EAM potentials.

`calphy` (Menon, Lysogorskiy, Drautz; PRMaterials 5, 103801) is the canonical Python-driver implementation: Frenkel–Ladd for solids, Uhlenbeck–Ford for liquids, reversible scaling for T/P dependence, and an automated solid+liquid crossover for Tm. It is LAMMPS-driven, has a clean Pydantic-validated input model (`calphy.input.Calculation`), and exposes a routines layer (`calphy.routines`) that does the actual TI integration. The integration cost is not in the science — calphy owns that — but in the API translation: calphy's natural input is a YAML schema with `lattice`/`pair_style`/`pair_coeff` strings, while this repo speaks `ase.Atoms` + dataclasses + the `Engine` Protocol.

What needs design: (a) how the user supplies the structure, potential, and LAMMPS binary in a way that matches this repo's idioms without forcing calphy through the `Engine` Protocol it doesn't fit; (b) the result dataclass that callers see; (c) the dependency boundary so users who don't need free energies aren't forced to install calphy + LAMMPS.

## Approach

A new `pyiron_workflow_atomistics.physics.free_energy` subpackage with six `@pwf.as_function_node` public nodes — one per calphy mode — that each build a `calphy.input.Calculation`, route into the user's `working_directory`, dispatch through a single shared `_run_calphy_job` helper, and return a typed `FreeEnergyOutput` dataclass.

`calphy` and `pyiron_workflow_lammps` ship as an **optional extra** (`pip install pyiron_workflow_atomistics[free-energy]`), behind lazy imports inside the public node bodies. Tier-1 unit tests run without the extra; Tier-2 integration tests are gated on `pytest.importorskip("calphy")` and a `lmp` binary on `PATH`.

Calphy does **not** fit the `Engine` Protocol — it doesn't return forces, and its TI machinery is hardwired in `calphy/phase.py` / `solid.py` / `liquid.py`. The package's existing `Engine` abstraction stays untouched; calphy gets its own subpackage parallel to `physics/phonons/`.

Five deliberate API decisions, each motivated below:

- **`LammpsEngine` is accepted but only `engine.command` is read.** Every other non-default field is rejected with a `ValueError` listing offenders. Rationale: calphy generates its own LAMMPS input from `pair_style`/`pair_coeff`/`potential_file`; the engine's `path_to_model`, `input_script_*`, `EngineInput`, and especially `raw_script` would silently have no effect, which is worse than being rejected. The engine still earns its place because it carries the LAMMPS launcher (`mpirun -np N lmp`) that the rest of the user's workflow already encodes — they don't re-type it.

- **Potential is a separate `LammpsPotential` dataclass.** It maps 1:1 to calphy's `pair_style` / `pair_coeff` / `potential_file` fields. Strings are passed verbatim; we don't parse or rewrite them. Surfacing the potential on the engine instead would create two write sites for `pair_style` (engine vs adapter), with the engine field silently losing.

- **Structure is `ase.Atoms` only.** Internally written as a LAMMPS data file in the simfolder; element list comes from `dict.fromkeys(atoms.get_chemical_symbols())` (same ordering rule the existing `LammpsEngine` uses); masses are looked up via `ase.data.atomic_masses`. Calphy's native `lattice='fcc'` shortcut is not exposed — consistency with the rest of the package outweighs the convenience.

- **One function-node per mode + one shared runner.** Public surface: `free_energy`, `reversible_scaling_temperature`, `reversible_scaling_pressure`, `melting_temperature`, `alchemy`, `composition_scaling`. Each has a typed signature with the kwargs that mode actually requires (e.g., `temperature_range: tuple[float, float]` for `reversible_scaling_temperature`, `output_chemical_composition: dict[str, int]` for `composition_scaling`). Internals factor through `_build_calphy_calculation(...)` and `_run_calphy_job(calc)` so adding a mode is one new public node + one row of dispatch.

- **`script_mode=True` is hardcoded.** It's the only mode that takes `lammps_executable` + `mpi_executable` (which we derive from `engine.command`) and runs LAMMPS via subprocess — matching how the engine itself runs LAMMPS, keeping logs on disk, and avoiding a runtime dependency on the in-process `lammps` Python library.

## Components

```
pyiron_workflow_atomistics/
└── physics/
    └── free_energy/
        ├── __init__.py            # public re-exports
        ├── inputs.py              # LammpsPotential dataclass
        ├── outputs.py             # FreeEnergyOutput dataclass
        ├── _calphy_adapter.py     # engine validation, command parsing, _build_calphy_calculation, _run_calphy_job, _load_rs_curve
        └── calphy.py              # six public @pwf.as_function_node entry points
```

### `inputs.py`

```python
@dataclass
class LammpsPotential:
    """LAMMPS interatomic potential, passed verbatim to calphy.

    Attributes
    ----------
    pair_style
        LAMMPS pair_style line, e.g. ``"eam/alloy"``, ``"pace"``, ``"grace"``.
    pair_coeff
        LAMMPS pair_coeff line, e.g. ``"* * /path/to/Cu01.eam.alloy Cu"``.
        Element ordering must match the structure's chemical-symbol
        first-occurrence order.
    potential_file
        Optional auxiliary potential file path (some potentials require one);
        passed to ``calphy.input.Calculation.potential_file``.
    """
    pair_style: str
    pair_coeff: str
    potential_file: str | None = None
```

### `outputs.py`

```python
@dataclass
class FreeEnergyOutput:
    """Result of one calphy free-energy calculation.

    Units
    -----
    free_energy / free_energy_error / einstein_free_energy: eV/atom (calphy native).
    temperature: K. pressure: bar (calphy native — differs from CalcInputMD.pressure
    which is Pa; do not mix).
    """

    mode: Literal["fe", "ts", "tscale", "pscale", "melting_temperature",
                  "alchemy", "composition_scaling"]
    reference_phase: Literal["solid", "liquid", "both"]   # "both" for melting_temperature
    free_energy: float
    free_energy_error: float
    temperature: float
    pressure: float
    n_atoms: int
    elements: list[str]                                   # calphy element-order
    simfolder: str                                        # absolute path
    report: dict[str, Any]                                # calphy's report.yaml content, verbatim

    # mode-specific; None when not applicable
    temperature_array: np.ndarray | None = None           # ts / tscale
    free_energy_array: np.ndarray | None = None           # ts / tscale (matches temperature_array)
    pressure_array: np.ndarray | None = None              # pscale
    melting_temperature: float | None = None              # melting_temperature
    melting_temperature_error: float | None = None
    composition_path: list[dict[str, int]] | None = None  # composition_scaling
    einstein_free_energy: float | None = None             # fe (solid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

The dataclass is required to be picklable (no `Phase` references, no LAMMPS handles) so pyiron_workflow can checkpoint outputs — matches `EngineOutput` / `PhononOutput` precedent.

### `_calphy_adapter.py`

Five helpers, all private:

- `_validate_engine_only_command(engine: LammpsEngine) -> None` — iterates `dataclasses.fields(LammpsEngine)`, skips the carve-out set (`{"EngineInput", "mode", "working_directory", "command"}`), compares each remaining field to its dataclass default (handling `default_factory`), and raises `ValueError` listing every non-default field with an actionable suggestion.
- `_split_lammps_command(cmd: str) -> tuple[str, str | None, int]` — returns `(lammps_executable, mpi_executable, cores)`. Detects `mpirun`/`mpiexec`/`srun` launchers, extracts `-np N` / `-n N`, strips trailing `-in <file>` / `-log <file>` (calphy provides its own), raises on unrecognised tokens.
- `_build_calphy_calculation(mode, structure, potential, lammps_engine, working_directory, **mode_kwargs) -> calphy.input.Calculation` — constructs the Pydantic `Calculation` model. Writes the Atoms to `simfolder/lammps.data` as `format='lammps-data'`; sets `lattice` to that data-file path; sets `file_format='lammps-data'`; sets element/mass arrays; sets `script_mode=True` with the parsed launcher; sets `npt`, `equilibration_control`, step counts, and per-mode fields (`temperature`, `pressure`, `melting_temperature.step`, etc.).
- `_run_calphy_job(calc: calphy.input.Calculation) -> tuple[Phase, dict]` — calls `calphy.kernel.setup_calculation(calc)` then `run_calculation(job)`; reads `job.simfolder/report.yaml` back as a dict (more robust than scraping live attributes across calphy versions); returns the `Phase` (or `MeltingTemp`) and the report dict.
- `_load_rs_curve(simfolder: str) -> tuple[np.ndarray, np.ndarray]` — parses calphy's `temp_comp_*` files written during reversible scaling, returns `(temperature_array, free_energy_array)`. Defined for `ts`/`tscale`; raises `FileNotFoundError` if invoked on a non-RS run.

### `calphy.py` (public nodes)

All six follow the same template:

```python
@pwf.as_function_node("free_energy_output")
def free_energy(
    *,
    structure: Atoms,
    lammps_engine: LammpsEngine,
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
    _require_calphy()
    _validate_engine_only_command(lammps_engine)
    _validate_structure(structure)

    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(simfolder)
        calc = _build_calphy_calculation(
            mode="fe", structure=structure, potential=potential,
            lammps_engine=lammps_engine, working_directory=simfolder,
            temperature=temperature, pressure=pressure,
            reference_phase=reference_phase,
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations, npt=npt,
            equilibration_control=equilibration_control,
        )
        job, report = _run_calphy_job(calc)
        return _pack_free_energy_output(mode="fe", job=job, report=report,
                                        simfolder=simfolder, structure=structure)
    finally:
        os.chdir(prev_cwd)
```

The other five nodes (`reversible_scaling_temperature`, `reversible_scaling_pressure`, `melting_temperature`, `alchemy`, `composition_scaling`) follow the same body with different signatures and `mode=` arguments:

| Node | Mode | Distinctive kwargs |
|---|---|---|
| `free_energy` | `fe` | `temperature` (scalar), `reference_phase` |
| `reversible_scaling_temperature` | `ts` | `temperature_range: tuple[float, float]`, `reference_phase` |
| `reversible_scaling_pressure` | `pscale` | `temperature`, `pressure_range: tuple[float, float]`, `reference_phase` |
| `melting_temperature` | `melting_temperature` | `temperature_guess: float \| None` (routes to `Calculation.temperature`; if `None`, calphy infers from `mendeleev`), `step: int = 200` (→ `Calculation.melting_temperature.step`), `max_attempts: int = 5` (→ `Calculation.melting_temperature.attempts`); no `reference_phase` kwarg (uses both) |
| `alchemy` | `alchemy` | `temperature`, `pair_style_target: str`, `pair_coeff_target: str` |
| `composition_scaling` | `composition_scaling` | `temperature`, `output_chemical_composition: dict[str, int]` |

`tscale` is intentionally not a separate node — calphy docs treat it as a near-duplicate of `ts`. If a future user needs it, surface it as a `method='ts'|'tscale'` kwarg on `reversible_scaling_temperature` (no new node).

`__init__.py` re-exports `FreeEnergyOutput`, `LammpsPotential`, and the six public nodes. All `calphy` and `pyiron_workflow_lammps` imports happen inside node bodies (or `_require_calphy()` / `_require_lammps_engine()` guards at the top of each node body) so importing the subpackage doesn't fail without the extra.

## Failure modes and recovery

| Failure | Detection point | Behavior |
|---|---|---|
| `calphy` not installed | `_require_calphy()` at node entry | `ModuleNotFoundError("calphy not installed. pip install 'pyiron_workflow_atomistics[free-energy]'")` |
| `pyiron_workflow_lammps` not installed | `_require_lammps_engine()` at node entry | Same idiom |
| `LammpsEngine` has non-default fields beyond `command` | `_validate_engine_only_command` | `ValueError` listing offenders + a one-line minimal-engine example |
| `engine.command` contains tokens beyond launcher + binary + `-in`/`-log` | `_split_lammps_command` | `ValueError("Unrecognized tokens in LammpsEngine.command: ...")` |
| `structure` empty / non-3D / mixed PBC | `_validate_structure` at node entry | `ValueError` with the specific failure |
| `reference_phase="liquid"` for a crystalline structure (or vice versa) | calphy's `tolerance.solid_fraction` / `liquid_fraction` mid-run | Propagate calphy's exception; `simfolder` retains artefacts for triage |
| Mode-specific bad kwarg shape (e.g., scalar passed as `temperature_range`) | Node-entry validation, before `_build_calphy_calculation` | `ValueError` with the expected type |
| calphy raises mid-run | Not caught | Propagate; cwd is restored by `try/finally` |
| `melting_temperature` exhausts `max_attempts` | calphy raises `MeltingTempError` (see `calphy/errors.py`) | Propagate; no partial `FreeEnergyOutput` produced |
| `report.yaml` parse failure | `_run_calphy_job` | Re-raise with the simfolder path appended so the user can inspect the file |

What we don't do: no retries (LAMMPS runs are expensive and the user owns the launcher config); no partial-result salvaging on mid-run failures (the free-energy value is meaningless without the matching reference); no `pressure_in_pa` auto-detection (the 1e5 factor would false-positive on legitimate low-pressure runs — docstrings call out the convention explicitly).

## Testing

Three tiers.

**Tier 1 — Cheap unit tests (no calphy/LAMMPS needed; always run):**

- `test_missing_calphy_raises_actionable` — patch `sys.modules["calphy"] = None`; assert the public node raises `ModuleNotFoundError` with the exact `pip install` line.
- `test_missing_pyiron_workflow_lammps_raises_actionable` — same idiom for the engine import.
- `test_engine_with_only_command_passes` — `LammpsEngine(EngineInput=CalcInputStatic(), command="lmp -in in.lmp -log log.lammps")` passes `_validate_engine_only_command` cleanly.
- `test_engine_with_nondefault_field_rejected` — set `engine.raw_script="..."`; then a fresh engine with `engine.path_to_model="/real/path"`; then `engine.input_script_pair_style="eam/alloy"`; assert each raises `ValueError` and the message lists the offending field name.
- `test_engine_command_parsing` — table-driven: `"lmp"` → `(lmp, None, 1)`; `"mpirun -np 4 lmp"` → `(lmp, "mpirun", 4)`; `"srun -n 8 lmp -in foo -log bar"` → `(lmp, "srun", 8)`; `"mpirun --bind-to none -np 2 lmp"` → `(lmp, "mpirun --bind-to none", 2)`. Bad input: `"mpirun -np 4 lmp -unknown-flag x"` raises.
- `test_free_energy_output_dataclass_shape` — required vs optional fields, `to_dict()` round-trips, dataclass picklable.
- `test_mode_kwarg_validation` — `reversible_scaling_temperature(temperature_range=300.0, ...)` raises (must be 2-tuple); `melting_temperature(temperature_guess=-100, ...)` raises.
- `test_structure_validation` — empty Atoms, mixed PBC, non-3D cell — each raises.
- `test_lammps_data_file_matches_structure` — write a known Atoms via the adapter, parse the resulting data file with `ase.io.read(format='lammps-data')`, assert round-trip equality on positions, cell, and atomic types.

**Tier 2 — Integration tests (gated on `pytest.importorskip("calphy")` AND a `lmp` binary on `PATH`):**

- `test_free_energy_fcc_cu_smoke` — Cu FCC, EAM (calphy's `examples/potentials/Cu01.eam.alloy` pinned as a test fixture), `temperature=100`, `reference_phase="solid"`, `n_equilibration_steps=2000`, `n_switching_steps=2000`. Assert `out.free_energy` finite and within a hard-coded ±0.01 eV/atom band of a pinned reference. Budget: ~30 s.
- `test_reversible_scaling_temperature_returns_curve` — same potential, `temperature_range=(100, 300)`. Assert `out.temperature_array.shape == out.free_energy_array.shape`, temperature monotonic, free energies finite.
- `test_melting_temperature_runs` — same potential, low-step budget, `temperature_guess=1300`. Assert `out.melting_temperature` finite and within wide bounds (e.g., 800–2000 K). Marked `slow`.
- `test_alchemy_smoke` and `test_composition_scaling_smoke` — single thin smoke each, assert run completes and `FreeEnergyOutput.free_energy` finite. Marked `slow`.
- `test_working_directory_is_honored` — pass `working_directory=tmp_path`, assert `tmp_path/free_energy/report.yaml`, `tmp_path/free_energy/in.lmp`, `tmp_path/free_energy/log.lammps` exist after the node returns; confirm cwd restored.
- `test_pscale_smoke` — `reversible_scaling_pressure` with `pressure_range=(0, 10_000)` (bar), assert `out.pressure_array` populated.

**Tier 3 — Determinism / regression:**

- `test_calphy_simfolder_path_is_absolute` — `FreeEnergyOutput.simfolder` is absolute so users can find artefacts after cwd changes.
- `test_cwd_restored_on_calphy_exception` — monkeypatch `_run_calphy_job` to raise; assert `os.getcwd()` unchanged after the node propagates the exception.

**What we don't test:**
- Numerical accuracy against published Tm values — that's notebook validation, not pytest.
- Multi-node SLURM/SGE paths — out of scope for v1.
- LAMMPS-version-specific output parsing — calphy owns that.

## Follow-ups (explicitly v2 or later)

- **Free-energy extraction from `phonopy` / `dynaphopy`.** Quasi-harmonic Helmholtz free energy comes for free out of `phonopy.Phonopy.run_thermal_properties` (we already have the FC2 step in `physics/phonons/harmonic.py`); anharmonic post-MD free energy is available from `dynaphopy` (used in `physics/phonons/md_renormalised.py`). Both should be surfaced as additional public nodes in this same `physics/free_energy/` subpackage, returning the same `FreeEnergyOutput` dataclass, so callers can compare TI/phonon/QHA free energies through one interface. This is the natural home for them — the dataclass is shared and the methodology comparison is the whole point.
- **`tscale` exposed as `method='ts'|'tscale'`** on `reversible_scaling_temperature` — add only when a concrete user asks; the integration math differs subtly enough that aliasing would be misleading.
- **Multi-launcher support** for SLURM/SGE — calphy already supports it; the v1 cut deferred it because pyiron_workflow has its own submission story and mixing the two creates ambiguity. Either expose a `CalphyQueue` dataclass, or wait for pyiron_workflow's submission to standardise and tunnel through that.
- **Pickle-stable `Phase` handle** — currently `FreeEnergyOutput` deliberately drops the `Phase` reference. If users start needing to re-derive things from it (e.g., recomputing entropy from a stored work distribution), revisit.
- **`composition_scaling` as a macro** rather than a function-node — once a real multi-leg use case lands, break the per-leg setup into a fan-out macro so each leg can run as a subengine subdir.
- **Auto-detection of `pair_style` element-ordering mismatch** vs `structure` — calphy itself already validates element-order consistency between `element` and `pair_coeff` in its `model_validator`, but it does so against the structure we pass it, not the structure the user reasoned about. Add a structure-vs-`pair_coeff` element-set/ordering check at the adapter level, raising before calphy gets the chance.
