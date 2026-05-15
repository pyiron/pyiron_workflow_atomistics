# Free-energy consolidation (harmonic / quasiharmonic / anharmonic)

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-15 |
| Repo | `pyiron/pyiron_workflow_atomistics` |
| Branch | `tests/improve-coverage` |
| Scope | New free-energy entry points wrapping `phonopy` (harmonic + QHA), `dynaphopy` (renormalised harmonic at one T, plus a T-grid sweep), and the existing calphy nodes — all reachable from `physics.free_energy` |
| Out of scope (v1) | Liquid free energies via phonons; non-analytic correction (BORN + ε∞); full ⟨∂H/∂λ⟩ dynaphopy TI; bespoke physical relocation of `phonons/*` modules; refactoring the κ(T) workflow |

## Problem

`physics.free_energy` is currently a calphy-only subpackage: six public nodes producing `FreeEnergyOutput` for `fe`, `ts`, `pscale`, `melting_temperature`, `alchemy`, `composition_scaling`. It is therefore the *home* of free energy in this repo, but the repo also already contains:

- A fully-working phonopy FC2 force-evaluation pipeline (`physics/phonons/harmonic.py`) whose `_compute_harmonic_observables` helper already returns F(T), S(T), Cv(T) at fixed reference volume. Today that data is **hidden** — accessible only when a user passes `harmonic_observables=True` into `calculate_phonon_thermal_conductivity`, the κ(T) macro. There is no user-facing "harmonic free energy" node.
- A dynaphopy MD-projection workflow (`physics/phonons/md_renormalised.py`) producing renormalised phonon frequencies and linewidths at finite T. Useful spectroscopically, but does **not** currently produce a free energy.
- A reusable EOS volume sweep + EOS fit (`physics/bulk.py:eos_volume_scan`). Suitable as the volume-grid backbone for QHA, but currently used only for lattice-parameter optimisation.
- No quasiharmonic / QHA support anywhere.

What needs design: (a) the public node surface that turns those three packages into a coherent free-energy story for solids; (b) where the new nodes live without disturbing the κ(T) workflow; (c) how the existing calphy-shaped `FreeEnergyOutput` extends to accommodate harmonic, QHA, and dynaphopy outputs; (d) the demo + tests that make the picture believable end-to-end.

The user-facing story we are scoping for: "compute the Helmholtz/Gibbs free energy of a solid at finite T, by your choice of level — harmonic, quasiharmonic, or anharmonic (dynaphopy at one T, dynaphopy over a T grid, or calphy TI)."

## Approach

A consolidation by re-export, not relocation. Four new public macros land under `physics.free_energy`, each wrapping helpers that already exist in `physics.phonons` and `physics.bulk` via thin import-and-wrap glue. `physics.phonons.{harmonic,anharmonic,md_renormalised}` are not moved — they remain the canonical home of FC2/FC3 plumbing and dynaphopy MD projection, since the κ(T) workflow continues to import from them.

```
pyiron_workflow_atomistics/physics/
├── bulk.py                                 # unchanged; eos_volume_scan reused by QHA
├── phonons/                                # unchanged in structure
│   ├── harmonic.py                         #   no behaviour changes
│   ├── anharmonic.py                       #   no behaviour changes
│   └── md_renormalised.py                  #   no behaviour changes
└── free_energy/
    ├── __init__.py                         # extended export list
    ├── _compat.py                          # unchanged
    ├── inputs.py                           # unchanged
    ├── outputs.py                          # FreeEnergyOutput extended (see § FreeEnergyOutput)
    ├── calphy.py                           # unchanged
    ├── _calphy_adapter.py                  # unchanged
    ├── harmonic.py                         # NEW
    ├── quasiharmonic.py                    # NEW
    └── anharmonic_dynaphopy.py             # NEW (single-T + TDI-over-T)
```

Four deliberate API-shape decisions, each motivated below:

- **Separate nodes per method, no dispatcher.** Matches the existing per-mode calphy layout (`free_energy / reversible_scaling_temperature / ...`). A dispatcher was considered and rejected: each method's required kwargs are sufficiently different (`temperatures` vs `temperature_range`, `strain_range` vs none, `q_mesh` vs none) that a `method="..."` enum would either silently accept wrong kwargs or duplicate every signature in a typing-overload tower. The user is expected to import the specific node they want.

- **Import-and-wrap, no physical relocation.** `physics.phonons` remains the home of FC2/FC3 plumbing because `calculate_phonon_thermal_conductivity` needs it there. The new `free_energy/harmonic.py` imports `_generate_fc2_supercells`, `_evaluate_supercells`, `_compute_harmonic_observables`, `_ase_to_phonopy`, `_phonopy_to_ase` from `physics.phonons.harmonic` and treats them as the consolidated FC2 building blocks. This keeps the design diff small and the κ(T) workflow untouched.

- **One `FreeEnergyOutput` dataclass, extended.** Every new mode returns the same `FreeEnergyOutput` type with method-specific arrays populated and the rest `None`. Rationale: keeps the import surface flat (one type to learn), and `to_dict()` continues to work as a homogeneous serialiser across calphy + harmonic + QHA + dynaphopy. Per-method typing is preserved via the discriminator field `mode`, which extends to `"harmonic" | "qha" | "anharmonic_dynaphopy" | "anharmonic_dynaphopy_tdi"`.

- **Dynaphopy TDI is renormalised-harmonic over a T grid, not full ⟨∂H/∂λ⟩ TI.** Each T in the grid runs an independent NVT trajectory through `calculate_phonon_md_renormalisation`, then F_anharm(T) is computed as the harmonic free energy on the renormalised spectrum. This is the simpler, well-scoped formulation. Full λ-integration is documented as a v2 follow-up; it requires MD plumbing dynaphopy does not directly expose and is out of scope for this round.

## Components

### `harmonic.py` — `harmonic_free_energy`

```python
@pwf.api.as_macro_node("free_energy_output")
def harmonic_free_energy(
    wf,
    *,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,                    # int / (3,) / (3,3)
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0),  # K
    displacement_distance: float = 0.03,     # Å
    is_plusminus: str | bool = "auto",
    working_directory: str = ".",
    subdir: str = "harmonic_free_energy",
    keep_handles: bool = False,
) -> FreeEnergyOutput: ...
```

Returns F(T), S(T), Cv(T) at fixed reference volume on the supplied temperature grid. `mode="harmonic"`, `reference_phase="solid"`, scalar `free_energy` = `free_energy_array[0]` (lowest T) for convenience.

Internal graph:

```
structure ──┐
            ├──▶ _generate_fc2_supercells ──▶ list[Atoms]
fc2_sc_mat ─┘                                  │
                                                ▼
                                  _evaluate_supercells(engine, prefix="fc2_disp_")
                                                │
                                                ▼
                              _produce_fc2_view  (NEW small helper)
                                                │
                                                ▼
                _compute_harmonic_observables (T-grid)  ──▶ (band, dos, free_energy_dict)
                                                │
                                                ▼
                                        _pack_harmonic_output
                                                │
                                                ▼
                                        FreeEnergyOutput
```

`_produce_fc2_view` builds a bare `phonopy.Phonopy` (no phono3py) from the FC2 supercell matrix and stacked forces, calls `produce_force_constants()`, and returns the `Phonopy` instance. Lives in `free_energy/harmonic.py`; ~10 LOC.

`_pack_harmonic_output` writes:
- `mode="harmonic"`, `reference_phase="solid"`
- `temperature = temperatures[0]` (lowest T)
- `pressure = 0.0` (harmonic FE is at fixed reference V; pressure is not a control variable here)
- `free_energy = F[0]` (eV/atom)
- `free_energy_error = 0.0` (no statistical error in deterministic FD)
- `n_atoms`, `elements`, `simfolder`, `report = {"method": "harmonic", "fc2_supercell_matrix": ..., "displacement_distance": ...}`
- `temperature_array`, `free_energy_array`, `entropy_array`, `heat_capacity_array`
- When `keep_handles=True`: `phonopy_handle`, `band_structure`, `phonon_dos`

**Engine constraint:** any `Engine` Protocol implementation.

**Error handling:**
- `_check_all_converged(fc2_engine_outputs, label="FC2")` (reused from `phonons/anharmonic.py`) raises a `RuntimeError` listing failed-supercell working directories.
- FC2 force / supercell count mismatch raises `RuntimeError("FC2 force/supercell mismatch: ...")` — same message family as `_run_phono3py_thermal_conductivity` (a documented invariant that displacement kwargs not drift between generation and synthesis).

### `quasiharmonic.py` — `quasiharmonic_free_energy`

```python
@pwf.api.as_macro_node("free_energy_output")
def quasiharmonic_free_energy(
    wf,
    *,
    structure: Atoms,                        # equilibrium reference cell
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0),
    pressure: float = 0.0,                   # GPa (Gibbs branch); 0 → Helmholtz
    strain_range: tuple[float, float] = (-0.03, 0.03),
    num_volumes: int = 7,
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    eos_type: str = "vinet",                 # phonopy.qha.QHA: vinet | birch_murnaghan | murnaghan
    working_directory: str = ".",
    subdir: str = "quasiharmonic_free_energy",
    keep_handles: bool = False,
) -> FreeEnergyOutput: ...
```

Runs `harmonic_free_energy` at `num_volumes` isotropically-strained cells, then `phonopy.qha.QHA` to give G(T,P), V*(T,P), B(T,P), α(T,P). `mode="qha"`, `reference_phase="solid"`.

Internal graph:

```
structure ─▶ generate_structures(axes=["iso"], strain_range, num_points=num_volumes)
                          │
                          ▼
                  list[Atoms] (n_V strained cells)
                          │
                          ├── evaluate_structures(engine, prefix="vol_E_")
                          │           │   (one-shot static energy per volume)
                          │           ▼
                          │   energies_per_volume  (n_V,)
                          │   volumes              (n_V,)
                          │
                          └── for each strained cell:
                                  harmonic_free_energy(strained_cell, fc2_supercell_matrix,
                                                       temperatures, subdir=f"vol_{i:03d}/harmonic")
                                  → F_phonon_per_T(V_i)
                          │
                          ▼
                _fit_qha(energies, volumes, F_phonon[T,V], T-grid, P_GPa, eos_type)
                          │  uses phonopy.qha.QHA
                          ▼
                FreeEnergyOutput with QHA fields populated
```

`evaluate_structures` (from `physics/bulk.py`) is reused for the per-volume static energies. The per-volume `harmonic_free_energy` macros each create their own subdirectory `vol_{i:03d}_{strain:+.3f}/harmonic_free_energy/`.

`_fit_qha` is a function-node:

```python
@pwf.as_function_node("qha_results")
def _fit_qha(
    energies, volumes, free_energy_per_T_V, entropy_per_T_V, cv_per_T_V,
    temperatures, pressure_GPa, eos_type,
):
    from phonopy.qha import QHA
    qha = QHA(
        volumes=np.asarray(volumes),
        electronic_energies=np.asarray(energies),
        temperatures=np.asarray(temperatures),
        free_energy=np.asarray(free_energy_per_T_V),    # (n_T, n_V)
        cv=np.asarray(cv_per_T_V),
        entropy=np.asarray(entropy_per_T_V),
        eos=eos_type,
        pressure=pressure_GPa,                          # GPa native to phonopy.qha
    )
    qha.run()
    return {
        "equilibrium_volume_array": np.asarray(qha.get_volume_temperature()),
        "gibbs_free_energy_array":  np.asarray(qha.get_gibbs_temperature()),
        "bulk_modulus_array":       np.asarray(qha.get_bulk_modulus_temperature()),
        "thermal_expansion_array":  np.asarray(qha.get_thermal_expansion()),
        "qha_handle": qha,
    }
```

`_pack_qha_output` writes:
- `mode="qha"`, `reference_phase="solid"`
- `temperature = temperatures[0]`, `pressure = pressure_GPa` (in GPa for QHA results)
- `free_energy = gibbs_free_energy_array[0]`, `free_energy_error = 0.0`
- `temperature_array`, `free_energy_array = gibbs_free_energy_array`,
  `entropy_array`, `heat_capacity_array`, `volumes`, `free_energy_volume_array`,
  `equilibrium_volume_array`, `gibbs_free_energy_array`, `bulk_modulus_array`,
  `thermal_expansion_array`
- When `keep_handles=True`: `qha_handle`, plus the inner harmonic `phonopy_handle` from one representative volume

**Engine constraint:** any `Engine` Protocol implementation.

**Error handling:**
- `_check_qha_volume_range` (new helper in `free_energy/quasiharmonic.py`, called inside `_fit_qha`): if `qha.get_volume_temperature()` returns NaN for any T, raise `RuntimeError` with the message "QHA equilibrium volume undefined at T=... K — widen `strain_range` or extend it to include positive strain (current range: ..., current V grid: ...)."
- Energy-vs-volume curve sanity: if energies are non-monotonic in V on both sides of the minimum, warn but proceed (numerical noise is recoverable by `phonopy.qha`; structural issues are not, but we can't distinguish here).

### `anharmonic_dynaphopy.py` — `anharmonic_free_energy_dynaphopy` (single T)

```python
@pwf.api.as_macro_node("free_energy_output")
def anharmonic_free_energy_dynaphopy(
    wf,
    *,
    structure: Atoms,
    engine: Engine,                          # MUST expose .calculator (ASE engine)
    fc2_supercell_matrix,
    temperature: float,                      # K, single T
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,                  # fs
    thermostat_time_constant: float = 100.0, # fs
    seed: int | None = None,
    q_mesh=(11, 11, 11),                     # commensurate mesh for the F-sum
    phono3py_output=None,                    # optional FC2 reuse from κ(T) macro
    working_directory: str = ".",
    subdir: str = "anharmonic_free_energy_dynaphopy",
    keep_handles: bool = False,
) -> FreeEnergyOutput: ...
```

Internal graph:

```
structure ─┐
           ├──▶ calculate_phonon_md_renormalisation
engine ────┤        q_points = commensurate(q_mesh, primitive_cell)
fc2_sc_mat ┘        temperature, MD plumbing, fc2 source
                                  │
                                  ▼
                          MdPhononOutput  ──▶ ω_renorm at commensurate q
                                  │
                                  ▼
                _free_energy_from_spectrum(ω_renorm, q_weights, T, n_atoms_primitive)
                                  │
                                  ▼
                          FreeEnergyOutput (mode="anharmonic_dynaphopy")
```

`q_points = commensurate(q_mesh, primitive_cell)` is the commensurate-q grid for the supplied mesh — dynaphopy projects correctly only on q-points commensurate with the supercell. Computed via `phonopy.structure.grid_points.GridPoints` (or `phonopy.harmonic.dynamical_matrix.get_commensurate_points` — implementation chooses based on phonopy version available).

`_free_energy_from_spectrum` is a function-node (~30 LOC, numpy only):

```python
@pwf.as_function_node("free_energy_per_atom", "entropy_per_atom", "cv_per_atom")
def _free_energy_from_spectrum(
    frequencies: np.ndarray,         # (n_q, n_band) THz
    q_weights: np.ndarray,           # (n_q,) summing to 1
    temperature: float,              # K
    n_atoms_primitive: int,
):
    """Harmonic free energy formula on a discrete (q, band) frequency grid.

    F = sum_q w_q * sum_b [ ℏω_qb/2 + k_B·T·ln(1 − exp(−ℏω_qb / k_B·T)) ]

    Acoustic modes at Γ are zeroed by dynaphopy's _project_with_dynaphopy
    (it clamps positions[:3]=0 at q==0), so no extra masking is needed here.
    Modes with ω ≤ 0 (imaginary) raise ValueError — F is undefined for an
    unstable spectrum.
    """
```

`_pack_anharmonic_dynaphopy_output` writes:
- `mode="anharmonic_dynaphopy"`, `reference_phase="solid"`
- `temperature`, `pressure = 0.0`
- `free_energy`, `entropy`, `heat_capacity` (scalars at the supplied T)
- `harmonic_frequencies`, `renormalised_frequencies`, `linewidths`, `q_mesh`
- `report = {"method": "anharmonic_dynaphopy", "n_md_steps": ..., "md_temperature_mean": ..., "md_temperature_std": ..., "md_health": <check_md_health()>}`
- When `keep_handles=True`: `dynaphopy_handle`, `phonopy_handle`

**Engine constraint:** `engine.calculator` must exist (inherited from `calculate_phonon_md_renormalisation`).

**Error handling:**
- MD health (`MdPhononOutput.check_md_health()`) issues are surfaced into `report["md_health"]` and re-emitted as warnings at the free-energy node level.
- Any NaN frequencies (failed Lorentzian fits) → `RuntimeError("Lorentzian fit failed for {n}/{N} (q, band) pairs; F_anharm cannot be computed. Inspect MdPhononOutput.linewidths / .renormalised_frequencies to debug.")`.
- Imaginary modes (ω ≤ 0) → `ValueError("Spectrum has {n} imaginary modes; harmonic free energy is undefined for an unstable spectrum.")`.

### `anharmonic_dynaphopy.py` — `anharmonic_free_energy_dynaphopy_tdi` (T grid)

```python
@pwf.api.as_macro_node("free_energy_output")
def anharmonic_free_energy_dynaphopy_tdi(
    wf,
    *,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(100.0, 200.0, 300.0, 400.0, 500.0),
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,
    thermostat_time_constant: float = 100.0,
    seed: int | None = None,
    q_mesh=(11, 11, 11),
    working_directory: str = ".",
    subdir: str = "anharmonic_free_energy_dynaphopy_tdi",
    keep_handles: bool = False,
) -> FreeEnergyOutput: ...
```

Loops `anharmonic_free_energy_dynaphopy` over each T in `temperatures` (independent NVT runs per T). Each iteration uses `subdir=f"T_{i:03d}_{T:.1f}K"`. A synthesis node aggregates:

```python
@pwf.as_function_node("free_energy_output")
def _stack_tdi_outputs(
    per_T_outputs: list[FreeEnergyOutput],
    structure, temperatures,
) -> FreeEnergyOutput:
    """Aggregate independent-T dynaphopy free energies into a single output."""
```

The stacked output has:
- `mode="anharmonic_dynaphopy_tdi"`, `reference_phase="solid"`
- `temperature = temperatures[0]`, `pressure = 0.0`
- `free_energy = free_energy_array[0]`, `free_energy_error = 0.0` (single-trajectory; bootstrap uncertainty is a v2 follow-up)
- `temperature_array`, `free_energy_array` (stacked)
- `entropy_array`, `heat_capacity_array` — central finite differences on `free_energy_array`; endpoints use forward/backward differences; if `len(temperatures) < 3`, only fill at interior points and set `report["derivative_warning"] = True`.
- `renormalised_frequencies_per_T`, `linewidths_per_T` (shape `(n_T, n_q, n_band)`) — separate fields from the single-T `renormalised_frequencies` / `linewidths` to keep each field's shape contract stable.
- `report["per_T_md_health"] = [out.report["md_health"] for out in per_T_outputs]`.

**Engine constraint:** `engine.calculator` required (inherited).

### `outputs.py` — `FreeEnergyOutput` extension

The dataclass remains pickleable (plain types + numpy + None-default handles). Changes summarised below; everything else is unchanged.

**`mode` literal extends to:**
```python
Literal[
    # calphy (existing)
    "fe", "ts", "tscale", "pscale", "melting_temperature",
    "alchemy", "composition_scaling",
    # NEW
    "harmonic", "qha",
    "anharmonic_dynaphopy", "anharmonic_dynaphopy_tdi",
]
```

**New scalar fields:**

| Field | Type / unit | Populated by |
|---|---|---|
| `entropy: float \| None` | eV/K/atom | harmonic (T₀), anharmonic_dynaphopy |
| `heat_capacity: float \| None` | eV/K/atom | harmonic (T₀), anharmonic_dynaphopy |

**New array fields (paired with `temperature_array`):**

| Field | Shape / unit | Populated by |
|---|---|---|
| `entropy_array` | (n_T,) eV/K/atom | harmonic, qha, anharmonic_dynaphopy_tdi |
| `heat_capacity_array` | (n_T,) eV/K/atom | harmonic, qha, anharmonic_dynaphopy_tdi |

**QHA-specific fields:**

| Field | Shape / unit | Notes |
|---|---|---|
| `volumes` | (n_V,) Å³/atom | per-strain volumes |
| `free_energy_volume_array` | (n_T, n_V) eV/atom | F(T,V) before volume minimisation |
| `equilibrium_volume_array` | (n_T,) Å³/atom | V*(T) at the requested P |
| `gibbs_free_energy_array` | (n_T,) eV/atom | G(T,P) = min_V[F(T,V) + PV] |
| `bulk_modulus_array` | (n_T,) GPa | B(T) at V*(T) |
| `thermal_expansion_array` | (n_T,) 1/K | α(T) at the requested P |

**Dynaphopy-specific fields:**

| Field | Shape / unit | Notes |
|---|---|---|
| `harmonic_frequencies` | (n_q, n_band) THz | reference (pre-renormalisation) |
| `renormalised_frequencies` | (n_q, n_band) THz | dynaphopy, single T |
| `linewidths` | (n_q, n_band) THz FWHM | dynaphopy, single T |
| `renormalised_frequencies_per_T` | (n_T, n_q, n_band) THz | dynaphopy TDI |
| `linewidths_per_T` | (n_T, n_q, n_band) THz | dynaphopy TDI |
| `q_mesh` | tuple[int, int, int] | mesh actually used for the F-sum |

**`keep_handles=True` extras (non-pickleable, dropped by `to_dict()`):**

| Field | Type | Populated by |
|---|---|---|
| `phonopy_handle: Any \| None` | `phonopy.Phonopy` | harmonic, qha |
| `qha_handle: Any \| None` | `phonopy.qha.QHA` | qha |
| `dynaphopy_handle: Any \| None` | `dynaphopy.Quasiparticle` | anharmonic_dynaphopy |
| `band_structure: dict \| None` | — | harmonic |
| `phonon_dos: dict \| None` | — | harmonic, qha |

**Unit conventions reaffirmed:**
- Energies: eV/atom (calphy native).
- Temperatures: K.
- Pressure: bar for calphy modes (`fe`, `ts`, `pscale`, `melting_temperature`, `alchemy`, `composition_scaling`). **GPa** for QHA-output fields (`bulk_modulus_array`, the `pressure` kwarg in `quasiharmonic_free_energy`). The unit discrepancy is documented in `FreeEnergyOutput`'s class docstring with an explicit per-mode table.

**`to_dict()`** continues to return plain dicts. Handle fields (`phonopy_handle`, `qha_handle`, `dynaphopy_handle`) are **always** excluded from the returned dict — including when they are populated — because the underlying phonopy/dynaphopy objects are not generally pickleable. This is a small change to the existing implementation: the `asdict` call is replaced with a field-by-field copy that explicitly skips a known set of handle fields. The dataclass instance still carries those handles in memory for callers that asked for `keep_handles=True`.

### `__init__.py` — public exports

```python
# additions to the existing __all__
from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
    harmonic_free_energy,
)
from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
    quasiharmonic_free_energy,
)
from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
    anharmonic_free_energy_dynaphopy,
    anharmonic_free_energy_dynaphopy_tdi,
)

__all__ = [
    "FreeEnergyOutput",
    "LammpsPotential",
    # calphy (existing)
    "alchemy",
    "composition_scaling",
    "free_energy",
    "melting_temperature",
    "reversible_scaling_pressure",
    "reversible_scaling_temperature",
    # NEW
    "harmonic_free_energy",
    "quasiharmonic_free_energy",
    "anharmonic_free_energy_dynaphopy",
    "anharmonic_free_energy_dynaphopy_tdi",
]
```

## Data flow & engine constraints — summary

| Method | Macro | Engine constraint | Force calls per run |
|---|---|---|---|
| Harmonic | `harmonic_free_energy` | any `Engine` | n_fc2_supercells |
| Quasiharmonic | `quasiharmonic_free_energy` | any `Engine` | `num_volumes × (1 + n_fc2_supercells)` |
| Anharmonic (1 T) | `anharmonic_free_energy_dynaphopy` | needs `engine.calculator` | n_fc2 + `production_steps` MD calls |
| Anharmonic (T grid) | `anharmonic_free_energy_dynaphopy_tdi` | needs `engine.calculator` | `n_T × (n_fc2 + production_steps)` |
| Anharmonic (calphy TI) | `free_energy`, `reversible_scaling_temperature` | `LammpsEngine` only | calphy-controlled |

The engine constraint is validated at runtime inside the macro (`hasattr(engine, "calculator")` for dynaphopy paths) with a clear error message pointing at the existing `ASEEngine`/`LammpsEngine` docstrings.

## Demo notebook — `notebooks/free_energy_solid.ipynb`

Single notebook, one solid (fcc Al), one composite figure at the end overlaying all four free-energy curves. Sections:

1. **Setup.** Build `Al` (fcc, a₀≈4.05 Å). Two engine instances:
   - `ase_engine = ASEEngine(calculator=EMT())` — for harmonic / QHA / dynaphopy.
   - `lammps_engine = LammpsEngine(command=...)` + `LammpsPotential(pair_style="eam/alloy", pair_coeff="* * Al99.eam.alloy Al", potential_file="Al99.eam.alloy")` — for calphy. Cell guards on `[free-energy]` extras availability and a `lmp` binary on `PATH`; falls back to a skip notice if unavailable.
2. **Harmonic.** `harmonic_free_energy(structure=Al, engine=ase_engine, fc2_supercell_matrix=2*np.eye(3, dtype=int), temperatures=np.arange(0, 1001, 50))`. Plot F(T), S(T), Cv(T). Discussion: fixed-volume, zero-point energy included.
3. **Quasiharmonic.** `quasiharmonic_free_energy(..., temperatures=np.arange(0, 1001, 50), strain_range=(-0.03, 0.03), num_volumes=7)`. Plot G(T,P=0), V*(T), α(T), B(T). Compare F_harmonic vs G_QHA on one panel.
4. **Anharmonic — dynaphopy single T.** `anharmonic_free_energy_dynaphopy(..., temperature=300.0, production_steps=5_000, q_mesh=(7,7,7))`. Print scalar F_anharm at 300 K. Plot ω_harmonic vs ω_renorm along an auto band path.
5. **Anharmonic — dynaphopy TDI.** `anharmonic_free_energy_dynaphopy_tdi(..., temperatures=[200, 400, 600, 800], production_steps=5_000, q_mesh=(7,7,7))`. Plot F_anharm(T) on top of F_harmonic(T) and G_QHA(T).
6. **Anharmonic — calphy TI.** `free_energy(..., reference_phase="solid", temperature=300, ...)` for one point, then `reversible_scaling_temperature(..., temperature_range=(100, 800), ...)` for the curve. Overlay onto panel 5.
7. **Composite plot + discussion.** F(T) for all four methods on one axis. Discussion notes:
   - Harmonic underestimates F at high T (no V relaxation).
   - QHA captures thermal expansion → typically drops below harmonic at high T.
   - dynaphopy renormalised → captures soft-mode renormalisation, classical.
   - calphy TI → reference, full anharmonicity, classical, most expensive.

Runtime budget for default notebook cells: target ~5 min on a developer laptop with EMT. A `# %% TUTORIAL` comment block calls out which parameters to bump for production-grade results (longer MD, denser T grid, larger supercell, denser q_mesh).

## Tests

Layout matches `tests/{unit,integration}` convention.

**Unit tests — `tests/unit/physics/free_energy/`:**

- `test_harmonic_output_shape.py` — `_free_energy_from_spectrum` on a synthetic 1-band, 1-q Einstein spectrum reproduces the closed-form `F = ℏω/2 + k_BT·ln(1−exp(−ℏω/k_BT))` at multiple T (tolerance 1e-10).
- `test_freeenergyoutput_pickleable.py` — round-trip every new `mode` through `pickle.dumps`/`pickle.loads`; assert `to_dict()` is JSON-friendly (every value is `None` / `int` / `float` / `str` / `list` / nested-dict / `numpy.ndarray`).
- `test_qha_volume_range_guard.py` — feed `_fit_qha` a synthetic too-narrow volume grid (NaN V*(T) at high T) and assert `RuntimeError` with the actionable message fires.
- `test_tdi_derivatives.py` — `_stack_tdi_outputs` on an analytic Debye-model F(T) gives entropy / Cv within 1% over an interior region; endpoints flagged via `report["derivative_warning"]` when `n_T < 3`.
- `test_freeenergyoutput_to_dict_drops_handles.py` — `to_dict()` excludes `phonopy_handle`, `qha_handle`, `dynaphopy_handle` when populated.

**Integration tests — `tests/integration/physics/free_energy/`:**

- `test_harmonic_free_energy_emt.py` — Al/EMT, `fc2_supercell_matrix=2*np.eye(3, dtype=int)`, T=[0, 300]. Asserts F(0) > 0 (zero-point), monotonically-decreasing F(T), S(0) ≈ 0 within 1e-6.
- `test_quasiharmonic_free_energy_emt.py` — Al/EMT, 2×2×2, `num_volumes=5`, T=[0, 300]. Asserts V*(300) > V*(0) (positive thermal expansion), α(300) > 0, B(0) within ±20 % of the harmonic-FE V*0 cell's bulk modulus.
- `test_anharmonic_dynaphopy_free_energy_emt.py` — Al/EMT, 2×2×2, T=300, `production_steps=2000`, `q_mesh=(5,5,5)`. Asserts `|F_anharm − F_harmonic| < 50 meV/atom` at 300 K (sanity bound on Al/EMT, not physical accuracy).
- `test_anharmonic_dynaphopy_tdi_emt.py` — same setup, T=[200, 400]. Asserts the `(n_T,)` shape contract on all `*_array` fields and `(n_T, n_q, n_band)` on `renormalised_frequencies_per_T`.

**Gating:**
- All integration tests use the existing `@pytest.mark.integration` marker.
- Dynaphopy tests skip if `dynaphopy` is not importable.
- QHA tests skip if `phonopy.qha` is not importable.
- No new pytest markers introduced; calphy integration tests unchanged (already gated behind `[free-energy]` extras).

**CI runtime budget:** the four new integration tests aim for ≤3 min combined on the existing GitHub Actions runner — Al/EMT/small supercell/short MD.

## Engine reuse — what stays public-vs-private

The following helpers in `physics.phonons.{harmonic,anharmonic,md_renormalised}` are imported by the new `free_energy/` modules and are part of the de-facto consolidation surface:

- `phonons.harmonic._generate_fc2_supercells` — pwf function-node
- `phonons.harmonic._compute_harmonic_observables` — plain helper
- `phonons.harmonic._ase_to_phonopy`, `phonons.harmonic._phonopy_to_ase` — plain helpers
- `phonons.harmonic._normalise_supercell_matrix` — plain helper
- `phonons.anharmonic._evaluate_supercells` — pwf function-node (already shared)
- `phonons.anharmonic._check_all_converged`, `phonons.anharmonic._stack_forces` — plain helpers
- `phonons.md_renormalised.calculate_phonon_md_renormalisation` — pwf macro (public)

They keep their leading-underscore "private-ish" status. The design doc flags them as the shared internal API; the only stable public commitments are the four new top-level macros + the existing calphy entry points + `calculate_phonon_thermal_conductivity` + `calculate_phonon_md_renormalisation`.

## Follow-ups (v2)

- **Full ⟨∂H/∂λ⟩ dynaphopy TI.** Requires bespoke MD plumbing that emits the integrand at each step. Out of scope for v1 — flagged in `anharmonic_free_energy_dynaphopy_tdi` docstring with a pointer to this spec section.
- **Bootstrap uncertainty on dynaphopy TDI.** Multiple independent NVT seeds per T → standard error on F_anharm(T). Currently `free_energy_error=0.0` for dynaphopy modes.
- **Liquid-phase QHA / dynaphopy.** Out of scope — liquid free energies remain the calphy `reference_phase="liquid"` branch.
- **Non-analytic correction (BORN + ε∞).** Mirrors the v2 follow-up tracked in the phono3py and dynaphopy specs; same NAC plumbing would apply here.
- **Volume-by-volume QHA at finite anharmonic T.** Coupling dynaphopy renormalisation with QHA volume minimisation. Substantially heavier; flagged as v3.

## Decision log

| Decision | Choice | Alternative considered | Why this one |
|---|---|---|---|
| API layout | Separate node per method | Single dispatch macro | Matches existing per-mode calphy layout; avoids signature-merging that hides per-method required kwargs |
| Code relocation | Import-and-wrap | Physically move `phonons/{harmonic,md_renormalised}` into `free_energy/` | κ(T) workflow still needs FC2 plumbing in `phonons/`; relocation has high blast radius for no functional gain |
| Output type | Extend `FreeEnergyOutput` | One dataclass per method | Single import surface; `to_dict()` stays homogeneous; method discriminator via `mode` literal |
| QHA architecture | Compose `harmonic_free_energy` per volume | Fused macro that inlines FC2 + F + QHA | Independent testability; smaller code surface; matches the pwf macro idiom |
| Dynaphopy TDI scope | Renormalised-harmonic over T grid | Full ⟨∂H/∂λ⟩ TI | Well-scoped, scientifically honest; full TI needs MD plumbing dynaphopy doesn't expose |
| Pressure units in QHA | GPa | bar (matches calphy elsewhere) | QHA outputs are thermo quantities; bar is a LAMMPS pressure-control convention. Documented in the dataclass docstring with a per-mode table |
| Demo format | One overlay notebook | Three method-specific notebooks | Side-by-side comparison of all four methods is the main teaching value |
| Test gating | Per-package importorskip + existing `@pytest.mark.integration` | New pytest markers | No new markers needed; matches the calphy / dynaphopy / phonopy gating already in tests/ |
