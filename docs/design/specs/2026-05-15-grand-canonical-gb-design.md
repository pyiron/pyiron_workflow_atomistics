# Grand canonical grain-boundary optimization (GRIP integration)

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-15 |
| Repo | `pyiron/pyiron_workflow_atomistics` |
| Source | `https://github.com/enze-chen/grip` (fork: `https://github.com/ligerzero-ai/grip`), commit `8ff6a43` at time of writing |
| Reference | Chen, Heo, Wood, Asta, Frolov. *Nature Communications* **15**, 7049 (2024). DOI: 10.1038/s41467-024-51330-9 |
| Scope (v1) | The grand-canonical optimization (GCO) loop: in-plane translations, replication sampling, vacancy fraction on GB plane, atom perturbation, Voronoi-based interstitial swap, optional MD-at-T, minimize, energy-gated keep, periodic dedup |
| Out of scope (v1) | LAMMPS-specific dump output, GRIP's MPI/PBS launcher code, `plot_gco.py`, the "extra" aggressive dedup pass, numerical equivalence with upstream GRIP, multi-component composition sampling |

## Problem

`pyiron_workflow_atomistics` has total-energy GB workflows in `physics/grain_boundary.py` (`pure_gb_study`, `cleavage_study`, `segregation_study`, length optimization), but no facility for **sampling** the microscopic degrees of freedom of a grain boundary — in-plane translation, vacancy fraction on the GB plane, interstitial decoration — to discover the lowest-energy GB phase at a given misorientation. Without that sampling, a GB calculation is only as good as the single starting structure the user typed in.

GRIP (Chen & Frolov, 2024) is the established algorithm for this: a grand-canonical optimization (GCO) loop that repeatedly perturbs and re-relaxes a bicrystal while varying the atom count on the GB plane, keeping a deduplicated set of low-energy structures. It is the foundation of the `n`-vs-`Egb` diagrams in the Nature Communications paper.

GRIP ships as a script package (`main.py` + `params.yaml`) with its own ad-hoc abstractions:

- A `Calculator` ABC with three concrete implementations (`LAMMPSCalculator`, `ASECalculator`, `GenericCalculator`), each of which owns its own MD-vs-relax dispatch and a custom `CalculationResult` dataclass.
- An MPI/SLURM/PBS launcher in `core/simulation.py` that derives a process-rank `pid` from environment variables.
- A `Bicrystal` class that mixes structure manipulation, Voronoi interstitial finding, and GB-energy bookkeeping in ~640 lines.
- A YAML schema (`params.yaml`) read at startup; mode flags split between an `algo` block and a `struct` block.

This repo already has well-formed analogues for almost every one of those abstractions: the `Engine` Protocol (`engine/protocol.py`) and `EngineOutput` dataclass replace `Calculator` and `CalculationResult`; `for_node` replaces the MPI process-rank pattern; `@pwf.as_function_node` + dataclasses replace the YAML schema. Vendoring GRIP under its native abstractions would duplicate four mechanisms that already exist here.

What needs design: how to **port the GCO sampling algorithm** into the repo's idioms while preserving the algorithm's per-iteration adaptive character (running `E_min × e_mult` gate, periodic dedup), keeping the engine boundary clean enough that a user can swap EMT for MACE for VASP, and supporting GRIP's MD-then-minimize iterations without leaking MD-specific concerns into the minimize path.

## Approach

A new public `pyiron_workflow_atomistics.physics.grand_canonical_gb` module exposing two function-nodes — `gco_search` (the loop) and `build_bicrystal_slabs` (the convenience slab builder) — backed by a vendored `_grand_canonical_gb_code/` internal subpackage that holds the ported `Bicrystal`, `Interstitial`, sampling utilities, slab construction, dedup logic, and the `GCOConfig` dataclass.

GRIP's `Calculator` hierarchy is deleted entirely: the workflow calls `calculate(structure, engine)` against any pwa `Engine`. To support MD-then-minimize, `gco_search` accepts a required `minimize_engine` and an optional `md_engine`; per-iteration MD temperature and step count are injected by `dataclasses.replace`-ing the MD engine's `EngineInput` before each call. Parallelism across independent GCO searches comes from `for_node(gco_search, seeds=[...])` — the user composes it, the workflow stays sequential internally because the algorithm's `E_min` gate is inherently sequential.

The loop returns `(DataFrame, list[Atoms])` — never raises mid-iteration. Iteration failures (engine crashes, non-converged minimizes, Voronoi degeneracies) are logged and skipped. Storage is in-memory; engine `working_directory` per-iteration subfolders are the disk artifact and can be inspected post-hoc.

Five deliberate API decisions, each motivated below:

- **`minimize_engine` required, `md_engine` optional.** The algorithm always relaxes; MD-at-T is gated by `md_run_probability` and only happens when an MD engine is supplied. Asymmetric defaults match the asymmetric role in the algorithm and make the no-MD path the trivial default for EMT/fast-MLIP studies. Configuration mismatch (`md_run_probability > 0` with `md_engine is None`, or either engine carrying the wrong `EngineInput` subclass) is rejected at the workflow boundary, not silently downgraded.

- **Per-iteration MD parameter variation via `dataclasses.replace`.** GRIP's `sample_params()` chooses a temperature and step count per iteration and passes them to `calculator.run_md(...)`. pwa engines hold their `EngineInput` at construction, so we instead build a fresh per-iteration engine: `replace(md_engine, EngineInput=replace(md_engine.EngineInput, temperature=T, n_ionic_steps=n))`. This relies on the convention (already in force for `ASEEngine`) that engines are dataclasses with a dataclass `EngineInput`. Engines that don't follow this convention cannot run in MD-mode GCO; documented.

- **One `GCOConfig` dataclass for sampling parameters; engines separate.** All algorithmic knobs (`frac_min`, `frac_max`, `gb_thick`, `e_mult`, MD probability, temperature window, …) live on `GCOConfig`. Structure inputs (`lower_slab`, `upper_slab`, `e_cohesive`, `dlat`) are positional kwargs because they are per-call data, not configuration. Engines stay parameters because that is the repo convention.

- **`@pwf.as_function_node` for both public entry points.** `gco_search` and `build_bicrystal_slabs` are nodes so they compose into larger workflow graphs (e.g. a "screen GBs across misorientations" macro that calls `for_node(build_bicrystal_slabs, ...)` followed by `for_node(gco_search, ...)`). Internal helpers (sampling utilities, Bicrystal manipulation methods) are plain Python — they do not need to be nodes and the cost of being one is non-trivial when called inside a hot loop.

- **Storage in-memory; disk is the engine's job.** `gco_search` returns a pandas DataFrame of kept results and a parallel `list[Atoms]`. There is no `best/` folder owned by the workflow — engines already write per-iteration artifacts to their `working_directory`, and post-hoc tooling can serialize the DataFrame however the user wants. This deletes GRIP's bespoke "lammps_*" filename schema, the dump-format writer, and the on-disk `clear_best` pass that drives `unique.py`. Dedup runs on the DataFrame in memory.

## Components

```
pyiron_workflow_atomistics/
└── physics/
    ├── grand_canonical_gb.py          # gco_search + build_bicrystal_slabs (public)
    └── _grand_canonical_gb_code/
        ├── __init__.py
        ├── config.py                  # GCOConfig dataclass
        ├── bicrystal.py               # Bicrystal class (port of GRIP core/bicrystal.py)
        ├── interstitial.py            # Interstitial dataclass (port of GRIP core/interstitial.py)
        ├── sampling.py                # translation/replication/MD-param sampling
        ├── slabs.py                   # make_crystals + dlat (port of GRIP utils/utils.py)
        ├── energies.py                # gb_energy() pure formula
        └── store.py                   # DataFrame dedup (port of GRIP utils/unique.py)
tests/unit/physics/
└── test_grand_canonical_gb.py         # unit tests
tests/integration/
└── test_grand_canonical_gb_emt.py     # EMT end-to-end
```

### `config.py`

```python
@dataclass(frozen=True)
class GCOConfig:
    """All algorithmic knobs for one gco_search invocation.

    Field names align with GRIP's ``params.yaml`` where possible; only
    `MD_*` → `md_*`, `Tmin/Tmax` → `t_min/t_max`, `Emult` → `e_mult`,
    and `reps` → `reps_mode` change.
    """

    # Bicrystal geometry
    gb_thick: float = 10.0           # GB region thickness (Å) on either side
    pad: float = 10.0                # Extra padding (Å) for relaxation
    gb_gap: float = 0.3              # Initial slab separation (Å)
    vacuum: float = 1.0              # Vacuum padding above stitched cell

    # In-plane translation + replication sampling
    ngrid: int = 100                 # Grid points per dim used for y-binning
    size0: tuple[int, int, int] = (1, 1, 1)
    size:  tuple[int, int, int] = (2, 4, 15)
    reps_mode: int = 2               # 1=exact, 2=uniform, 3=exp-small, 4=exp-large

    # Vacancy fraction on GB plane
    frac_min: float = 0.0
    frac_max: float = 1.0

    # Perturbation
    perturb_u: float = 0.0           # Max Å perturbation, atoms above GB
    perturb_l: float = 0.0           # Max Å perturbation, atoms below GB

    # Voronoi interstitial swap
    inter_p: float = 0.0             # Probability of running the swap step
    inter_n: int = 0                 # Number of GB atoms to swap
    inter_t: float = 1.5             # Search thickness (Å) around GB plane
    inter_u: bool = False            # Restrict to unique interstitial sites
    inter_r: bool = True             # True = random choice; False = largest volume

    # MD (only consulted if md_engine is not None)
    md_run_probability: float = 0.0
    t_min: int = 300
    t_max: int = 1200
    md_min_steps: int = 5000
    md_max_steps: int = 500000
    md_step_sampling: str = "exponential"   # "exact" | "linear" | "exponential"

    # Filtering / dedup
    e_mult: float = 2.0              # Keep iff Egb < e_mult * running_min_Egb
    dedup_every: int = 50            # Iterations between dedup passes; 0 disables
```

### `grand_canonical_gb.py`

```python
@pwf.as_function_node("results", "best_structures")
def gco_search(
    minimize_engine: Engine,
    lower_slab: Atoms,
    upper_slab: Atoms,
    e_cohesive: float,
    config: GCOConfig = GCOConfig(),
    n_iters: int = 100,
    md_engine: Engine | None = None,
    seed: int = 0,
    dlat: float = 0.0,
) -> tuple[pd.DataFrame, list[Atoms]]:
    """Grand-canonical sampling of GB microstructures.

    Per iteration: build bicrystal → sample translation/replication →
    vacancies on GB plane → perturb → optional Voronoi interstitial
    swap → optional MD-at-T → minimize → compute Egb → conditional
    keep (running ``E_min × e_mult`` gate) → periodic dedup.

    Returns
    -------
    results : pandas.DataFrame
        Columns: ``Egb, n, dx, dy, rx, ry, T, n_md_steps, iter, converged``.
        One row per kept iteration. Empty DataFrame if nothing converged.
    best_structures : list[ase.Atoms]
        Same order as ``results`` rows.
    """


@pwf.as_function_node("lower_slab", "upper_slab", "dlat")
def build_bicrystal_slabs(
    crystal: str,
    symbol: str,
    a: float,
    upper_dirs: list[list[int]],
    lower_dirs: list[list[int]],
    c: float = 0.0,
    cutoff: float = 35.0,
) -> tuple[Atoms, Atoms, float]:
    """Build matched upper/lower slabs from a crystal type + tilt directions.

    Port of GRIP's ``make_crystals``; supports ``fcc``, ``bcc``, ``hcp``,
    ``dc``, ``sc``. ``c`` is required only for HCP. ``cutoff=0`` disables
    z-axis trimming (full slab kept).
    """
```

### `bicrystal.py`

Direct port of GRIP's `core/bicrystal.py`. Class shape preserved (`Bicrystal(lower, upper, struct_dict, algo_dict, dlat)` → kept; we adapt by passing `config.__dict__` derivatives in `gco_search`), methods unchanged in semantics:

- `copy_ul`, `shift_upper`, `replicate`, `get_bounds`
- `get_gbplane_atoms_u`, `make_vacancies_u`, `defect_upper`
- `perturb_atoms`, `join_gb`, `write_gb`
- `compute_voronoi`, `classify_sites`, `find_interstitials`
- `swap_gb_interstitials`, `find_and_swap_inters`

Three changes from upstream:

1. Drop the `lammps_data` write path inside `write_gb`; ASE's `ase.io.write` handles every format we need.
2. Replace prints with `pyiron_snippets.logger`-based logging.
3. Remove the `self.debug` branches — replaced with logger level.

### `interstitial.py`

Direct port of GRIP's `core/interstitial.py`. The class is a small dataclass holding `(x, y, z, label, nn, nnd)`; no behavioral changes.

### `sampling.py`

Pure functions, no class:

- `sample_xy_translation(slab, rng, ngrid) -> tuple[float, float]` — port of GRIP's `get_xy_translation`. The SLURM/PBS env-var and `pid`-based y-binning branches are deleted: upstream split the y-axis across MPI ranks so each rank explored its own band; here, every search samples the full y range and different seeds give different draws. `ngrid` is retained because future work may reintroduce stratified y-sampling at the workflow level (one search per band).
- `sample_xy_replications(rng, weights) -> tuple[int, int]` — port of `get_xy_replications`.
- `compute_weights(config) -> dict` — port of `compute_weights`.
- `sample_md_steps(config, rng) -> int` — port of `Simulation.sample_params`'s MD-step branch (modes `exact`/`linear`/`exponential`).
- `sample_md_temperature(config, rng) -> int` — multiples-of-100 K within `[t_min, t_max]`.

### `slabs.py`

Port of `make_crystals` and `compute_dhkl` from GRIP `utils/utils.py`. The function-node wrapper lives in `grand_canonical_gb.py` (`build_bicrystal_slabs`); this module holds the plain-Python implementation it calls into.

### `energies.py`

```python
def gb_energy(
    final_energy_ev: float,
    n_gb_atoms: int,
    gb_area_a2: float,
    e_cohesive_ev: float,
) -> float:
    """Grain-boundary energy in J/m².

        Egb = (E_total - n_gb_atoms × E_coh) / area × 16.021766

    Negative results indicate an unphysical configuration; we clamp to
    100 J/m² and emit a warning (matches GRIP's
    ``Calculator.get_gb_energy``).
    """
```

### `store.py`

```python
def dedup(rows: list[dict], atoms: list[Atoms]) -> tuple[list[dict], list[Atoms]]:
    """Deduplicate kept structures.

    Key: ``(round(Egb, 3), round(n, 3))``. Tiebreak: smallest ``rx*ry``
    (fewer atoms), then smallest ``dx**2 + dy**2`` (smaller in-plane shift).
    Port of GRIP's ``clear_best`` (non-extra path).
    """
```

The `extra=True` aggressive pass (which prunes ~50% of files) is intentionally not ported in v1.

## Per-iteration data flow

```
rng         = np.random.default_rng(seed)
best_Egb    = +inf
kept_rows, kept_atoms = [], []
weights     = compute_weights(config)

for i in range(n_iters):
    # ---- structure sampling ----
    bc = Bicrystal(lower_slab.copy(), upper_slab.copy(), config, dlat)
    dx, dy = sample_xy_translation(upper_slab, rng, config.ngrid, seed)
    bc.shift_upper(dx, dy)
    bc.get_bounds(config)
    rx, ry = sample_xy_replications(rng, weights)
    bc.replicate(rx, ry)
    bc.get_gbplane_atoms_u()
    bc.defect_upper(config, rng)
    bc.perturb_atoms(rng)
    bc.join_gb(config)
    bc.find_and_swap_inters(rng)
    atoms, bounds, n_frac = bc.gb, bc.bounds, bc.n

    # ---- optional MD ----
    T, n_md = 0, 0
    if md_engine is not None and rng.random() < config.md_run_probability:
        T = sample_md_temperature(config, rng)
        n_md = sample_md_steps(config, rng)
        md_e = dataclasses.replace(
            md_engine,
            EngineInput=dataclasses.replace(
                md_engine.EngineInput, temperature=T, n_ionic_steps=n_md,
            ),
        ).with_working_directory(f"iter_{i:05d}/md")
        try:
            out_md = calculate.node_function(atoms, md_e)
            atoms = out_md.final_structure
        except Exception as e:
            logger.warning("iter %d MD failed: %s", i, e)
            continue   # skip iteration entirely

    # ---- minimize ----
    min_e = minimize_engine.with_working_directory(f"iter_{i:05d}/min")
    try:
        out = calculate.node_function(atoms, min_e)
    except Exception as e:
        logger.warning("iter %d minimize failed: %s", i, e)
        continue
    if not out.converged:
        continue

    # ---- score + keep ----
    n_gb = _count_atoms_in_gb_region(out.final_structure, bounds)
    area = out.final_structure.cell[0, 0] * out.final_structure.cell[1, 1]
    Egb  = gb_energy(out.final_energy, n_gb, area, e_cohesive)
    if Egb < best_Egb * config.e_mult:
        if Egb < best_Egb:
            best_Egb = Egb
        kept_rows.append(dict(Egb=Egb, n=n_frac, dx=dx, dy=dy, rx=rx, ry=ry,
                              T=T, n_md_steps=n_md, iter=i, converged=True))
        kept_atoms.append(out.final_structure)

    # ---- periodic dedup ----
    if config.dedup_every and (i + 1) % config.dedup_every == 0:
        kept_rows, kept_atoms = dedup(kept_rows, kept_atoms)

return pd.DataFrame(kept_rows), kept_atoms
```

`_count_atoms_in_gb_region` is a private helper in `grand_canonical_gb.py` — direct port of GRIP `Simulation.get_gb_energy`'s `mask = (z >= z_min + lowerb - pad) & (z <= z_max - upperb + pad)`.

## Error handling

**Pre-loop validation** (raises `ValueError`):

| Condition | Reason |
|---|---|
| `n_iters <= 0` | No iterations would execute. |
| `not 0 <= config.frac_min <= config.frac_max <= 1` | Vacancy fraction bounds inverted. |
| `config.e_mult < 1.0` | Filter threshold below running min is unreachable. |
| `config.md_run_probability > 0 and md_engine is None` | Configuration says to run MD but no engine supplied. |
| `md_engine is not None and not isinstance(md_engine.EngineInput, CalcInputMD)` | MD engine has wrong input type. |
| `not isinstance(minimize_engine.EngineInput, CalcInputMinimize)` | Minimize engine has wrong input type. |

**Pre-loop warnings** (`logger.warning`, do not raise):

- `config.gb_thick < 5` (GB region thin)
- `config.pad < 3` (padding thin)
- `e_cohesive > 0` (sign-error catch — common mistake)

**Per-iteration recovery** — never abort the search:

| Failure | Handling |
|---|---|
| `calculate(...)` raises (MD or minimize) | Log; skip iteration; continue. |
| `out.converged is False` | Skip storing; continue. Counted in `summary` later. |
| `Egb < 0` | Clamp to 100 J/m²; warn. (Matches GRIP.) |
| Voronoi returns 0 sites with `inter_n > 0` | Warn once per search; skip swap; continue. |
| Fewer GB atoms than `inter_n` | Swap as many as possible (`min(inter_n, len(gb), len(sites))`); warn. |

**Return contract:** `gco_search` always returns `(DataFrame, list[Atoms])` and never raises after the pre-loop validation passes. Empty DataFrame is a valid output meaning "no iteration produced a converged, gate-passing result."

## Testing

### Unit (fast, no engine)

| Test | Pins |
|---|---|
| `test_gco_config_defaults` | Default values match GRIP's `params.yaml` baseline |
| `test_gco_config_validation` | `e_mult < 1`, inverted `frac_*`, MD config mismatch all raise |
| `test_bicrystal_translate_replicate_join` | shift_upper → replicate → join_gb produces expected atom count + cell |
| `test_bicrystal_defect_upper` | Vacancy count respects `frac_min/frac_max`; `n` recomputed correctly |
| `test_bicrystal_perturb_atoms` | Only atoms within `gb_thick/2` of GB plane move; seeded rng is deterministic |
| `test_voronoi_interstitial_finder` | Bulk FCC → octahedral + tetrahedral sites with correct NN counts |
| `test_sample_md_steps` | `exact`/`linear`/`exponential` modes produce samples in `[md_min_steps, md_max_steps]`; `exact` mode yields `md_min_steps` |
| `test_sample_xy_translation_deterministic` | Seeded rng → identical `(dx, dy)` across calls |
| `test_gb_energy_formula` | Formula matches; negative result clamps to 100 with warning |
| `test_dedup_tiebreak` | Same `(Egb, n)` → smaller `rx*ry` wins; then smaller `dx²+dy²` |
| `test_build_bicrystal_slabs_each_crystal` | fcc/bcc/hcp/dc/sc each produce non-empty slabs with sensible cells |

### Integration (EMT, ~1–2 min)

```python
def test_gco_search_emt_minimize_only(tmp_path):
    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc", symbol="Cu", a=3.6,
        upper_dirs=[[1,1,0],[0,0,1],[1,-1,0]],
        lower_dirs=[[1,-1,0],[0,0,1],[-1,-1,0]],
        cutoff=20.0,
    )
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.1, max_iterations=50),
        calculator=EMT(),
        working_directory=str(tmp_path / "min"),
    )
    df, atoms = gco_search.node_function(
        minimize_engine=engine, lower_slab=lower, upper_slab=upper, dlat=dlat,
        e_cohesive=-3.59,
        config=GCOConfig(
            frac_min=0.7, frac_max=1.0,
            ngrid=10, size=(1, 2, 5), size0=(1, 1, 1),
            md_run_probability=0.0, dedup_every=0,
        ),
        n_iters=5, seed=0,
    )
    assert len(df) > 0
    assert (df["Egb"] >= 0).all()


def test_gco_search_emt_with_md(tmp_path):
    # Same plumbing as above but with a second engine using CalcInputMD
    # (NVT, ~10 steps), md_run_probability=1.0, n_iters=2.
    ...
```

### Out of test scope (v1)

- Numerical equivalence with upstream GRIP. Different RNG paths, different optimizer; we port the algorithm, not the bit-output.
- LAMMPS-engine integration. Covered by `pyiron_workflow_lammps` once it ships MD support.
- Multi-seed `for_node` parallelism. Smoke-tested once with two seeds; deeper coverage belongs in `for_node`'s own tests.
- Performance / scaling benchmarks.

## Public API change

`pyiron_workflow_atomistics/physics/__init__.py` gains:

```python
from .grand_canonical_gb import gco_search, build_bicrystal_slabs, GCOConfig
```

No existing symbols move. The `physics/grain_boundary.py` workflows (`pure_gb_study`, etc.) stay where they are — the GCO loop is a separate concern, and the user mixes them by feeding `gco_search`'s output structures into the existing GB-energy / cleavage / segregation pipelines.

## Versioning + release

Additive — no breaking changes to existing symbols. Next patch tag, `pyiron_workflow_atomistics-0.0.10` (assuming current is `0.0.9`).

`CHANGELOG.md` gains a section:

```
## [0.0.10]
### Added
- `physics.grand_canonical_gb.gco_search` — grand-canonical optimization
  loop for GB-phase discovery, port of Chen & Frolov's GRIP algorithm
  (https://github.com/enze-chen/grip) onto the Engine Protocol.
- `physics.grand_canonical_gb.build_bicrystal_slabs` — convenience slab
  builder supporting fcc/bcc/hcp/dc/sc crystals.
- `physics.grand_canonical_gb.GCOConfig` — sampling configuration dataclass.
```

## Risk register

1. **GRIP `Bicrystal` is 640 lines mixing structure, Voronoi, and bookkeeping.** Port-as-is keeps the v1 PR tractable and preserves the algorithm's exact behavior, at the cost of a long file. If the file becomes a maintenance burden, future work splits the Voronoi/interstitial methods into their own module — but on its own that's not worth a separate PR cycle.

2. **The MD path assumes engines are dataclasses with a dataclass `EngineInput`.** `ASEEngine` satisfies this; `pyiron_workflow_lammps`' engine and the planned VASP engine do too. An engine that doesn't can still run minimize-only GCO; running MD-mode GCO with it will fail at the `dataclasses.replace` call. Documented in the `gco_search` docstring.

3. **Voronoi degeneracies on small cells.** Some small bicrystal cells produce zero in-bounds Voronoi vertices. Handled per-iteration (warn + skip swap), but if the user sets `inter_p=1.0` on a cell that always degenerates they get a search with no interstitial sampling. Acceptable; the warning is loud.

4. **GB-region atom count for the energy formula depends on the post-MD structure.** GRIP recomputes the mask after every calculation; we do the same. Risk: if MD displaces atoms past the `pad`-extended GB region, the count drops and `Egb` jumps. Same risk lived in GRIP; documented as a knob (`config.pad`).

5. **In-memory dedup grows DataFrames linearly until `dedup_every` fires.** For typical GRIP runs (10k–100k iterations) this is fine. Above ~1M kept structures it would be felt; reintroduce on-disk spill at that point.

## Out of scope

- LAMMPS-engine MD integration (waits on `pyiron_workflow_lammps` shipping `CalcInputMD` support; this design will work unchanged once that lands).
- Multi-component composition sampling. GRIP has `inter_s` (single species) and `inter_w` (a mix mode) for substitutions; we port the single-species path only. Multi-component goes into a follow-up.
- `plot_gco.py`. The existing matplotlib idioms in `physics/grain_boundary.py` cover the n-vs-Egb plot; we don't need a vendored equivalent.
- Bayesian/active-learning narrowing of sampling parameters (an upstream GRIP TODO).
- GB-atom on-the-fly identification (another upstream TODO).
