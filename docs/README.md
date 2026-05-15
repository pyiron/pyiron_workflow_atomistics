# pyiron_workflow_atomistics

[![Tests](https://github.com/pyiron/pyiron_workflow_atomistics/actions/workflows/push-pull.yml/badge.svg)](https://github.com/pyiron/pyiron_workflow_atomistics/actions/workflows/push-pull.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/pyiron_workflow_atomistics/badge.svg?branch=main)](https://coveralls.io/github/pyiron/pyiron_workflow_atomistics?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/pyiron_workflow_atomistics.svg)](https://pypi.org/project/pyiron_workflow_atomistics/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyiron_workflow_atomistics.svg)](https://pypi.org/project/pyiron_workflow_atomistics/)
[![Documentation Status](https://readthedocs.org/projects/pyiron_workflow_atomistics/badge/?version=latest)](https://pyiron_workflow_atomistics.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

`pyiron_workflow_atomistics` provides atomistic-simulation workflows for the [pyiron](https://pyiron.org) ecosystem. It exposes a single generic **`Engine`** interface that workflows talk to uniformly, plus topical physics macros (bulk, surface, point defects, grain boundaries) that compose into pyiron workflows.

## Layout

```
pyiron_workflow_atomistics/
├── engine/      # Engine Protocol, EngineOutput, run(), ASEEngine
├── structure/   # Generic builders / transforms / defect-structure generation
├── physics/     # Topical workflow macros — import per-topic
│   ├── bulk.py
│   ├── surface.py
│   ├── point_defect.py
│   ├── grain_boundary.py
│   └── phonons/        # κ(T) via phono3py + MD renormalisation via dynaphopy
│                       # (optional [phonons] / [phonons-md] install extras)
├── analysis/    # Featurisers, GB-plane finder, derived quantities
└── _internal/   # Private plumbing (not part of the public API)
```

Users import from the four public subpackages (`engine`, `structure`, `physics.*`, `analysis`); `_internal` is intentionally private.

## Installation

```bash
pip install pyiron_workflow_atomistics
# or
conda install -c conda-forge pyiron_workflow_atomistics
```

Optional extras for the phonons stack:

```bash
# κ(T) via phono3py: pulls phonopy + phono3py + symfc
pip install "pyiron_workflow_atomistics[phonons]"

# Adds MD-trajectory anharmonic renormalisation via dynaphopy (superset of [phonons])
pip install "pyiron_workflow_atomistics[phonons-md]"
```

## Quick start

Every workflow follows the same pattern: build an `Engine`, build a structure, and either call `run(structure, engine)` directly or drop into a topical macro.

```python
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize, run

engine = ASEEngine(
    EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
    calculator=EMT(),
    working_directory="./_runs",
)

structure = bulk("Cu", "fcc", a=3.6, cubic=True)
node = run(structure, engine=engine)
node.run()

out = node.outputs.engine_output.value           # EngineOutput dataclass
print(out.final_energy, out.converged)
```

`Engine.with_working_directory("subdir")` returns a pickleable copy with the path composed — use it to fan out per-calculation directories.

## The Engine layer

```python
from pyiron_workflow_atomistics.engine import (
    Engine,              # the Protocol every backend implements (runtime-checkable)
    EngineOutput,        # @dataclass returned by every engine call
    run,                 # the single workflow node: run(structure, engine)
    subengine,           # @as_function_node wrapper around engine.with_working_directory
    subdir_path,         # @as_function_node returning os.path.join(engine.wd, subdir)
    CalcInputStatic,
    CalcInputMinimize,
    CalcInputMD,
    ASEEngine,
)
```

The three `CalcInput*` dataclasses are jargon-free physics-level inputs (`force_convergence_tolerance`, `temperature`, `thermostat_time_constant`, ...) — engines map them to their backend's native parameters.

## Topical physics workflows

`physics/__init__.py` deliberately re-exports nothing — import per-topic so the path tells you what you're using.

### Bulk

```python
from pyiron_workflow_atomistics.physics.bulk import eos_volume_scan, optimise_cubic_lattice_parameter
from pyiron_workflow_atomistics.structure import get_bulk

structure = get_bulk.node_function("Cu", crystalstructure="fcc", a=3.6, cubic=True)
wf = eos_volume_scan(base_structure=structure, engine=engine,
                     axes=["a", "b", "c"], strain_range=(-0.05, 0.05), num_points=7)
wf.run()
print(f"v0 = {wf.outputs.v0.value:.3f} Å^3   B = {wf.outputs.B.value:.1f} GPa")
```

### Surface

```python
from ase.build import bulk
from pyiron_workflow_atomistics.physics.surface import calculate_surface_energy

cu_bulk = bulk("Cu", "fcc", a=3.6, cubic=True)
wf = calculate_surface_energy(
    bulk_structure=cu_bulk, engine=engine,
    miller_indices=(1, 1, 1), layers=3, vacuum=10.0,
)
wf.run()
print(wf.outputs.surface_energy.value, "J/m²")
```

### Point defects (vacancy + substitutional)

```python
from pyiron_workflow_atomistics.physics.point_defect import (
    get_vacancy_formation_energy,
    get_substitutional_formation_energy,
)

wf = get_vacancy_formation_energy(
    structure=bulk("Cu", "fcc", a=3.6, cubic=True),
    engine=engine,
    min_dimensions=[12, 12, 12],
)
wf.run()
print(wf.outputs.vacancy_formation_energy.value, "eV")

wf_sub = get_substitutional_formation_energy(
    structure=bulk("Cu", "fcc", a=3.6, cubic=True),
    engine=engine,
    new_symbol="Ni",
    min_dimensions=[12, 12, 12],
)
wf_sub.run()
```

### Grain boundaries

```python
from pyiron_workflow_atomistics.physics.grain_boundary import (
    cleave_gb_structure,
    find_viable_cleavage_planes_around_plane,
    get_GB_energy,
    pure_gb_study,
)
```

`pure_gb_study` composes length optimisation, segregation, and cleavage in one macro; the individual functions are available for finer control. See `notebooks/pure_grain_boundary_study.ipynb` and `notebooks/gb_cleavage.ipynb`.

### Phonons — κ(T) and MD-renormalised dispersion

Two macros backed by phonopy / phono3py / dynaphopy. Both fan force evaluations through the standard `Engine` Protocol so any ASE-style calculator (EMT, MACE, GRACE, M3GNet, CHGNet, DFT-via-ASE, …) works without modification. Gated on the `[phonons]` and `[phonons-md]` install extras.

**Lattice thermal conductivity κ(T)** — v0.0.7, `[phonons]` extra. Fits FC2 + FC3 via finite displacements (FD) or random displacements + symfc, runs the linearised Boltzmann transport equation on a q-mesh:

```python
import numpy as np
from ase.build import bulk
from pyiron_workflow_atomistics.physics.phonons import (
    calculate_phonon_thermal_conductivity,
)

wf = calculate_phonon_thermal_conductivity(
    structure=bulk("Cu", "fcc", a=3.6),
    engine=engine,
    fc2_supercell_matrix=2 * np.eye(3, dtype=int),
    fc3_supercell_matrix=2 * np.eye(3, dtype=int),
    temperatures=[300.0, 500.0, 700.0],
    q_mesh=(11, 11, 11),
    keep_handles=True,   # keep FC2/FC3/phono3py handles for downstream reuse
)
wf.run()
out = wf.outputs.phonon_output.value     # PhononOutput dataclass
print(out.kappa.shape, out.converged)    # (n_T, 6) in W/m·K (Voigt)
```

**Anharmonic phonon renormalisation via MD** — v0.0.8, `[phonons-md]` extra. Runs an ASE Langevin NVT segment using the engine's calculator, projects the trajectory onto the harmonic eigenmodes via dynaphopy, and fits Lorentzians to extract finite-temperature renormalised frequencies and linewidths:

```python
from pyiron_workflow_atomistics.physics.phonons import (
    calculate_phonon_md_renormalisation,
)

wf = calculate_phonon_md_renormalisation(
    structure=bulk("Si", "diamond", a=5.43),
    engine=engine,
    fc2_supercell_matrix=2 * np.eye(3, dtype=int),
    temperature=300.0,
    equilibration_steps=1000,
    production_steps=4000,
    time_step=1.0,                          # fs
    q_points=[[0.5, 0.0, 0.0]],             # primitive-cell X-point
    seed=42,
    power_spectra=True,                     # populate per-band MD power spectra
    keep_handles=True,                      # keep phonopy / dynaphopy handles
)
wf.run()
out = wf.outputs.md_phonon_output.value    # MdPhononOutput dataclass
print(out.harmonic_frequencies[0])         # phonopy at q-point, THz
print(out.renormalised_frequencies[0])     # dynaphopy MD-projected, THz
print(out.check_md_health())               # auto-warn on ⟨T⟩ drift / σ_T anomaly
```

The MD-renormalisation macro can also **reuse FC2** from a prior κ(T) run by passing `phono3py_output=out_kappa` instead of `fc2_supercell_matrix=...`, skipping the displacement-force fit entirely. Worked end-to-end example in [`notebooks/dynaphopy_grace_example.ipynb`](../notebooks/dynaphopy_grace_example.ipynb) — runs the GRACE-1L-OAM foundation model on Si 2×2×2 through both ASE Langevin and native LAMMPS (`pair_style grace`) MD drivers and compares the results.

## Structure manipulation

```python
from pyiron_workflow_atomistics.structure import (
    get_bulk, create_surface_slab,
    add_vacuum, create_supercell, create_supercell_with_min_dimensions, rattle,
    create_vacancy, substitutional_swap,
)
```

## Analysis

```python
from pyiron_workflow_atomistics.analysis import (
    voronoi_site_featuriser,
    distance_matrix_site_featuriser,
    soap_site_featuriser,
    find_gb_plane, plot_gb_plane,
    get_per_atom_quantity,
)
```

## Implementing a custom engine

`Engine` is a `typing.Protocol` — any class that satisfies the contract works. There is no base class to inherit from.

```python
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable
from ase import Atoms

from pyiron_workflow_atomistics.engine import (
    CalcInputStatic, CalcInputMinimize, CalcInputMD, EngineOutput,
)


@dataclass
class MyCustomEngine:
    """Drop-in replacement for ASEEngine targeting your backend."""

    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    backend_config: dict[str, Any] = field(default_factory=dict)
    working_directory: str = field(default_factory=os.getcwd)

    def get_calculate_fn(self, structure: Atoms) -> tuple[Callable[..., EngineOutput], dict[str, Any]]:
        """Return ``(callable, kwargs)``. The callable is invoked as
        ``callable(structure=structure, **kwargs)`` and must return an
        :class:`EngineOutput`. ``structure`` must NOT be in kwargs."""

        from my_backend import run_calculation

        kwargs = {
            "working_directory": self.working_directory,
            "config": self.backend_config,
            # map self.EngineInput.* into your backend's native parameters
        }
        return run_calculation, kwargs

    def with_working_directory(self, subdir: str) -> "MyCustomEngine":
        """Pure copy with the working directory composed — never mutate self."""
        return replace(
            self, working_directory=os.path.join(self.working_directory, subdir)
        )
```

The contract:
- **Pickleable** — workflows checkpoint to disk and may resubmit to SLURM.
- **`with_working_directory` is pure** — return a copy, do not mutate `self`. The recommended idiom is `dataclasses.replace`.
- **`get_calculate_fn(structure)`** returns `(callable, kwargs)`; the callable returns an `EngineOutput`.

Use `subengine(engine=engine, subdir="foo")` inside `@pwf.as_macro_node` bodies — calling `engine.with_working_directory(...)` directly on a channel input crashes pyiron_workflow's readiness checks.

## EngineOutput

```python
@dataclass
class EngineOutput:
    final_structure: Atoms
    final_energy: float
    converged: bool

    final_forces:        np.ndarray | None = None
    final_stress:        np.ndarray | None = None   # (3, 3)
    final_stress_voigt:  np.ndarray | None = None   # (6,)
    final_volume:        float       | None = None
    final_magmoms:       np.ndarray  | None = None

    energies:      list[float]        | None = None
    forces:        list[np.ndarray]   | None = None
    stresses:      list[np.ndarray]   | None = None
    structures:    list[Atoms]        | None = None
    n_ionic_steps: int                | None = None
```

`EngineOutput.to_dict()` returns a `dataclasses.asdict` view, with ASE objects preserved by reference.

## Notebooks

Worked examples covering every public workflow live in [`notebooks/`](../notebooks/). Each notebook supplies its own calculator (EMT for the Cu / Ni / Pd / Ag / Pt / Au / Al demos; EAM with the bundled `Al-Fe.eam.fs` for the Fe ones) so it executes self-contained.

Phonon workflows have their own dedicated notebooks:

- [`phonon_thermal_conductivity.ipynb`](../notebooks/phonon_thermal_conductivity.ipynb) — κ(T) on EMT Cu via `calculate_phonon_thermal_conductivity`, demonstrating the FC2 + FC3 displacement fits and the BTE solver.
- [`dynaphopy_grace_example.ipynb`](../notebooks/dynaphopy_grace_example.ipynb) — `calculate_phonon_md_renormalisation` on Si 2×2×2 with the GRACE-1L-OAM foundation model, showing both an ASE Langevin path and a native LAMMPS `pair_style grace` path projected through dynaphopy. Outputs and inline plots committed.

## Documentation

For full API documentation: [ReadTheDocs](https://pyiron_workflow_atomistics.readthedocs.io).

## Contributing

We welcome contributions — see [CONTRIBUTING.rst](../CONTRIBUTING.rst) for the development workflow.

## License

BSD 3-Clause — see [LICENSE](../LICENSE).

## Citation

```bibtex
@software{pyiron_workflow_atomistics,
  author = {pyiron team},
  title  = {pyiron_workflow_atomistics},
  year   = {2024},
  url    = {https://github.com/pyiron/pyiron_workflow_atomistics}
}
```
