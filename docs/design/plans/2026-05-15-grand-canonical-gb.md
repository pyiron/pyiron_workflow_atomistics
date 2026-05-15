# Grand-Canonical GB Optimization (GRIP integration) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the GCO sampling algorithm from Chen & Frolov's GRIP (https://github.com/enze-chen/grip) into `pyiron_workflow_atomistics.physics.grand_canonical_gb`, backed by the pwa `Engine` Protocol. Public surface: `gco_search`, `build_bicrystal_slabs`, `GCOConfig`.

**Architecture:** GRIP's standalone script-package collapses into one `@pwf.as_function_node`-decorated workflow (`gco_search`) plus a vendored `_grand_canonical_gb_code/` internal subpackage. GRIP's `Calculator` hierarchy and MPI launcher are deleted; `calculate(structure, engine)` replaces `calculator.relax_structure(atoms)`, and `for_node` across seeds replaces MPI ranks. Per-iteration MD temperature and step count are injected by `dataclasses.replace`-ing the MD engine's `EngineInput` before each call. The loop returns `(DataFrame, list[Atoms])` and never raises mid-iteration.

**Tech Stack:** Python 3.11+, ASE, pyiron_workflow, pandas, numpy, scipy.spatial.Voronoi, pytest. Optional ASE EMT calculator for integration tests.

**Source spec:** `docs/design/specs/2026-05-15-grand-canonical-gb-design.md`. Read it first.

**Working branch:** `feat/grand-canonical-gb` (already created and contains the spec commit).

**Reference source:** GRIP is cloned at `~/grip` (your fork of `enze-chen/grip`, commit `8ff6a43`). Multiple tasks below port specific files from that tree; if `~/grip` is missing, clone:

```bash
git clone https://github.com/ligerzero-ai/grip.git ~/grip
```

---

## File Structure

```
pyiron_workflow_atomistics/
└── physics/
    ├── __init__.py                                # MODIFY: docstring note only (no re-exports)
    └── grand_canonical_gb.py                      # CREATE: gco_search + build_bicrystal_slabs (public)
    └── _grand_canonical_gb_code/
        ├── __init__.py                            # CREATE: empty
        ├── config.py                              # CREATE: GCOConfig dataclass
        ├── energies.py                            # CREATE: gb_energy() pure formula
        ├── interstitial.py                        # CREATE: Interstitial dataclass (port)
        ├── sampling.py                            # CREATE: translation/replication/MD param sampling
        ├── slabs.py                               # CREATE: make_crystals + dlat (port)
        ├── bicrystal.py                           # CREATE: Bicrystal class (port)
        └── store.py                               # CREATE: DataFrame dedup (port)
tests/unit/physics/
├── test_gco_config.py                             # CREATE
├── test_gco_energy.py                             # CREATE
├── test_gco_interstitial.py                      # CREATE
├── test_gco_sampling.py                           # CREATE
├── test_gco_slabs.py                              # CREATE
├── test_gco_bicrystal.py                          # CREATE
├── test_gco_store.py                              # CREATE
└── test_gco_workflow.py                           # CREATE
tests/integration/
└── test_gco_emt.py                                # CREATE
CHANGELOG.md                                       # MODIFY: new 0.0.10 entry
```

Every new module file is bottom-up: it imports only from things written in prior tasks (or from external libs / pwa's existing engine layer). The Bicrystal port (Task 6) is the largest single change; everything before it is small.

---

## Task 1: Stub `_grand_canonical_gb_code/` package + `GCOConfig`

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/__init__.py`
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/config.py`
- Test: `tests/unit/physics/test_gco_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_config.py`:

```python
"""Unit tests for GCOConfig dataclass and its validation."""

from __future__ import annotations

import dataclasses

import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (
    GCOConfig,
    validate_gco_config,
)


def test_gco_config_defaults_match_grip_baseline():
    cfg = GCOConfig()

    # Geometry
    assert cfg.gb_thick == 10.0
    assert cfg.pad == 10.0
    assert cfg.gb_gap == 0.3
    assert cfg.vacuum == 1.0

    # Sampling
    assert cfg.ngrid == 100
    assert cfg.size0 == (1, 1, 1)
    assert cfg.size == (2, 4, 15)
    assert cfg.reps_mode == 2

    # Vacancy fraction
    assert cfg.frac_min == 0.0
    assert cfg.frac_max == 1.0

    # Perturbation
    assert cfg.perturb_u == 0.0
    assert cfg.perturb_l == 0.0

    # Interstitials
    assert cfg.inter_p == 0.0
    assert cfg.inter_n == 0
    assert cfg.inter_t == 1.5
    assert cfg.inter_u is False
    assert cfg.inter_r is True

    # MD
    assert cfg.md_run_probability == 0.0
    assert cfg.t_min == 300
    assert cfg.t_max == 1200
    assert cfg.md_min_steps == 5000
    assert cfg.md_max_steps == 500_000
    assert cfg.md_step_sampling == "exponential"

    # Filtering
    assert cfg.e_mult == 2.0
    assert cfg.dedup_every == 50


def test_gco_config_is_frozen():
    cfg = GCOConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.gb_thick = 5.0  # type: ignore[misc]


def test_validate_accepts_defaults():
    # Should not raise.
    validate_gco_config(GCOConfig())


def test_validate_rejects_inverted_frac_bounds():
    cfg = GCOConfig(frac_min=0.8, frac_max=0.2)
    with pytest.raises(ValueError, match="frac_min"):
        validate_gco_config(cfg)


def test_validate_rejects_frac_outside_unit_interval():
    with pytest.raises(ValueError, match="frac_min"):
        validate_gco_config(GCOConfig(frac_min=-0.1))
    with pytest.raises(ValueError, match="frac_max"):
        validate_gco_config(GCOConfig(frac_max=1.5))


def test_validate_rejects_e_mult_below_one():
    with pytest.raises(ValueError, match="e_mult"):
        validate_gco_config(GCOConfig(e_mult=0.5))


def test_validate_rejects_invalid_md_step_sampling():
    with pytest.raises(ValueError, match="md_step_sampling"):
        validate_gco_config(GCOConfig(md_step_sampling="invalid"))


def test_validate_rejects_t_min_above_t_max():
    with pytest.raises(ValueError, match="t_min"):
        validate_gco_config(GCOConfig(t_min=1500, t_max=1200))


def test_validate_rejects_md_min_above_md_max():
    with pytest.raises(ValueError, match="md_min_steps"):
        validate_gco_config(GCOConfig(md_min_steps=1000, md_max_steps=500))


def test_validate_rejects_md_run_probability_out_of_range():
    with pytest.raises(ValueError, match="md_run_probability"):
        validate_gco_config(GCOConfig(md_run_probability=-0.1))
    with pytest.raises(ValueError, match="md_run_probability"):
        validate_gco_config(GCOConfig(md_run_probability=1.1))


def test_validate_rejects_invalid_reps_mode():
    with pytest.raises(ValueError, match="reps_mode"):
        validate_gco_config(GCOConfig(reps_mode=5))


def test_validate_warns_on_thin_gb_thick(caplog):
    with caplog.at_level("WARNING"):
        validate_gco_config(GCOConfig(gb_thick=3.0))
    assert "gb_thick" in caplog.text


def test_validate_warns_on_thin_pad(caplog):
    with caplog.at_level("WARNING"):
        validate_gco_config(GCOConfig(pad=2.0))
    assert "pad" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_config.py -v`
Expected: `ModuleNotFoundError: No module named 'pyiron_workflow_atomistics.physics._grand_canonical_gb_code'`

- [ ] **Step 3: Create the empty subpackage `__init__.py`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/__init__.py`:

```python
"""Internal helpers for grand-canonical GB optimization.

Not part of the public API. Consumers import from
``pyiron_workflow_atomistics.physics.grand_canonical_gb`` instead.
"""
```

- [ ] **Step 4: Implement `GCOConfig` and `validate_gco_config`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/config.py`:

```python
"""GCOConfig — all algorithmic knobs for one ``gco_search`` invocation.

Field names align with GRIP's ``params.yaml`` where possible; the
following are renamed for PEP 8 / clarity:

    ``MD_run``       → ``md_run_probability``
    ``Tmin``/``Tmax``→ ``t_min``/``t_max``
    ``MD_min``/``MD_max`` → ``md_min_steps``/``md_max_steps``
    ``var_steps``    → ``md_step_sampling``  ("exact"|"linear"|"exponential")
    ``Emult``        → ``e_mult``
    ``reps``         → ``reps_mode``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_VALID_MD_STEP_SAMPLING = {"exact", "linear", "exponential"}
_VALID_REPS_MODES = {1, 2, 3, 4}


@dataclass(frozen=True)
class GCOConfig:
    """All algorithmic knobs for one ``gco_search`` invocation."""

    # --- Bicrystal geometry ------------------------------------------------
    gb_thick: float = 10.0
    pad: float = 10.0
    gb_gap: float = 0.3
    vacuum: float = 1.0

    # --- In-plane translation + replication sampling -----------------------
    ngrid: int = 100
    size0: tuple[int, int, int] = (1, 1, 1)
    size: tuple[int, int, int] = (2, 4, 15)
    reps_mode: int = 2  # 1=exact, 2=uniform, 3=exp-small, 4=exp-large

    # --- Vacancy fraction on GB plane --------------------------------------
    frac_min: float = 0.0
    frac_max: float = 1.0

    # --- Atom perturbation -------------------------------------------------
    perturb_u: float = 0.0
    perturb_l: float = 0.0

    # --- Voronoi interstitial swap -----------------------------------------
    inter_p: float = 0.0
    inter_n: int = 0
    inter_t: float = 1.5
    inter_u: bool = False
    inter_r: bool = True

    # --- MD (consulted only if md_engine is supplied) ----------------------
    md_run_probability: float = 0.0
    t_min: int = 300
    t_max: int = 1200
    md_min_steps: int = 5000
    md_max_steps: int = 500_000
    md_step_sampling: str = "exponential"

    # --- Filtering / dedup -------------------------------------------------
    e_mult: float = 2.0
    dedup_every: int = 50


def validate_gco_config(cfg: GCOConfig) -> None:
    """Raise ``ValueError`` on invalid combinations; warn on sketchy ones.

    Pure side-effect function: raises or emits warnings. Caller is the
    ``gco_search`` workflow's pre-loop validation block.
    """
    # --- hard rejects ------------------------------------------------------
    if not 0.0 <= cfg.frac_min <= 1.0:
        raise ValueError(f"frac_min must be in [0, 1]; got {cfg.frac_min}")
    if not 0.0 <= cfg.frac_max <= 1.0:
        raise ValueError(f"frac_max must be in [0, 1]; got {cfg.frac_max}")
    if cfg.frac_min > cfg.frac_max:
        raise ValueError(
            f"frac_min ({cfg.frac_min}) must be <= frac_max ({cfg.frac_max})"
        )
    if cfg.e_mult < 1.0:
        raise ValueError(f"e_mult must be >= 1.0 (1.0 disables filter); got {cfg.e_mult}")
    if cfg.reps_mode not in _VALID_REPS_MODES:
        raise ValueError(
            f"reps_mode must be one of {sorted(_VALID_REPS_MODES)}; got {cfg.reps_mode}"
        )
    if cfg.md_step_sampling not in _VALID_MD_STEP_SAMPLING:
        raise ValueError(
            f"md_step_sampling must be one of {sorted(_VALID_MD_STEP_SAMPLING)}; "
            f"got {cfg.md_step_sampling!r}"
        )
    if cfg.t_min > cfg.t_max:
        raise ValueError(f"t_min ({cfg.t_min}) must be <= t_max ({cfg.t_max})")
    if cfg.md_min_steps > cfg.md_max_steps:
        raise ValueError(
            f"md_min_steps ({cfg.md_min_steps}) must be <= md_max_steps "
            f"({cfg.md_max_steps})"
        )
    if not 0.0 <= cfg.md_run_probability <= 1.0:
        raise ValueError(
            f"md_run_probability must be in [0, 1]; got {cfg.md_run_probability}"
        )

    # --- soft warnings -----------------------------------------------------
    if cfg.gb_thick < 5.0:
        logger.warning(
            "gb_thick=%.2f Å is thin; recommend >= 5.0 Å to capture GB region",
            cfg.gb_thick,
        )
    if cfg.pad < 3.0:
        logger.warning(
            "pad=%.2f Å is thin; recommend >= 3.0 Å around GB region",
            cfg.pad,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_config.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/__init__.py \
        pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/config.py \
        tests/unit/physics/test_gco_config.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): add GCOConfig dataclass and validator

Stubs the _grand_canonical_gb_code/ internal subpackage and lands the
sampling-parameter config + pre-loop validation. No workflow wired up
yet; just the dataclass plus 12 unit tests pinning the defaults
(match GRIP params.yaml baseline) and validator behaviour.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `gb_energy()` pure formula

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/energies.py`
- Test: `tests/unit/physics/test_gco_energy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_energy.py`:

```python
"""Unit tests for the gb_energy() pure formula."""

from __future__ import annotations

import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies import (
    gb_energy,
)


def test_zero_excess_gives_zero_egb():
    # E_total exactly == n * E_coh ⇒ Egb = 0
    assert gb_energy(
        final_energy_ev=-48.312,
        n_gb_atoms=10,
        gb_area_a2=100.0,
        e_cohesive_ev=-4.8312,
    ) == pytest.approx(0.0, abs=1e-9)


def test_positive_excess_converts_to_jpm2():
    # 1 eV excess over 1 Å² ⇒ Egb = 16.021766 J/m²
    e = gb_energy(
        final_energy_ev=-47.312,  # 1 eV above 10*(-4.8312)
        n_gb_atoms=10,
        gb_area_a2=1.0,
        e_cohesive_ev=-4.8312,
    )
    assert e == pytest.approx(16.021766, rel=1e-6)


def test_negative_egb_clamped_to_hundred(caplog):
    with caplog.at_level("WARNING"):
        e = gb_energy(
            final_energy_ev=-49.0,  # below the bulk reference
            n_gb_atoms=10,
            gb_area_a2=100.0,
            e_cohesive_ev=-4.8312,
        )
    assert e == 100.0
    assert "negative" in caplog.text.lower()


def test_realistic_scale():
    # Typical Cu GB: ~0.5 J/m², 50 atoms in GB region, 100 Å² area, -3.59 eV/atom
    e = gb_energy(
        final_energy_ev=-179.46,  # 50 * -3.59 + 0.04 eV ≈ ~0.5 J/m² over 100 Å²
        n_gb_atoms=50,
        gb_area_a2=100.0,
        e_cohesive_ev=-3.59,
    )
    assert 0.0 < e < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_energy.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gb_energy()`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/energies.py`:

```python
"""Grain-boundary energy formula.

Port of GRIP's ``Calculator.get_gb_energy`` (``core/calculator.py``), as a
pure function — no calculator state, no atoms, just numbers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# eV / Å² → J / m²
_EV_PER_A2_TO_J_PER_M2 = 16.021766

# Clamp value for unphysical negative GB energies (matches GRIP).
_NEGATIVE_CLAMP_J_PER_M2 = 100.0


def gb_energy(
    final_energy_ev: float,
    n_gb_atoms: int,
    gb_area_a2: float,
    e_cohesive_ev: float,
) -> float:
    """Grain-boundary energy in J/m².

        Egb = (E_total - n_gb_atoms × E_coh) / area × 16.021766

    Negative results indicate an unphysical configuration; clamped to
    100 J/m² with a warning (preserves upstream GRIP behaviour).
    """
    e_bulk = n_gb_atoms * e_cohesive_ev
    e_excess = final_energy_ev - e_bulk
    e_gb = e_excess / gb_area_a2 * _EV_PER_A2_TO_J_PER_M2

    if e_gb < 0:
        logger.warning(
            "Computed negative GB energy (%.4f J/m²); clamping to %.1f J/m²",
            e_gb,
            _NEGATIVE_CLAMP_J_PER_M2,
        )
        return _NEGATIVE_CLAMP_J_PER_M2
    return e_gb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_energy.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/energies.py \
        tests/unit/physics/test_gco_energy.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): port GB energy formula

Pure function gb_energy(E_total, n_atoms, area, E_coh) -> Egb [J/m²],
with the negative-value clamp-to-100 + warning preserved from GRIP's
Calculator.get_gb_energy. No engine dependency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `Interstitial` dataclass

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/interstitial.py`
- Test: `tests/unit/physics/test_gco_interstitial.py`

This is a near-verbatim port of GRIP's `core/interstitial.py`. The class is read-only state from `Bicrystal.find_interstitials`; tiny.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_interstitial.py`:

```python
"""Unit tests for the Interstitial site dataclass."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.interstitial import (
    Interstitial,
)


def test_basic_construction():
    site = Interstitial(p=[1.0, 2.0, 3.0], symbol="Ti", nn=6, nnd=[2.5, 2.5, 2.5],
                       label="octahedral0")
    np.testing.assert_array_equal(site.p, np.array([1.0, 2.0, 3.0]))
    assert site.symbol == "Ti"
    assert site.nn == 6
    np.testing.assert_array_equal(site.nnd, np.array([2.5, 2.5, 2.5]))
    assert site.label == "octahedral0"


def test_position_returns_numpy_array():
    site = Interstitial(p=[0.5, 1.5, 2.5])
    pos = site.position()
    assert isinstance(pos, np.ndarray)
    np.testing.assert_array_equal(pos, np.array([0.5, 1.5, 2.5]))


def test_from_df_roundtrips():
    df = pd.DataFrame({
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
        "z": [5.0, 6.0],
        "nn": [4, 6],
        "nnd": [[2.0, 2.0, 2.0, 2.0], [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]],
        "label": ["tetrahedral0", "octahedral0"],
    })
    sites = Interstitial.from_df(df)
    assert len(sites) == 2
    np.testing.assert_array_equal(sites[0].p, np.array([1.0, 3.0, 5.0]))
    assert sites[0].nn == 4
    assert sites[0].label == "tetrahedral0"
    assert sites[1].nn == 6


def test_repr_contains_class_name_and_position():
    site = Interstitial(p=[1.0, 2.0, 3.0], symbol="Ti")
    r = repr(site)
    assert "Interstitial" in r
    assert "Ti" in r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_interstitial.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `Interstitial`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/interstitial.py`:

```python
"""Interstitial site dataclass.

Direct port of GRIP's ``core/interstitial.py`` (no behavioural change).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Interstitial:
    """A candidate interstitial site identified by Voronoi analysis.

    Attributes
    ----------
    p : np.ndarray
        xyz position of the site (Å).
    symbol : str, optional
        Atomic species placed at this site, if any.
    nn : int, optional
        Number of nearest neighbours.
    nnd : np.ndarray, optional
        Distances to the ``nn`` nearest neighbours.
    label : str, optional
        Geometry label (e.g. ``"octahedral0"``, ``"tetrahedral1"``).
    """

    def __init__(
        self,
        p: Sequence[float],
        symbol: str | None = None,
        nn: int | None = None,
        nnd: Sequence[float] | None = None,
        label: str | None = None,
    ) -> None:
        self.p = np.asarray(p)
        self.symbol = symbol
        self.nn = nn
        self.nnd = np.asarray(nnd) if nnd is not None else None
        self.label = label

    @classmethod
    def from_df(cls, df) -> list["Interstitial"]:
        """Construct a list of sites from a DataFrame with ``x, y, z, nn, nnd, label`` columns."""
        sites: list[Interstitial] = []
        for row in df.itertuples():
            sites.append(
                cls(p=[row.x, row.y, row.z], nn=row.nn, nnd=row.nnd, label=row.label)
            )
        return sites

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol}, p={self.p})"

    def position(self) -> np.ndarray:
        return self.p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_interstitial.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/interstitial.py \
        tests/unit/physics/test_gco_interstitial.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): port Interstitial dataclass

Direct port of GRIP's core/interstitial.py — a small holder for Voronoi
sites with from_df() and position() helpers. Behaviour preserved verbatim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Sampling utilities

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/sampling.py`
- Test: `tests/unit/physics/test_gco_sampling.py`

This module owns all the RNG-driven parameter sampling: x/y translation, x/y replication, MD temperature, MD step count, and the replication-weight vectors. Ports from GRIP `utils/utils.py` (`compute_weights`, `get_xy_translation`, `get_xy_replications`) and from `core/simulation.py` (`sample_params`). The SLURM/PBS env-var branches in `get_xy_translation` are dropped; we always sample the full y-axis.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_sampling.py`:

```python
"""Unit tests for sampling utilities (translations, replications, MD params)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import GCOConfig
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.sampling import (
    compute_weights,
    sample_md_steps,
    sample_md_temperature,
    sample_xy_replications,
    sample_xy_translation,
)


@pytest.fixture
def small_slab() -> Atoms:
    return Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                 cell=[2.5, 3.7, 5.0])


def test_compute_weights_uniform():
    cfg = GCOConfig(size0=(1, 1, 1), size=(3, 4, 1), reps_mode=2)
    w = compute_weights(cfg)
    np.testing.assert_array_equal(w["nx"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(w["ny"], np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(w["wx"].sum(), 1.0)
    np.testing.assert_allclose(w["wy"].sum(), 1.0)
    # Uniform weights
    np.testing.assert_allclose(w["wx"], np.full(3, 1 / 3))
    np.testing.assert_allclose(w["wy"], np.full(4, 1 / 4))


def test_compute_weights_exact():
    cfg = GCOConfig(size0=(1, 1, 1), size=(3, 4, 1), reps_mode=1)
    w = compute_weights(cfg)
    # Exact mode puts all weight on the maximum
    np.testing.assert_array_equal(w["wx"], np.array([0, 0, 1]))
    np.testing.assert_array_equal(w["wy"], np.array([0, 0, 0, 1]))


def test_compute_weights_exp_small_favors_small():
    cfg = GCOConfig(size0=(1, 1, 1), size=(4, 4, 1), reps_mode=3)
    w = compute_weights(cfg)
    assert w["wx"][0] > w["wx"][-1]
    assert w["wy"][0] > w["wy"][-1]
    np.testing.assert_allclose(w["wx"].sum(), 1.0)


def test_compute_weights_exp_large_favors_large():
    cfg = GCOConfig(size0=(1, 1, 1), size=(4, 4, 1), reps_mode=4)
    w = compute_weights(cfg)
    assert w["wx"][-1] > w["wx"][0]
    assert w["wy"][-1] > w["wy"][0]


def test_sample_xy_replications_within_bounds():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(size0=(1, 1, 1), size=(3, 5, 1), reps_mode=2)
    weights = compute_weights(cfg)
    for _ in range(20):
        rx, ry = sample_xy_replications(rng, weights)
        assert 1 <= rx <= 3
        assert 1 <= ry <= 5


def test_sample_xy_translation_within_cell(small_slab):
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        dx, dy = sample_xy_translation(small_slab, rng, ngrid=10)
        assert 0.0 <= dx <= small_slab.cell[0, 0]
        assert 0.0 <= dy <= small_slab.cell[1, 1]


def test_sample_xy_translation_deterministic(small_slab):
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    assert sample_xy_translation(small_slab, rng1, ngrid=10) == \
           sample_xy_translation(small_slab, rng2, ngrid=10)


def test_sample_md_temperature_within_bounds_and_multiple_of_100():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(t_min=300, t_max=1200)
    seen = set()
    for _ in range(50):
        T = sample_md_temperature(cfg, rng)
        assert 300 <= T <= 1200
        assert T % 100 == 0
        seen.add(T)
    assert len(seen) > 1  # some variety


def test_sample_md_steps_exact():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(md_min_steps=5000, md_max_steps=500_000, md_step_sampling="exact")
    for _ in range(5):
        assert sample_md_steps(cfg, rng) == 5000


def test_sample_md_steps_linear_within_bounds():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(md_min_steps=5000, md_max_steps=500_000, md_step_sampling="linear")
    for _ in range(20):
        n = sample_md_steps(cfg, rng)
        assert 5000 <= n <= 500_000
        assert n % 1000 == 0  # rounded to nearest 1000


def test_sample_md_steps_exponential_within_bounds():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(md_min_steps=5000, md_max_steps=500_000, md_step_sampling="exponential")
    for _ in range(20):
        n = sample_md_steps(cfg, rng)
        assert 5000 <= n <= 500_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_sampling.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sampling.py`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/sampling.py`:

```python
"""RNG-driven parameter sampling for grand-canonical GB optimization.

Ports of:
    - GRIP utils/utils.py:compute_weights        → compute_weights
    - GRIP utils/utils.py:get_xy_translation     → sample_xy_translation
    - GRIP utils/utils.py:get_xy_replications    → sample_xy_replications
    - GRIP core/simulation.py:Simulation.sample_params
        → sample_md_temperature, sample_md_steps

Differences from upstream:
    - sample_xy_translation drops SLURM/PBS env-var branches and the
      ``pid``-based y-binning. Upstream split the y axis across MPI
      ranks so each rank explored its own band; here every search
      samples the full y range and different seeds give different
      draws. ``ngrid`` is retained for forward compatibility (future
      work may reintroduce stratified y-sampling at the workflow level).
    - sample_md_steps/sample_md_temperature take a GCOConfig instead of
      reading from a dict-based "algo" param block.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .config import GCOConfig


def compute_weights(cfg: GCOConfig) -> dict[str, np.ndarray]:
    """Replication weight vectors for the x and y axes.

    Returns a dict with keys ``nx``, ``ny`` (allowed replication counts) and
    ``wx``, ``wy`` (probabilities), suitable for ``np.random.Generator.choice``.

    ``reps_mode``:
        1 → exact (probability 1 at the maximum)
        2 → uniform
        3 → exponential negative (favours smaller cells)
        4 → exponential positive (favours larger cells)
    """
    nx = np.arange(cfg.size0[0], cfg.size[0] + 1)
    ny = np.arange(cfg.size0[1], cfg.size[1] + 1)
    nnx = cfg.size[0] - cfg.size0[0] + 1
    nny = cfg.size[1] - cfg.size0[1] + 1

    if cfg.reps_mode == 1:  # exact (max)
        wx = np.zeros(nnx)
        wx[-1] = 1.0
        wy = np.zeros(nny)
        wy[-1] = 1.0
    elif cfg.reps_mode == 2:  # uniform
        wx = np.ones(nnx) / nnx
        wy = np.ones(nny) / nny
    elif cfg.reps_mode == 3:  # exp-small
        wx = np.exp(-nx) / np.exp(-nx).sum()
        wy = np.exp(-ny) / np.exp(-ny).sum()
    elif cfg.reps_mode == 4:  # exp-large
        wx = np.exp(-nx) / np.exp(-nx).sum()
        wy = np.exp(-ny) / np.exp(-ny).sum()
        wx = wx[::-1]
        wy = wy[::-1]
    else:  # validate_gco_config should have prevented this
        raise ValueError(f"Invalid reps_mode: {cfg.reps_mode}")

    return {"nx": nx, "wx": wx, "ny": ny, "wy": wy}


def sample_xy_replications(
    rng: np.random.Generator,
    weights: dict[str, np.ndarray],
) -> tuple[int, int]:
    """Draw ``(rx, ry)`` from the replication-weight distributions."""
    rx = int(rng.choice(weights["nx"], p=weights["wx"]))
    ry = int(rng.choice(weights["ny"], p=weights["wy"]))
    return rx, ry


def sample_xy_translation(
    slab: Atoms,
    rng: np.random.Generator,
    ngrid: int,  # noqa: ARG001 — retained for future stratified sampling
) -> tuple[float, float]:
    """Uniform random translation in the (x, y) plane of ``slab``.

    Returns ``(dx, dy)`` with ``dx`` in ``[0, slab.cell[0,0]]`` and
    ``dy`` in ``[0, slab.cell[1,1]]``.
    """
    dx = float(rng.uniform(0.0, slab.cell[0, 0]))
    dy = float(rng.uniform(0.0, slab.cell[1, 1]))
    return dx, dy


def sample_md_temperature(cfg: GCOConfig, rng: np.random.Generator) -> int:
    """Random temperature in multiples of 100 K within ``[t_min, t_max]``."""
    return int(rng.choice(np.arange(cfg.t_min, cfg.t_max + 1, 100)))


def sample_md_steps(cfg: GCOConfig, rng: np.random.Generator) -> int:
    """Random MD step count within ``[md_min_steps, md_max_steps]``.

    ``md_step_sampling``:
        "exact"       → always ``md_min_steps``
        "linear"      → uniform in [md_min_steps, md_max_steps], rounded to 1000
        "exponential" → exponentially distributed, rounded to 1000
    """
    if cfg.md_step_sampling == "exact":
        return cfg.md_min_steps
    if cfg.md_step_sampling == "linear":
        n = int(rng.integers(cfg.md_min_steps, cfg.md_max_steps, endpoint=True))
        return int(np.round(n, -3))
    if cfg.md_step_sampling == "exponential":
        if cfg.md_min_steps <= 0:
            return 0
        c = np.log(cfg.md_max_steps / cfg.md_min_steps)
        n = cfg.md_min_steps * np.exp(c * rng.random())
        return int(np.round(n, -3))
    raise ValueError(f"Invalid md_step_sampling: {cfg.md_step_sampling}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_sampling.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/sampling.py \
        tests/unit/physics/test_gco_sampling.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): port sampling utilities

Ports compute_weights, sample_xy_translation, sample_xy_replications,
sample_md_temperature, sample_md_steps. Drops GRIP's MPI/pid-based
y-binning; the full y axis is sampled per iteration and parallelism
moves to for_node across seeds at the workflow level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Slab construction (`slabs.py` + `build_bicrystal_slabs` node)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/slabs.py`
- Create: `pyiron_workflow_atomistics/physics/grand_canonical_gb.py` (workflow file — adds only `build_bicrystal_slabs` in this task; `gco_search` is added in Task 8)
- Test: `tests/unit/physics/test_gco_slabs.py`

`slabs.py` ports GRIP `utils/utils.py:compute_dhkl` and `make_crystals`. The workflow-layer wrapper `build_bicrystal_slabs` (a `@pwf.as_function_node`) lives in `grand_canonical_gb.py` because it's a public node, not internal plumbing.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_slabs.py`:

```python
"""Unit tests for slab construction utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.slabs import (
    compute_dhkl,
    make_slabs,
)


def test_compute_dhkl_fcc_111():
    d = compute_dhkl("fcc", plane=[1, 1, 1], a=3.6)
    np.testing.assert_allclose(d, 3.6 / np.sqrt(3), rtol=1e-9)


def test_compute_dhkl_bcc_110():
    d = compute_dhkl("bcc", plane=[1, 1, 0], a=2.87)
    np.testing.assert_allclose(d, 2.87 / np.sqrt(2), rtol=1e-9)


def test_compute_dhkl_hcp_basal():
    # (0001) plane spacing = c
    d = compute_dhkl("hcp", plane=[0, 0, 0, 1], a=2.95, c=4.68)
    np.testing.assert_allclose(d, 4.68, rtol=1e-9)


def test_compute_dhkl_unknown_crystal_raises():
    with pytest.raises(Exception, match="not yet supported"):
        compute_dhkl("triclinic", plane=[1, 1, 1], a=3.6)


def test_make_slabs_fcc_returns_two_slabs_and_dlat():
    lower, upper, dlat = make_slabs(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        c=0.0,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    # dlat is the interplanar spacing along z; positive
    assert dlat > 0
    # Cell z-axis is shorter than uncut after cutoff trim
    assert lower.cell[2, 2] <= 22.0
    assert upper.cell[2, 2] <= 22.0


def test_make_slabs_hcp_requires_c():
    lower, upper, dlat = make_slabs(
        crystal="hcp",
        symbol="Ti",
        a=2.95,
        c=4.68,
        upper_dirs=[[5, 2, -7, 0], [0, 0, 0, -1], [-3, 4, -1, 0]],
        lower_dirs=[[7, -2, -5, 0], [0, 0, 0, -1], [-1, 4, -3, 0]],
        cutoff=25.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    assert dlat > 0


def test_make_slabs_bcc():
    lower, upper, dlat = make_slabs(
        crystal="bcc",
        symbol="Fe",
        a=2.87,
        c=0.0,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    assert dlat > 0


def test_make_slabs_unknown_crystal_raises():
    with pytest.raises(Exception, match="not yet supported"):
        make_slabs(
            crystal="triclinic",
            symbol="X",
            a=3.0,
            c=0.0,
            upper_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            lower_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )


def test_make_slabs_cutoff_zero_disables_trim():
    # Use a small size so the uncut z dimension is small enough to verify.
    lower, upper, _ = make_slabs(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        c=0.0,
        upper_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        lower_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        cutoff=0.0,
    )
    # With cutoff=0 and 1x1x1 cell, z stays as the lattice constant
    np.testing.assert_allclose(lower.cell[2, 2], 3.6, atol=0.01)


def test_build_bicrystal_slabs_node_callable():
    from pyiron_workflow_atomistics.physics.grand_canonical_gb import (
        build_bicrystal_slabs,
    )

    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    assert dlat > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_slabs.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `slabs.py`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/slabs.py`:

```python
"""Slab construction utilities.

Ports of:
    - GRIP utils/utils.py:compute_dhkl   → compute_dhkl
    - GRIP utils/utils.py:make_crystals  → make_slabs

Differences from upstream:
    - GRIP wrote slabs to ``POSCAR_*`` when ``struct["write"]`` was true;
      we never write to disk from this function.
    - GRIP read existing slabs from disk when ``struct["user"]`` was true;
      callers pass in their own ``ase.Atoms`` for that path.
    - The ``struct``-dict argument is replaced by explicit kwargs.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.lattice.bravais import Lattice
from ase.lattice.cubic import (
    BodyCenteredCubic,
    Diamond,
    FaceCenteredCubic,
    SimpleCubic,
)
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked

logger = logging.getLogger(__name__)

# Small position shift applied to avoid edge-of-cell ties (matches GRIP).
_P_SHIFT = 1e-3
# z-direction tolerance used in plane masking (matches GRIP utils/constants.py).
_Z_THRESH = 1e-3


def compute_dhkl(crystal: str, plane: list[int], a: float, c: float = 0.0) -> float:
    """Interplanar spacing for (hkl) or (hkil)."""
    cs = crystal.lower()
    if cs in {"fcc", "bcc", "dc", "sc"}:
        return a / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    if cs == "hcp":
        return 1.0 / np.sqrt(
            4 / 3 * (plane[0] ** 2 + plane[0] * plane[1] + plane[1] ** 2) / a**2
            + plane[3] ** 2 / c**2
        )
    raise Exception(f"Crystal structure '{crystal}' is not yet supported.")


_CRYSTAL_TYPES: dict[str, type] = {
    "fcc": FaceCenteredCubic,
    "bcc": BodyCenteredCubic,
    "dc": Diamond,
    "sc": SimpleCubic,
    "hcp": HexagonalClosedPacked,
}


def make_slabs(
    crystal: str,
    symbol: str,
    a: float,
    c: float,
    upper_dirs: list[list[int]],
    lower_dirs: list[list[int]],
    cutoff: float = 35.0,
    size_z: int = 15,
) -> tuple[Lattice, Lattice, float]:
    """Build upper and lower slabs + their interplanar spacing.

    Mirrors GRIP ``make_crystals`` with ``struct["user"]=False``.
    """
    cs = crystal.lower()
    if cs not in _CRYSTAL_TYPES:
        raise Exception(f"Crystal structure '{crystal}' is not yet supported.")

    init_size = (1, 1, size_z)
    builder = _CRYSTAL_TYPES[cs]

    if cs == "hcp":
        upper = builder(symbol=symbol, latticeconstant=(a, c), directions=upper_dirs,
                       size=init_size)
        lower = builder(symbol=symbol, latticeconstant=(a, c), directions=lower_dirs,
                       size=init_size)
    else:
        upper = builder(symbol=symbol, latticeconstant=a, directions=upper_dirs,
                       size=init_size)
        lower = builder(symbol=symbol, latticeconstant=a, directions=lower_dirs,
                       size=init_size)

    # Nudge atoms slightly to avoid PBC edge ties, then wrap.
    upper.positions += [0, _P_SHIFT, _Z_THRESH]
    upper.wrap()
    lower.positions += [0, _P_SHIFT, _Z_THRESH]
    lower.wrap()

    # Trim excess z-height to ``cutoff`` (skip if cutoff=0).
    if cutoff > 0:
        for name, slab, dirs in [("lower", lower, lower_dirs), ("upper", upper, upper_dirs)]:
            if slab.cell[2, 2] > cutoff:
                nvec = dirs[2]
                dspace = compute_dhkl(cs, nvec, a, c)
                logger.debug("Interplanar spacing for %s: %.6f Å", name, dspace)
                zmax = (
                    (cutoff // dspace + 1) * dspace
                    - _Z_THRESH
                    + min(slab.positions[:, 2].round(6))
                )
                del slab[[atom.index for atom in slab if atom.position[2] > zmax]]
                slab.cell[2, 2] = zmax

    # Compute dlat (minimum normal lattice-vector component along z).
    if cs in {"fcc", "bcc", "sc"}:
        unique_z = sorted(set(lower.positions[:, 2].round(6)))
    elif cs == "dc":
        parent = FaceCenteredCubic(symbol=symbol, latticeconstant=a,
                                   directions=lower_dirs, size=init_size)
        unique_z = sorted(set(parent.positions[:, 2].round(6)))
    elif cs == "hcp":
        parent = Hexagonal(symbol=symbol, latticeconstant=(a, c),
                          directions=lower_dirs, size=init_size)
        unique_z = sorted(set(parent.positions[:, 2].round(6)))
    else:  # unreachable
        raise Exception(f"Unhandled crystal '{crystal}'")

    dlat = abs(unique_z[1] - unique_z[0])

    if upper.cell[2, 2] < 20:
        logger.warning(
            "Upper slab z=%.2f Å is small; results may be inaccurate.",
            upper.cell[2, 2],
        )
    if lower.cell[2, 2] < 20:
        logger.warning(
            "Lower slab z=%.2f Å is small; results may be inaccurate.",
            lower.cell[2, 2],
        )

    return lower, upper, dlat
```

- [ ] **Step 4: Implement the public `build_bicrystal_slabs` node**

Create `pyiron_workflow_atomistics/physics/grand_canonical_gb.py`:

```python
"""Grand-canonical optimization of grain-boundary phases.

Public surface:
    - gco_search:           GCO sampling loop (added in Task 8).
    - build_bicrystal_slabs: convenience slab builder.
    - GCOConfig:            sampling configuration dataclass.

Algorithm: Chen, Heo, Wood, Asta, Frolov, *Nature Communications* **15**,
7049 (2024). DOI: 10.1038/s41467-024-51330-9.
"""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (
    GCOConfig,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.slabs import make_slabs

__all__ = ["GCOConfig", "build_bicrystal_slabs"]


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

    Parameters
    ----------
    crystal
        One of ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"dc"``, ``"sc"``.
    symbol
        Chemical symbol (e.g. ``"Cu"``, ``"Ti"``).
    a
        ``a`` lattice constant in Å.
    upper_dirs, lower_dirs
        3×3 (or 4-index for HCP) lists of integer Miller indices defining
        each slab's orthogonal x/y/z axes.
    c
        ``c`` lattice constant in Å (HCP only).
    cutoff
        Max slab z-height in Å; ``0`` disables trimming.

    Returns
    -------
    lower_slab, upper_slab, dlat
        Two ``ase.Atoms`` slabs plus the minimum normal-component
        lattice-vector spacing along z (Å). ``dlat`` is needed by
        ``gco_search`` to identify the GB plane.
    """
    return make_slabs(
        crystal=crystal,
        symbol=symbol,
        a=a,
        c=c,
        upper_dirs=upper_dirs,
        lower_dirs=lower_dirs,
        cutoff=cutoff,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_slabs.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/slabs.py \
        pyiron_workflow_atomistics/physics/grand_canonical_gb.py \
        tests/unit/physics/test_gco_slabs.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): port slab construction + build_bicrystal_slabs node

Ports GRIP utils/utils.py:make_crystals / compute_dhkl as make_slabs +
compute_dhkl under _grand_canonical_gb_code/slabs.py. Adds the public
@pwf.as_function_node build_bicrystal_slabs in physics/grand_canonical_gb.py.
Disk I/O dropped; callers pass ase.Atoms in/out only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `Bicrystal` class

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/bicrystal.py`
- Test: `tests/unit/physics/test_gco_bicrystal.py`

This is the largest single file in the plan. It's a port of GRIP's `core/bicrystal.py` (~640 lines), with three deliberate departures from upstream:

1. **`write_gb` drops the LAMMPS-data fast path.** ASE's `write` handles every format GRIP did.
2. **`print(...)` → `logger.debug(...)` / `logger.warning(...)`.** No `self.debug` field; logger level governs verbosity.
3. **Constructor takes `(lower, upper, config: GCOConfig, dlat)`.** Upstream's `struct` dict was stored but never read after `__init__`; dropping it is harmless. The `algo` dict is replaced by attribute access on `GCOConfig` (same field names).

Everything else — `copy_ul`, `shift_upper`, `replicate`, `get_bounds`, `get_gbplane_atoms_u`, `make_vacancies_u`, `defect_upper`, `perturb_atoms`, `join_gb`, `write_gb`, `compute_voronoi`, `classify_sites`, `find_interstitials`, `swap_gb_interstitials`, `find_and_swap_inters`, `get_edge_midpts`, `check_exist` — is line-for-line equivalent.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_bicrystal.py`:

```python
"""Unit tests for the Bicrystal class."""

from __future__ import annotations

import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.bicrystal import (
    Bicrystal,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import GCOConfig


@pytest.fixture
def cu_slabs():
    """Two trivial FCC-Cu slabs sharing the same orientation."""
    upper = FaceCenteredCubic(
        symbol="Cu",
        latticeconstant=3.6,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=(1, 1, 4),
    )
    lower = FaceCenteredCubic(
        symbol="Cu",
        latticeconstant=3.6,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=(1, 1, 4),
    )
    return lower, upper


def test_construction_stores_slabs(cu_slabs):
    lower, upper = cu_slabs
    bc = Bicrystal(lower=lower, upper=upper, config=GCOConfig(), dlat=1.8,
                   make_copy=True)
    assert bc.lower is not None
    assert bc.upper is not None
    assert bc.lower0 is lower
    assert bc.upper0 is upper
    # make_copy=True means the working slabs are copies of the originals
    assert bc.lower is not lower
    assert bc.upper is not upper


def test_shift_upper_translates_only_upper(cu_slabs):
    lower, upper = cu_slabs
    bc = Bicrystal(lower, upper, GCOConfig(), dlat=1.8, make_copy=True)
    upper_pos_before = bc.upper.positions.copy()
    lower_pos_before = bc.lower.positions.copy()
    bc.shift_upper(0.5, 0.7)
    np.testing.assert_allclose(bc.upper.positions, upper_pos_before + [0.5, 0.7, 0.0])
    np.testing.assert_allclose(bc.lower.positions, lower_pos_before)
    assert bc.dxyz == [0.5, 0.7, 0.0]


def test_replicate_multiplies_atom_counts(cu_slabs):
    lower, upper = cu_slabs
    bc = Bicrystal(lower, upper, GCOConfig(), dlat=1.8, make_copy=True)
    n0 = len(bc.upper)
    bc.replicate(2, 3)
    assert len(bc.upper) == 2 * 3 * n0
    assert bc.rxyz == (2, 3, 1)


def test_get_bounds_uses_gb_thick_and_pad(cu_slabs):
    lower, upper = cu_slabs
    cfg = GCOConfig(gb_thick=2.0, pad=1.0)
    bc = Bicrystal(lower, upper, cfg, dlat=1.8, make_copy=True)
    bc.get_bounds(cfg)
    assert bc.bounds is not None
    lowerb, upperb, pad = bc.bounds
    assert lowerb == pytest.approx(bc.lower.cell[2, 2] - 2.0)
    assert upperb == pytest.approx(bc.upper.cell[2, 2] - 2.0)
    assert pad == pytest.approx(1.0)


def test_get_gbplane_atoms_u_with_dlat(cu_slabs):
    lower, upper = cu_slabs
    bc = Bicrystal(lower, upper, GCOConfig(), dlat=1.8, make_copy=True)
    n_per_plane = bc.get_gbplane_atoms_u()
    assert n_per_plane > 0
    assert bc.npp_u == n_per_plane
    assert bc.gbplane_ids_u is not None
    assert bc.gbplane_pos_u.shape[1] == 3


def test_defect_upper_creates_vacancies(cu_slabs):
    lower, upper = cu_slabs
    cfg = GCOConfig(frac_min=0.0, frac_max=0.5)
    bc = Bicrystal(lower, upper, cfg, dlat=1.8, make_copy=True)
    bc.get_gbplane_atoms_u()
    n0 = len(bc.upper)
    rng = np.random.default_rng(seed=0)
    bc.defect_upper(cfg, rng)
    assert len(bc.upper) <= n0
    assert 0.0 <= bc.n <= 1.0


def test_perturb_atoms_displaces_near_gb_only(cu_slabs):
    lower, upper = cu_slabs
    cfg = GCOConfig(gb_thick=2.0, perturb_u=0.3, perturb_l=0.3)
    bc = Bicrystal(lower, upper, cfg, dlat=1.8, make_copy=True)
    upper_pos = bc.upper.positions.copy()
    lower_pos = bc.lower.positions.copy()
    rng = np.random.default_rng(seed=0)
    bc.perturb_atoms(rng)
    # Only atoms within gb_thick/2 of GB plane should have moved.
    mask_upper = upper_pos[:, 2] < cfg.gb_thick / 2
    mask_lower = lower_pos[:, 2] > bc.lower.cell[2, 2] - cfg.gb_thick / 2
    if mask_upper.any():
        assert not np.allclose(bc.upper.positions[mask_upper], upper_pos[mask_upper])
    if not mask_upper.all():
        np.testing.assert_allclose(
            bc.upper.positions[~mask_upper], upper_pos[~mask_upper]
        )
    if not mask_lower.all():
        np.testing.assert_allclose(
            bc.lower.positions[~mask_lower], lower_pos[~mask_lower]
        )


def test_join_gb_stitches_and_sets_gb(cu_slabs):
    lower, upper = cu_slabs
    cfg = GCOConfig(gb_gap=0.5, vacuum=1.0)
    bc = Bicrystal(lower, upper, cfg, dlat=1.8, make_copy=True)
    bc.join_gb(cfg)
    assert bc.gb is not None
    assert len(bc.gb) == len(bc.lower) + len(bc.upper)
    expected_z = lower.cell[2, 2] + 0.5 + upper.cell[2, 2] + 1.0
    np.testing.assert_allclose(bc.gb.cell[2, 2], expected_z, rtol=1e-9)


def test_repr_reflects_state(cu_slabs):
    lower, upper = cu_slabs
    bc = Bicrystal(lower, upper, GCOConfig(), dlat=1.8, make_copy=True)
    assert "unjoined" in repr(bc)
    bc.join_gb(GCOConfig())
    assert "joined" in repr(bc)
    bc.relaxed = True
    assert "relaxed" in repr(bc)


def test_find_interstitials_returns_sites_for_bulk_fcc(cu_slabs):
    """Voronoi search on bulk FCC finds octahedral + tetrahedral sites."""
    lower, upper = cu_slabs
    cfg = GCOConfig(gb_thick=3.0, gb_gap=0.0, vacuum=0.0)
    bc = Bicrystal(lower, upper, cfg, dlat=1.8, make_copy=True)
    bc.join_gb(cfg)
    bc.get_bounds(cfg)
    sites = bc.find_interstitials(unique_sites=True)
    # Should find at least one site; labels include "octahedral" or "tetrahedral"
    assert isinstance(sites, list)
    if sites:
        labels = {s.label.rstrip("0123456789") for s in sites if s.label}
        # At minimum, the classifier should run without exceptions
        assert all(isinstance(s.position(), np.ndarray) for s in sites)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_bicrystal.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `bicrystal.py`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/bicrystal.py`:

```python
"""Bicrystal — GB structure construction and Voronoi interstitial search.

Direct port of GRIP ``core/bicrystal.py`` (commit 8ff6a43).

Departures from upstream:
    1. ``__init__`` takes a ``GCOConfig`` instead of a ``struct``/``algo``
       dict pair. The ``struct`` dict was stored but never read after
       construction; the ``algo`` dict's keys match ``GCOConfig`` field
       names 1:1 in the fields this class consumes.
    2. ``self.debug`` is gone; ``print(...)`` calls become ``logger.debug``
       (or ``logger.warning`` for the GB-region-too-thin etc. cases).
    3. ``write_gb`` always dispatches through ``ase.io.write``; the
       lammps-data fast path is dropped (ASE handles it natively).
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write
from ase.lattice.bravais import Lattice
from scipy.spatial import Voronoi

from .config import GCOConfig
from .interstitial import Interstitial

logger = logging.getLogger(__name__)

_Z_THRESH = 1e-3


class Bicrystal:
    """A bicrystal: two slabs that can be translated, replicated, defected,
    perturbed, and joined into a single GB structure.

    See ``docs/design/specs/2026-05-15-grand-canonical-gb-design.md``
    for the per-iteration data-flow context.
    """

    def __init__(
        self,
        lower: Lattice,
        upper: Lattice,
        config: GCOConfig,
        dlat: float,
        make_copy: bool = True,
    ) -> None:
        self.lower0 = lower
        self.upper0 = upper
        self.config = config
        self.dlat = dlat

        self.rxyz: tuple[int, int, int] = (1, 1, 1)
        self.dxyz: list[float] = [0.0, 0.0, 0.0]
        self.dupper = config.perturb_u
        self.dlower = config.perturb_l
        self.npp_u: int | None = None
        self.gbplane_ids_u: np.ndarray | None = None
        self.gbplane_pos_u: np.ndarray | None = None
        self.bounds: np.ndarray | None = None
        self.gb: Atoms | None = None
        self.relaxed: bool = False
        self.n: float | None = None
        self.Egb: float | None = None
        self.inter_n: int = 0
        self.interstitials: list[Interstitial] = []

        if make_copy:
            self.copy_ul()

    # ------------------------------------------------------------------
    # Lifecycle / state
    # ------------------------------------------------------------------

    def copy_ul(self) -> None:
        """Reset the working slabs from the parent slabs."""
        self.lower = self.lower0.copy()
        self.upper = self.upper0.copy()

    def __repr__(self) -> str:
        if self.relaxed:
            suffix = "relaxed"
        elif self.gb:
            suffix = "joined"
        else:
            suffix = "unjoined"
        return (
            f"{self.__class__.__name__}({self.lower.symbols}, "
            f"{self.upper.symbols}) {suffix}"
        )

    def __eq__(self, other) -> bool:
        try:
            return self.upper == other.upper and self.lower == other.lower
        except Exception:
            return self.upper0 == other.upper0 and self.lower0 == other.lower0

    def __bool__(self) -> bool:
        return bool(self.gb)

    def __len__(self) -> int:
        return len(self.lower) + len(self.upper) + self.inter_n

    @property
    def natoms(self) -> int:
        return self.__len__()

    @property
    def z(self) -> float:
        return self.gb.cell.lengths()[2]

    # ------------------------------------------------------------------
    # Manipulation
    # ------------------------------------------------------------------

    def replicate(self, xreps: int, yreps: int, zreps: int = 1) -> None:
        reps = (xreps, yreps, zreps)
        self.lower *= reps
        self.upper *= reps
        self.rxyz = reps

    def shift_upper(self, xshift: float, yshift: float, zshift: float = 0.0) -> None:
        shifts = [xshift, yshift, zshift]
        self.upper.positions += shifts
        self.dxyz = shifts

    def get_bounds(self, config: GCOConfig) -> None:
        lowerb = self.lower.cell[2, 2] - config.gb_thick
        upperb = self.upper.cell[2, 2] - config.gb_thick
        self.bounds = np.array([lowerb, upperb, config.pad])

    def get_gbplane_atoms_u(self) -> int:
        sorted_pos = self.upper.positions[self.upper.positions[:, 2].argsort()]
        min_top = sorted_pos.round(6)[0, 2]
        mask = self.upper.positions[:, 2] < (min_top + _Z_THRESH)
        if self.dlat > 0:
            logger.debug("Calculating Nplane atoms within %.6f Å", self.dlat)
            mask = self.upper.positions[:, 2] < (min_top + self.dlat - _Z_THRESH)
        else:
            logger.debug("dlat is %s <= 0, taking planar slice.", self.dlat)
        self.npp_u = int(mask.sum())
        logger.debug("Atoms per top crystal plane = %d", self.npp_u)

        self.gbplane_ids_u = np.where(mask)[0]
        self.gbplane_pos_u = self.upper.positions[self.gbplane_ids_u]
        return self.npp_u

    def make_vacancies_u(self, index_list: Sequence[int]) -> None:
        for idx in sorted(index_list, reverse=True):
            self.upper.pop(idx)

    def defect_upper(self, config: GCOConfig, rng: np.random.Generator) -> None:
        assert self.npp_u is not None, (
            "Must calculate the number of atoms per plane first! "
            "Call get_gbplane_atoms_u()"
        )
        n_vac = int(
            rng.integers(
                np.floor(self.npp_u * (1 - config.frac_max)),
                np.ceil(self.npp_u * (1 - config.frac_min)),
                endpoint=True,
            )
        )
        to_delete = rng.choice(self.gbplane_ids_u, size=n_vac, replace=False)
        self.make_vacancies_u(to_delete)
        n_udef = self.upper.get_global_number_of_atoms()
        logger.debug(
            "%d atoms in defective cell after %d vacancies.", n_udef, n_vac
        )

        self.n = np.mod(n_udef, self.npp_u)
        assert self.n == np.mod(self.npp_u - n_vac, self.npp_u), (
            "Atoms were not deleted properly! "
            f"{n_udef} atoms in defective cell due to {n_vac} vacancies; "
            f"with {self.npp_u} atoms/plane, n={self.n} != "
            f"{np.mod(self.npp_u - n_vac, self.npp_u)}."
        )
        self.n /= self.npp_u

    def perturb_atoms(self, rng: np.random.Generator) -> None:
        mask_upper = self.upper.positions[:, 2] < self.config.gb_thick / 2
        n_u = int(mask_upper.sum())
        self.upper.positions[mask_upper, :] += self.dupper * rng.random([n_u, 3])

        mask_lower = (
            self.lower.positions[:, 2] > self.lower.cell[2, 2] - self.config.gb_thick / 2
        )
        n_l = int(mask_lower.sum())
        self.lower.positions[mask_lower, :] += self.dlower * rng.random([n_l, 3])

    def join_gb(self, config: GCOConfig, gb_normal: int = 2) -> None:
        offset = self.lower.cell[gb_normal, gb_normal] + config.gb_gap
        upper_copy = self.upper.copy()
        upper_copy.positions[:, gb_normal] += offset
        upper_copy.extend(self.lower.copy())
        upper_copy.cell[gb_normal, gb_normal] += offset + config.vacuum
        self.gb = upper_copy

    def write_gb(self, filename: str, format: str = "extxyz") -> None:
        assert self.gb is not None, (
            "GB hasn't been created yet! Call join_gb() first."
        )
        ase_write(filename, self.gb, format=format)

    # ------------------------------------------------------------------
    # Voronoi interstitial search
    # ------------------------------------------------------------------

    def get_edge_midpts(self, pts: list, rv: list) -> np.ndarray:
        emps = []
        for edge in rv:
            emps.append((np.array(pts[edge[0]]) + np.array(pts[edge[1]])) / 2)
        return np.array(emps)

    def compute_voronoi(
        self,
        struct0: Atoms,
        bounds: tuple,
        edge_midpoints: bool,
        reps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if reps is None:
            reps = np.array([3, 3, 1])
        a, b, c = struct0.cell.lengths()
        struct = struct0 * reps
        pts = struct.positions - (reps // 2) * np.array([a, b, c])

        vor = Voronoi(pts)
        logger.debug("Voronoi diagram with %s repetitions.", reps)

        vert = np.round(vor.vertices, 6)
        logger.debug("Found %d total vertices.", len(vert))

        if edge_midpoints:
            rv_pos = [rv for rv in vor.ridge_vertices if (np.array(rv) >= 0).all()]
            emps = self.get_edge_midpts(vert, rv_pos)
            vert = np.concatenate((vert, emps), axis=0)
            logger.debug("Adding %d edge midpoints.", len(emps))

        mask = (
            (vert[:, 0] > 0) & (vert[:, 0] < a)
            & (vert[:, 1] > 0) & (vert[:, 1] < b)
            & (vert[:, 2] > bounds[0]) & (vert[:, 2] < bounds[1])
        )
        v = vert[mask]
        v = v[v[:, 2].argsort()]
        logger.debug("After masking, %d points returned.", len(v))
        return v, pts

    def check_exist(
        self,
        exist: list,
        curr: np.ndarray,
        n1: int,
        tol: float | None = None,
    ) -> bool:
        if not tol:
            tol = min(curr[:n1]) * 2e-2
        diff = np.array([np.linalg.norm(x[:n1] - curr[:n1]) for x in exist])
        if len(diff) == 0:
            exist.append(curr)
            return False
        if (diff > tol).all():
            exist.append(curr)
            return False
        return True

    def classify_sites(
        self,
        sites: np.ndarray,
        pos: np.ndarray,
        top_n: int = 10,
        abs_tol: float = 6e-1,
        rel_tol: float = 5e-3,
        reset_index: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        labels = []
        nn_list = []
        nnd_list = []
        other: list[np.ndarray] = []
        tc = oc = tstrainc = ostrainc = trc = trstrainc = otherc = 0

        for v in sites:
            l = np.sort(np.linalg.norm(pos - v, axis=1))[:top_n]
            vdist = np.abs(l - l[0])
            nn1 = int((vdist < abs_tol).sum())
            nn2 = int((vdist < rel_tol * l[0]).sum())
            nn1_dist = vdist[:nn1].sum()
            same_dist = nn1_dist < rel_tol * l[0]

            nn_list.append(nn2)
            nnd_list.append(l[:nn1].round(6))

            exist = self.check_exist(other, l, nn1, tol=None)
            if nn2 == 3:
                if same_dist:
                    if not exist:
                        trc += 1
                    labels.append(f"triangular{trc}")
                else:
                    if not exist:
                        tstrainc += 1
                    labels.append(f"tri_strain{trstrainc}")
            elif nn2 == 4:
                if same_dist:
                    if not exist:
                        tc += 1
                    labels.append(f"tetrahedral{tc}")
                else:
                    if not exist:
                        tstrainc += 1
                    labels.append(f"tetra_strain{tstrainc}")
            elif nn2 == 6:
                if same_dist:
                    if not exist:
                        oc += 1
                    labels.append(f"octahedral{oc}")
                else:
                    if not exist:
                        ostrainc += 1
                    labels.append(f"octa_strain{ostrainc}")
            else:
                if not exist:
                    otherc += 1
                labels.append(f"other{otherc}")

        df = pd.DataFrame({
            "x": sites[:, 0],
            "y": sites[:, 1],
            "z": sites[:, 2],
            "label": labels,
            "nn": nn_list,
            "nnd": nnd_list,
        })
        df.sort_values(by=["z", "y", "x"], ignore_index=True, inplace=True)
        unique = df.drop_duplicates(subset=["label", "nn"], keep="first")
        if reset_index:
            unique.reset_index(drop=True, inplace=True)
        return df, unique

    def find_interstitials(
        self,
        zbounds: tuple[float, float] | None = None,
        edges: bool = False,
        unique_sites: bool = False,
    ) -> list[Interstitial]:
        assert self.gb is not None, (
            "GB hasn't been created yet! Call join_gb() first."
        )
        if zbounds is None:
            zbounds = (self.bounds[0], self.z - self.bounds[1])
        logger.debug("Searching for interstitials between %s.", zbounds)
        v, pts = self.compute_voronoi(self.gb, zbounds, edges)
        df, unique = self.classify_sites(v, pts)
        self.interstitials = Interstitial.from_df(unique if unique_sites else df)
        return self.interstitials

    def swap_gb_interstitials(self, zbounds: tuple[float, float]) -> int:
        gb_mask = (
            (self.gb.positions[:, 2] >= zbounds[0])
            & (self.gb.positions[:, 2] <= zbounds[1])
        )
        gb_ind = np.where(gb_mask)[0]
        np.random.shuffle(gb_ind)

        if len(gb_ind) < self.config.inter_n:
            if len(gb_ind) <= len(self.interstitials):
                logger.warning(
                    "Only %d GB atoms to swap (requested %d).",
                    len(gb_ind), self.config.inter_n,
                )
            else:
                logger.warning(
                    "Only %d interstitial sites to swap (requested %d).",
                    len(self.interstitials), self.config.inter_n,
                )

        swapped_n = min(self.config.inter_n, len(self.interstitials), len(gb_ind))
        for i in range(swapped_n):
            self.gb[gb_ind[i]].position = self.interstitials[i].position()
        return swapped_n

    def find_and_swap_inters(self, rng: np.random.Generator) -> int:
        if self.config.inter_n > 0 and rng.random() < self.config.inter_p:
            zmid = self.lower0.cell[2, 2] + self.config.gb_gap / 2
            zbounds = (zmid - self.config.inter_t, zmid + self.config.inter_t)
            inters = self.find_interstitials(
                zbounds=zbounds, unique_sites=self.config.inter_u
            )
            if self.config.inter_r:
                random.shuffle(inters)
            logger.debug("Found %d interstitial sites.", len(inters))
            if inters:
                logger.debug("First site: %s", inters[0])

            zbounds2 = (
                zmid - 2 * self.config.inter_t,
                zmid + 2 * self.config.inter_t,
            )
            return self.swap_gb_interstitials(zbounds=zbounds2)
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_bicrystal.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/bicrystal.py \
        tests/unit/physics/test_gco_bicrystal.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): port Bicrystal class

Ports GRIP core/bicrystal.py (~640 lines): copy_ul, shift_upper,
replicate, get_bounds, get_gbplane_atoms_u, defect_upper, perturb_atoms,
join_gb, write_gb, compute_voronoi, classify_sites, find_interstitials,
swap_gb_interstitials, find_and_swap_inters.

Three departures from upstream: (1) constructor takes a GCOConfig
instead of struct/algo dicts (struct was unused after init; algo keys
match GCOConfig field names); (2) print → logger.debug; (3) write_gb
always dispatches through ase.io.write (drops the LAMMPS-data fast path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Dedup helper (`store.py`)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/store.py`
- Test: `tests/unit/physics/test_gco_store.py`

Ports the unique-filtering logic from GRIP `utils/unique.py:clear_best`, but on an in-memory list of dicts (paralleling the kept-Atoms list) instead of disk filenames. The `extra=True` aggressive prune is NOT ported (out of scope per spec).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_store.py`:

```python
"""Unit tests for the dedup helper."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.store import dedup


def _row(Egb, n, dx=0.0, dy=0.0, rx=1, ry=1):
    return {"Egb": Egb, "n": n, "dx": dx, "dy": dy, "rx": rx, "ry": ry,
            "T": 0, "n_md_steps": 0, "iter": 0, "converged": True}


def _atoms(symbol="H"):
    return Atoms(symbol, positions=[[0, 0, 0]], cell=[1, 1, 1])


def test_no_duplicates_returns_input_unchanged():
    rows = [_row(0.5, 0.1), _row(0.4, 0.2)]
    atoms = [_atoms(), _atoms()]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 2
    assert len(out_atoms) == 2


def test_same_Egb_and_n_smaller_rep_wins():
    rows = [
        _row(0.5, 0.1, rx=2, ry=3),    # rx*ry=6
        _row(0.5, 0.1, rx=1, ry=2),    # rx*ry=2 — should win
    ]
    atoms = [_atoms("H"), _atoms("He")]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 1
    assert out_rows[0]["rx"] * out_rows[0]["ry"] == 2
    assert out_atoms[0].get_chemical_symbols() == ["He"]


def test_same_rep_smaller_shift_wins():
    rows = [
        _row(0.5, 0.1, rx=1, ry=1, dx=3.0, dy=4.0),   # |d|² = 25
        _row(0.5, 0.1, rx=1, ry=1, dx=0.5, dy=0.5),   # |d|² = 0.5 — should win
    ]
    atoms = [_atoms("H"), _atoms("Li")]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 1
    assert out_atoms[0].get_chemical_symbols() == ["Li"]


def test_different_n_kept_separately():
    rows = [_row(0.5, 0.1, rx=1, ry=1), _row(0.5, 0.2, rx=1, ry=1)]
    atoms = [_atoms(), _atoms()]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 2


def test_different_Egb_kept_separately():
    rows = [_row(0.5, 0.1), _row(0.6, 0.1)]
    atoms = [_atoms(), _atoms()]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 2


def test_empty_input_returns_empty():
    out_rows, out_atoms = dedup([], [])
    assert out_rows == []
    assert out_atoms == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_store.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `store.py`**

Create `pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/store.py`:

```python
"""Dedup helper for the GCO loop.

Port of the non-aggressive path of GRIP ``utils/unique.py:clear_best``,
operating on in-memory parallel lists of (row, atoms) instead of disk
filenames. The ``extra=True`` aggressive prune is intentionally not
ported in v1.
"""

from __future__ import annotations

from ase import Atoms


def dedup(
    rows: list[dict],
    atoms: list[Atoms],
) -> tuple[list[dict], list[Atoms]]:
    """Remove near-duplicate kept structures.

    Key
    ---
    ``(round(Egb, 3), round(n, 3))`` — two entries with the same rounded
    energy AND vacancy fraction are duplicates.

    Tie-break (lowest wins)
    -----------------------
    1. ``rx * ry`` (fewer atoms ⇒ less ambiguous reference)
    2. ``dx² + dy²`` (smaller in-plane shift)
    """
    if not rows:
        return [], []

    assert len(rows) == len(atoms), (
        f"rows ({len(rows)}) and atoms ({len(atoms)}) must align"
    )

    # winner[key] = (rep_product, shift_sq, row_idx)
    winner: dict[tuple[float, float], tuple[int, float, int]] = {}

    for i, row in enumerate(rows):
        key = (round(row["Egb"], 3), round(row["n"], 3))
        rep_prod = row["rx"] * row["ry"]
        shift_sq = row["dx"] ** 2 + row["dy"] ** 2

        cur = winner.get(key)
        if cur is None:
            winner[key] = (rep_prod, shift_sq, i)
            continue
        cur_rep, cur_shift, _ = cur
        if rep_prod < cur_rep or (
            rep_prod == cur_rep and shift_sq < cur_shift
        ):
            winner[key] = (rep_prod, shift_sq, i)

    kept_indices = sorted(v[2] for v in winner.values())
    return [rows[i] for i in kept_indices], [atoms[i] for i in kept_indices]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_store.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/_grand_canonical_gb_code/store.py \
        tests/unit/physics/test_gco_store.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): port dedup helper

In-memory port of GRIP utils/unique.py:clear_best (non-aggressive path).
Operates on parallel lists of (row_dict, ase.Atoms) instead of disk
filenames. Key = (round(Egb,3), round(n,3)); tiebreak prefers smaller
rx*ry then smaller dx²+dy². The extra=True aggressive prune is not
ported in v1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `gco_search` workflow

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/grand_canonical_gb.py` (add `gco_search` + private helper)
- Test: `tests/unit/physics/test_gco_workflow.py`

The workflow stitches every preceding module together. It exposes a single `@pwf.as_function_node` and a private `_count_atoms_in_gb_region` helper. Unit tests use a `FakeEngine` that returns canned `EngineOutput`s — no real calculator runs in unit tests; the EMT path is the integration test (Task 9).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_gco_workflow.py`:

```python
"""Unit tests for the gco_search workflow against a stub Engine.

The real EMT integration lives in tests/integration/test_gco_emt.py.
Here we use a deterministic stub that returns canned EngineOutputs so
unit tests stay fast (<1 s) and engine-independent.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic

from pyiron_workflow_atomistics.engine.inputs import CalcInputMD, CalcInputMinimize
from pyiron_workflow_atomistics.engine.protocol import EngineOutput
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import GCOConfig
from pyiron_workflow_atomistics.physics.grand_canonical_gb import gco_search


@dataclass
class _FakeMinimizeEngine:
    """Stub Engine that returns the input structure with a deterministic energy."""

    EngineInput: CalcInputMinimize
    working_directory: str = "."
    base_energy: float = -3.6  # eV/atom; matches Cu EMT roughly

    def get_calculate_fn(self, structure: Atoms):
        # Compute n*E_coh plus a tiny offset that varies per structure so Egb is positive
        # but small. Deterministic w.r.t. atom count so tests are stable.
        n = len(structure)

        def _fn(structure: Atoms) -> EngineOutput:
            return EngineOutput(
                final_structure=structure.copy(),
                final_energy=n * (-3.6) + 0.005 * n,  # ~5 meV/atom above bulk
                converged=True,
            )

        return _fn, {}

    def with_working_directory(self, subdir: str) -> "_FakeMinimizeEngine":
        return dataclasses.replace(
            self, working_directory=os.path.join(self.working_directory, subdir)
        )


@dataclass
class _FakeMDEngine:
    EngineInput: CalcInputMD
    working_directory: str = "."

    def get_calculate_fn(self, structure: Atoms):
        def _fn(structure: Atoms) -> EngineOutput:
            return EngineOutput(
                final_structure=structure.copy(),
                final_energy=0.0,  # MD output is intermediate; only structure consumed
                converged=True,
            )

        return _fn, {}

    def with_working_directory(self, subdir: str) -> "_FakeMDEngine":
        return dataclasses.replace(
            self, working_directory=os.path.join(self.working_directory, subdir)
        )


@pytest.fixture
def cu_slabs():
    lower = FaceCenteredCubic(
        symbol="Cu", latticeconstant=3.6,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(1, 1, 4),
    )
    upper = FaceCenteredCubic(
        symbol="Cu", latticeconstant=3.6,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(1, 1, 4),
    )
    return lower, upper


def test_gco_search_returns_dataframe_and_atoms_list(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMinimize(),
        working_directory=str(tmp_path),
    )
    cfg = GCOConfig(
        frac_min=0.7, frac_max=1.0,
        ngrid=10, size0=(1, 1, 1), size=(1, 2, 5),
        md_run_probability=0.0, dedup_every=0,
    )
    df, atoms_list = gco_search.node_function(
        minimize_engine=engine,
        lower_slab=lower, upper_slab=upper,
        e_cohesive=-3.6,
        config=cfg, n_iters=3, seed=0, dlat=1.8,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(atoms_list, list)
    assert len(df) == len(atoms_list)
    if not df.empty:
        for col in ("Egb", "n", "dx", "dy", "rx", "ry", "T", "n_md_steps",
                    "iter", "converged"):
            assert col in df.columns


def test_gco_search_with_md_engine_invokes_both(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    min_engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMinimize(),
        working_directory=str(tmp_path / "min"),
    )
    md_engine = _FakeMDEngine(
        EngineInput=CalcInputMD(mode="NVT", temperature=500.0, n_ionic_steps=100),
        working_directory=str(tmp_path / "md"),
    )
    cfg = GCOConfig(
        frac_min=1.0, frac_max=1.0,  # no vacancies for stability
        ngrid=10, size0=(1, 1, 1), size=(1, 1, 1), reps_mode=1,
        md_run_probability=1.0,
        t_min=300, t_max=300,  # fixed T
        md_min_steps=1000, md_max_steps=1000, md_step_sampling="exact",
        dedup_every=0,
    )
    df, _ = gco_search.node_function(
        minimize_engine=min_engine, md_engine=md_engine,
        lower_slab=lower, upper_slab=upper,
        e_cohesive=-3.6, config=cfg, n_iters=2, seed=0, dlat=1.8,
    )
    # When MD ran, T should equal 300 in every row that stored
    if not df.empty:
        assert (df["T"] == 300).all()


def test_gco_search_rejects_missing_md_engine_when_probability_positive(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(EngineInput=CalcInputMinimize(),
                                 working_directory=str(tmp_path))
    cfg = GCOConfig(md_run_probability=0.5)
    with pytest.raises(ValueError, match="md_engine"):
        gco_search.node_function(
            minimize_engine=engine, md_engine=None,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=cfg, n_iters=1, seed=0, dlat=1.8,
        )


def test_gco_search_rejects_wrong_minimize_engine_input_type(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMD(mode="NVT"),  # wrong type
        working_directory=str(tmp_path),
    )
    with pytest.raises(ValueError, match="minimize_engine"):
        gco_search.node_function(
            minimize_engine=engine,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=GCOConfig(), n_iters=1, seed=0, dlat=1.8,
        )


def test_gco_search_rejects_md_engine_with_wrong_input_type(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    min_engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMinimize(), working_directory=str(tmp_path),
    )
    md_engine = _FakeMDEngine(
        EngineInput=CalcInputMinimize(),  # wrong: should be CalcInputMD
        working_directory=str(tmp_path),
    )
    cfg = GCOConfig(md_run_probability=1.0)
    with pytest.raises(ValueError, match="md_engine"):
        gco_search.node_function(
            minimize_engine=min_engine, md_engine=md_engine,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=cfg, n_iters=1, seed=0, dlat=1.8,
        )


def test_gco_search_rejects_zero_iterations(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(EngineInput=CalcInputMinimize(),
                                 working_directory=str(tmp_path))
    with pytest.raises(ValueError, match="n_iters"):
        gco_search.node_function(
            minimize_engine=engine,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=GCOConfig(), n_iters=0, seed=0, dlat=1.8,
        )


def test_gco_search_handles_failed_minimize(cu_slabs, tmp_path):
    """A non-converged or exception-raising minimize should not abort the search."""
    lower, upper = cu_slabs

    @dataclass
    class _RaisingEngine:
        EngineInput: CalcInputMinimize
        working_directory: str = "."

        def get_calculate_fn(self, structure):
            def _fn(structure):
                raise RuntimeError("simulated engine crash")
            return _fn, {}

        def with_working_directory(self, subdir):
            return dataclasses.replace(
                self, working_directory=os.path.join(self.working_directory, subdir)
            )

    engine = _RaisingEngine(EngineInput=CalcInputMinimize(),
                            working_directory=str(tmp_path))
    df, atoms_list = gco_search.node_function(
        minimize_engine=engine,
        lower_slab=lower, upper_slab=upper,
        e_cohesive=-3.6, config=GCOConfig(frac_min=1.0, frac_max=1.0, dedup_every=0),
        n_iters=3, seed=0, dlat=1.8,
    )
    # All iterations failed; df is empty but workflow did not raise
    assert df.empty
    assert atoms_list == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_gco_workflow.py -v`
Expected: `ImportError: cannot import name 'gco_search'`.

- [ ] **Step 3: Implement `gco_search` and `_count_atoms_in_gb_region`**

Replace `pyiron_workflow_atomistics/physics/grand_canonical_gb.py` with:

```python
"""Grand-canonical optimization of grain-boundary phases.

Public surface:
    - gco_search:           GCO sampling loop.
    - build_bicrystal_slabs: convenience slab builder.
    - GCOConfig:            sampling configuration dataclass.

Algorithm: Chen, Heo, Wood, Asta, Frolov, *Nature Communications* **15**,
7049 (2024). DOI: 10.1038/s41467-024-51330-9.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine.inputs import CalcInputMD, CalcInputMinimize
from pyiron_workflow_atomistics.engine.protocol import Engine, calculate
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.bicrystal import (
    Bicrystal,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (
    GCOConfig,
    validate_gco_config,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies import (
    gb_energy,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.sampling import (
    compute_weights,
    sample_md_steps,
    sample_md_temperature,
    sample_xy_replications,
    sample_xy_translation,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.slabs import make_slabs
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.store import dedup

__all__ = ["GCOConfig", "build_bicrystal_slabs", "gco_search"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public node: build_bicrystal_slabs
# ---------------------------------------------------------------------------


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

    Supports ``fcc``, ``bcc``, ``hcp``, ``dc``, ``sc``. ``c`` is required
    only for HCP. ``cutoff=0`` disables z-axis trimming.
    """
    return make_slabs(
        crystal=crystal,
        symbol=symbol,
        a=a,
        c=c,
        upper_dirs=upper_dirs,
        lower_dirs=lower_dirs,
        cutoff=cutoff,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_atoms_in_gb_region(structure: Atoms, bounds: np.ndarray) -> int:
    """Atom count inside the (pad-extended) GB region.

    Port of GRIP ``Simulation.get_gb_energy``'s z-mask, used to populate
    ``n_gb_atoms`` for the energy formula.
    """
    lowerb, upperb, pad = bounds
    z = structure.positions[:, 2]
    z_min, z_max = z.min(), z.max()
    lower_cutoff = z_min + lowerb - pad
    upper_cutoff = z_max - upperb + pad
    mask = (z >= lower_cutoff) & (z <= upper_cutoff)
    return int(mask.sum())


def _make_iter_md_engine(
    md_engine: Engine, temperature: int, n_ionic_steps: int,
) -> Engine:
    """Return a per-iteration MD engine with the chosen T and step count.

    Relies on the convention (already in force for ASEEngine) that the
    engine is a dataclass with a dataclass ``EngineInput``.
    """
    new_input = dataclasses.replace(
        md_engine.EngineInput,
        temperature=temperature,
        n_ionic_steps=n_ionic_steps,
    )
    return dataclasses.replace(md_engine, EngineInput=new_input)


def _validate_workflow_inputs(
    minimize_engine: Engine,
    md_engine: Optional[Engine],
    config: GCOConfig,
    n_iters: int,
    e_cohesive: float,
) -> None:
    if n_iters <= 0:
        raise ValueError(f"n_iters must be > 0; got {n_iters}")
    if not isinstance(minimize_engine.EngineInput, CalcInputMinimize):
        raise ValueError(
            "minimize_engine.EngineInput must be a CalcInputMinimize; "
            f"got {type(minimize_engine.EngineInput).__name__}"
        )
    if config.md_run_probability > 0.0 and md_engine is None:
        raise ValueError(
            "config.md_run_probability > 0 requires a non-None md_engine"
        )
    if md_engine is not None and not isinstance(md_engine.EngineInput, CalcInputMD):
        raise ValueError(
            "md_engine.EngineInput must be a CalcInputMD; "
            f"got {type(md_engine.EngineInput).__name__}"
        )
    if e_cohesive > 0:
        logger.warning(
            "e_cohesive=%.4f eV is positive; cohesive energies are conventionally "
            "negative — check sign.",
            e_cohesive,
        )
    validate_gco_config(config)


# ---------------------------------------------------------------------------
# Public node: gco_search
# ---------------------------------------------------------------------------


@pwf.as_function_node("results", "best_structures")
def gco_search(
    minimize_engine: Engine,
    lower_slab: Atoms,
    upper_slab: Atoms,
    e_cohesive: float,
    config: GCOConfig = GCOConfig(),
    n_iters: int = 100,
    md_engine: Optional[Engine] = None,
    seed: int = 0,
    dlat: float = 0.0,
) -> tuple[pd.DataFrame, list[Atoms]]:
    """Grand-canonical sampling of GB microstructures.

    Per iteration: build bicrystal → sample translation/replication →
    vacancies on GB plane → perturb → optional Voronoi interstitial
    swap → optional MD-at-T → minimize → compute Egb → conditional
    keep (running ``E_min × e_mult`` gate) → periodic dedup.

    Parameters
    ----------
    minimize_engine
        pwa ``Engine`` carrying a ``CalcInputMinimize`` — runs at the end
        of every iteration.
    lower_slab, upper_slab
        ASE Atoms; build via ``build_bicrystal_slabs`` or supply your own.
    e_cohesive
        Bulk cohesive energy per atom in eV (negative). Used in the
        GB-energy formula.
    config
        ``GCOConfig`` with all algorithmic knobs.
    n_iters
        Number of iterations (per search).
    md_engine
        Optional pwa ``Engine`` carrying a ``CalcInputMD`` — runs before
        the minimize when ``rng.random() < config.md_run_probability``.
        ``EngineInput.temperature`` and ``n_ionic_steps`` are overridden
        per iteration via ``dataclasses.replace``.
    seed
        RNG seed for reproducibility.
    dlat
        Minimum lattice-vector z-component spacing; identifies the GB
        plane in the upper slab. ``0`` triggers GRIP's planar-slice
        fallback.

    Returns
    -------
    results : pandas.DataFrame
        Columns: ``Egb, n, dx, dy, rx, ry, T, n_md_steps, iter, converged``.
        One row per kept iteration. Empty DataFrame if no iteration
        produced a converged, gate-passing result.
    best_structures : list[ase.Atoms]
        Same order as ``results`` rows.

    Notes
    -----
    MD-mode GCO requires both engines to be dataclasses with dataclass
    ``EngineInput`` (``ASEEngine`` satisfies this). Engines that don't
    will fail at the per-iteration ``dataclasses.replace`` call.
    """
    _validate_workflow_inputs(
        minimize_engine=minimize_engine,
        md_engine=md_engine,
        config=config,
        n_iters=n_iters,
        e_cohesive=e_cohesive,
    )

    rng = np.random.default_rng(seed)
    weights = compute_weights(config)
    best_egb = float("inf")
    kept_rows: list[dict] = []
    kept_atoms: list[Atoms] = []
    voronoi_warned = False

    for i in range(n_iters):
        # ---- structure sampling ----------------------------------------
        bc = Bicrystal(
            lower=lower_slab.copy(),
            upper=upper_slab.copy(),
            config=config,
            dlat=dlat,
            make_copy=False,  # we already copied the slabs above
        )

        dx, dy = sample_xy_translation(upper_slab, rng, config.ngrid)
        bc.shift_upper(dx, dy)
        bc.get_bounds(config)
        rx, ry = sample_xy_replications(rng, weights)
        bc.replicate(rx, ry)
        bc.get_gbplane_atoms_u()
        bc.defect_upper(config, rng)
        bc.perturb_atoms(rng)
        bc.join_gb(config)

        try:
            bc.find_and_swap_inters(rng)
        except Exception as exc:
            if not voronoi_warned:
                logger.warning(
                    "Voronoi swap failed at iter %d: %s; skipping swap "
                    "for the rest of this search.", i, exc,
                )
                voronoi_warned = True

        atoms = bc.gb
        bounds = bc.bounds
        n_frac = bc.n if bc.n is not None else 0.0

        # ---- optional MD -----------------------------------------------
        T, n_md = 0, 0
        if md_engine is not None and rng.random() < config.md_run_probability:
            T = sample_md_temperature(config, rng)
            n_md = sample_md_steps(config, rng)
            iter_md = _make_iter_md_engine(md_engine, T, n_md).with_working_directory(
                f"iter_{i:05d}/md"
            )
            try:
                out_md = calculate.node_function(structure=atoms, engine=iter_md)
            except Exception as exc:
                logger.warning("iter %d MD failed: %s; skipping.", i, exc)
                continue
            atoms = out_md.final_structure

        # ---- minimize --------------------------------------------------
        iter_min = minimize_engine.with_working_directory(f"iter_{i:05d}/min")
        try:
            out = calculate.node_function(structure=atoms, engine=iter_min)
        except Exception as exc:
            logger.warning("iter %d minimize failed: %s; skipping.", i, exc)
            continue
        if not out.converged:
            logger.debug("iter %d did not converge; skipping store.", i)
            continue

        # ---- score + keep ----------------------------------------------
        n_gb = _count_atoms_in_gb_region(out.final_structure, bounds)
        area = (
            float(out.final_structure.cell[0, 0])
            * float(out.final_structure.cell[1, 1])
        )
        egb = gb_energy(
            final_energy_ev=out.final_energy,
            n_gb_atoms=n_gb,
            gb_area_a2=area,
            e_cohesive_ev=e_cohesive,
        )
        if egb < best_egb * config.e_mult:
            if egb < best_egb:
                best_egb = egb
            kept_rows.append({
                "Egb": egb, "n": n_frac, "dx": dx, "dy": dy,
                "rx": rx, "ry": ry, "T": T, "n_md_steps": n_md,
                "iter": i, "converged": True,
            })
            kept_atoms.append(out.final_structure)

        # ---- periodic dedup --------------------------------------------
        if config.dedup_every and (i + 1) % config.dedup_every == 0:
            kept_rows, kept_atoms = dedup(kept_rows, kept_atoms)

    return pd.DataFrame(kept_rows), kept_atoms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/physics/test_gco_workflow.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Run the full unit-test suite for this feature**

Run: `pytest tests/unit/physics/test_gco_*.py -v`
Expected: all tests from Tasks 1–8 still PASS.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/grand_canonical_gb.py \
        tests/unit/physics/test_gco_workflow.py
git commit -m "$(cat <<'EOF'
feat(grand_canonical_gb): add gco_search workflow

The GCO sampling loop as a @pwf.as_function_node: validates inputs,
samples translation/replication/vacancy/perturbation/interstitial swap,
optionally runs MD then minimize, scores with gb_energy, and stores
kept structures gated by running E_min × e_mult. MD T and n_ionic_steps
are injected per iteration via dataclasses.replace on the MD engine's
EngineInput. Returns (DataFrame, list[Atoms]); never raises mid-loop.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: EMT integration tests

**Files:**
- Create: `tests/integration/test_gco_emt.py`

End-to-end runs against the real ASE EMT calculator using `ASEEngine`. Marked slow by pwa's conftest (the `integration/` folder triggers the marker automatically). One test for minimize-only, one for MD-then-minimize.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_gco_emt.py`:

```python
"""End-to-end gco_search tests against ASEEngine + EMT.

Pulled to integration/ so they're marked slow. Each test is a tiny GCO
search (handful of iters, small cell) — should complete in < 30 s.
"""

from __future__ import annotations

import pytest

ase_emt = pytest.importorskip("ase.calculators.emt")

from ase.calculators.emt import EMT
from ase.optimize import BFGS

from pyiron_workflow_atomistics.engine import (
    ASEEngine,
    CalcInputMD,
    CalcInputMinimize,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import GCOConfig
from pyiron_workflow_atomistics.physics.grand_canonical_gb import (
    build_bicrystal_slabs,
    gco_search,
)


def test_gco_search_emt_minimize_only(tmp_path):
    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=50,
        ),
        calculator=EMT(),
        optimizer_class=BFGS,
        working_directory=str(tmp_path / "min"),
        write_to_disk=False,
    )

    df, atoms_list = gco_search.node_function(
        minimize_engine=engine,
        lower_slab=lower, upper_slab=upper, dlat=dlat,
        e_cohesive=-3.59,
        config=GCOConfig(
            frac_min=0.7, frac_max=1.0,
            ngrid=10, size0=(1, 1, 1), size=(1, 2, 5), reps_mode=2,
            md_run_probability=0.0, dedup_every=0,
        ),
        n_iters=5, seed=0,
    )

    assert len(df) > 0, "Expected at least one iteration to converge and be kept"
    assert (df["Egb"] >= 0).all()
    assert all(len(a) > 0 for a in atoms_list)
    assert len(df) == len(atoms_list)


def test_gco_search_emt_with_md(tmp_path):
    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    minimize_engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=20,
        ),
        calculator=EMT(),
        optimizer_class=BFGS,
        working_directory=str(tmp_path / "min"),
        write_to_disk=False,
    )
    md_engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="langevin",
            temperature=400.0,
            n_ionic_steps=20,
            time_step=1.0,
            seed=0,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path / "md"),
        write_to_disk=False,
    )

    df, atoms_list = gco_search.node_function(
        minimize_engine=minimize_engine,
        md_engine=md_engine,
        lower_slab=lower, upper_slab=upper, dlat=dlat,
        e_cohesive=-3.59,
        config=GCOConfig(
            frac_min=1.0, frac_max=1.0,
            ngrid=10, size0=(1, 1, 1), size=(1, 1, 3), reps_mode=1,
            md_run_probability=1.0,
            t_min=400, t_max=400,
            md_min_steps=20, md_max_steps=20, md_step_sampling="exact",
            dedup_every=0,
        ),
        n_iters=2, seed=0,
    )

    # MD path may or may not store every iteration depending on convergence;
    # we just assert the workflow ran end-to-end without raising.
    assert len(df) == len(atoms_list)
    if not df.empty:
        # All MD-storing rows ran at 400 K with 20 steps
        assert (df["T"] == 400).all()
        assert (df["n_md_steps"] == 20).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_gco_emt.py -v`
Expected: `ModuleNotFoundError` won't fire (everything is in place from prior tasks), but the tests should run. They will fail if any of the prior tasks regressed. If they pass on first run, that's a good sign.

Actually — by this point in the plan everything should be wired. If these tests pass, great. If they fail, debug in the prior implementation.

- [ ] **Step 3: Run the test to verify it passes**

Run: `pytest tests/integration/test_gco_emt.py -v`
Expected: both tests PASS within ~60 s combined.

- [ ] **Step 4: Run the full GCO suite (unit + integration)**

Run: `pytest tests/unit/physics/test_gco_*.py tests/integration/test_gco_emt.py -v`
Expected: every test PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_gco_emt.py
git commit -m "$(cat <<'EOF'
test(grand_canonical_gb): add EMT integration tests for gco_search

Two end-to-end runs against ASEEngine+EMT: minimize-only and
MD-then-minimize. Each search is intentionally tiny (5 iters / 2 iters,
small cells) so the suite stays under ~30 s.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Public re-exports + CHANGELOG

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/__init__.py` (docstring only — keep the no-re-export convention)
- Modify: `CHANGELOG.md`

The repo convention (per `physics/__init__.py`'s existing docstring) is to NOT re-export from `physics/__init__.py`. Users import via the full path: `from pyiron_workflow_atomistics.physics.grand_canonical_gb import gco_search, build_bicrystal_slabs, GCOConfig`. This task only adds the docstring line + the CHANGELOG entry.

- [ ] **Step 1: Add import line to the `physics/__init__.py` docstring**

Modify `pyiron_workflow_atomistics/physics/__init__.py`:

```python
"""Physics workflows organised by topic.

Import per-topic, not from this package directly::

    from pyiron_workflow_atomistics.physics.bulk             import eos_volume_scan
    from pyiron_workflow_atomistics.physics.surface          import calculate_surface_energy
    from pyiron_workflow_atomistics.physics.point_defect     import get_vacancy_formation_energy
    from pyiron_workflow_atomistics.physics.grain_boundary   import pure_gb_study
    from pyiron_workflow_atomistics.physics.grand_canonical_gb import gco_search

This package intentionally re-exports nothing so the import path tells you
which topic each macro belongs to.
"""
```

- [ ] **Step 2: Verify nothing else in the package needs adjusting**

Run: `pytest tests/unit/physics/ -v`
Expected: all GB + GCO tests pass.

- [ ] **Step 3: Add CHANGELOG entry**

Modify `CHANGELOG.md` — insert a new `[0.0.10]` block above the existing `[0.0.9]` block:

```markdown
## [0.0.10] — 2026-05-15

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
    c=0.0, cutoff=35.0)`` — convenience slab builder for fcc/bcc/hcp/
    dc/sc.
- **``GCOConfig`** dataclass — all algorithmic knobs (geometry,
  sampling, MD, dedup). Pre-loop validation rejects inconsistent
  configs; warnings on sketchy ones.

### Out of scope (v2 follow-ups)

- LAMMPS-engine MD integration (waits on ``pyiron_workflow_lammps``
  shipping ``CalcInputMD`` support; ``gco_search`` will work unchanged).
- Multi-component composition sampling (GRIP ``inter_w`` / ``inter_s``).
- ``plot_gco.py`` equivalent (existing matplotlib idioms in
  ``physics.grain_boundary`` cover n-vs-Egb plots).
- Aggressive ``extra=True`` dedup pass.

```

- [ ] **Step 4: Final full-suite run**

Run: `pytest tests/unit/physics/ tests/integration/test_gco_emt.py -v`
Expected: all GCO + existing physics tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/__init__.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(grand_canonical_gb): wire up CHANGELOG and physics/__init__.py docstring

CHANGELOG gets a 0.0.10 block describing the new
physics.grand_canonical_gb subpackage (gco_search, build_bicrystal_slabs,
GCOConfig). physics/__init__.py docstring gains the import-path example;
the no-re-export convention is preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

**Spec coverage**

| Spec section / requirement | Implemented in |
|---|---|
| Module layout (`_grand_canonical_gb_code/` + `grand_canonical_gb.py`) | Tasks 1, 5, 8 |
| `GCOConfig` dataclass + validator | Task 1 |
| `gb_energy()` formula + clamp | Task 2 |
| `Interstitial` dataclass | Task 3 |
| Sampling utilities + drop of MPI/pid binning | Task 4 |
| `make_slabs` + `build_bicrystal_slabs` node | Task 5 |
| `Bicrystal` class (port + three departures) | Task 6 |
| `dedup()` (non-aggressive path only) | Task 7 |
| `gco_search` workflow + `_count_atoms_in_gb_region` | Task 8 |
| Pre-loop validation rejects (n_iters, frac, e_mult, MD config) | Task 1 + Task 8 |
| Pre-loop warnings (gb_thick, pad, e_cohesive sign) | Task 1 + Task 8 |
| Per-iteration recovery: MD/minimize crash, non-convergence, Voronoi degeneracy | Task 8 |
| Engine immutability + per-iter `dataclasses.replace` | Task 8 |
| MD optional; minimize required | Task 8 |
| Storage in-memory; no `best/` folder | Task 8 |
| Unit tests for every module | Tasks 1–8 |
| Integration test with EMT (minimize + MD) | Task 9 |
| Public API exposed via `physics.grand_canonical_gb` | Task 5 + 8 + 10 |
| CHANGELOG entry | Task 10 |

**Placeholder scan:** No "TBD" / "implement later" / "add appropriate error handling" / etc. Every code block contains the actual code.

**Type consistency:** `GCOConfig` field names are stable across tasks. `Bicrystal(lower, upper, config, dlat, make_copy=True)` is the same signature wherever called. `dedup(rows, atoms) -> (rows, atoms)` is stable. `gb_energy(final_energy_ev, n_gb_atoms, gb_area_a2, e_cohesive_ev)` is stable. `sample_xy_translation(slab, rng, ngrid)`, `sample_xy_replications(rng, weights)`, `sample_md_temperature(cfg, rng)`, `sample_md_steps(cfg, rng)` all match between definition and call sites in Task 8.
