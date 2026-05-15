# Free-energy consolidation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/design/specs/2026-05-15-free-energy-consolidation-design.md`

**Goal:** Add four new pyiron_workflow macros under `physics/free_energy/` — `harmonic_free_energy`, `quasiharmonic_free_energy`, `anharmonic_free_energy_dynaphopy`, `anharmonic_free_energy_dynaphopy_tdi` — so the same package houses all three levels of solid free energy (harmonic / QHA / anharmonic) alongside the existing calphy nodes, plus a teaching notebook overlaying all four methods.

**Architecture:** Import-and-wrap. `physics/phonons/{harmonic,anharmonic,md_renormalised}.py` are untouched; the new `free_energy/{harmonic,quasiharmonic,anharmonic_dynaphopy}.py` modules call into them. The existing `FreeEnergyOutput` dataclass is extended with optional fields per method; `mode` literal grows by four values. Reference engine is `ASEEngine(calculator=EMT())` for the EMT-friendly methods; calphy keeps its `LammpsEngine` requirement.

**Tech Stack:** `pyiron_workflow` (function-nodes + macros), `phonopy` (harmonic + QHA), `dynaphopy` (anharmonic via MD projection), `calphy` (anharmonic via TI — unchanged), `ase` (EMT calculator), `pytest` (with `@pytest.mark.slow` markers, gated on `pytest.importorskip(...)` per package).

**Test layout note:** The spec sketched `tests/unit/physics/free_energy/` (nested). The repo convention is flat: `tests/unit/physics/test_*.py`. This plan follows the repo convention.

---

## File structure

**Create:**
- `pyiron_workflow_atomistics/physics/free_energy/harmonic.py` — `harmonic_free_energy` macro + private helpers `_produce_fc2_view`, `_pack_harmonic_output`.
- `pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py` — `quasiharmonic_free_energy` macro + `_fit_qha`, `_check_qha_volume_range`, `_pack_qha_output`.
- `pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py` — `anharmonic_free_energy_dynaphopy` + `_tdi` macros + `_free_energy_from_spectrum`, `_pack_anharmonic_dynaphopy_output`, `_stack_tdi_outputs`.
- `tests/unit/physics/test_free_energy_harmonic.py` — unit + Tier-2 EMT smoke.
- `tests/unit/physics/test_free_energy_qha.py` — unit + Tier-2 EMT smoke.
- `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py` — unit + Tier-2 EMT smoke.
- `notebooks/free_energy_solid.ipynb` — Al/EMT four-method overlay.

**Modify:**
- `pyiron_workflow_atomistics/physics/free_energy/outputs.py` — extend `FreeEnergyOutput` (new fields + extended `mode` literal + handle-aware `to_dict()`).
- `pyiron_workflow_atomistics/physics/free_energy/__init__.py` — add four new exports.
- `tests/unit/physics/test_free_energy.py` — additional pickle/asdict round-trip cases for the new `mode` values.

---

## Task 1 — Extend `FreeEnergyOutput`

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/outputs.py`
- Test: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Add failing tests for new fields + handle-skipping `to_dict()`**

Append to `tests/unit/physics/test_free_energy.py`:

```python
def test_free_energy_output_accepts_harmonic_mode():
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="harmonic",
        reference_phase="solid",
        free_energy=-3.0,
        free_energy_error=0.0,
        temperature=0.0,
        pressure=0.0,
        n_atoms=4,
        elements=["Al"],
        simfolder="/tmp",
        report={},
        temperature_array=np.array([0.0, 300.0]),
        free_energy_array=np.array([-3.0, -3.05]),
        entropy_array=np.array([0.0, 1e-4]),
        heat_capacity_array=np.array([0.0, 2e-4]),
        entropy=0.0,
        heat_capacity=0.0,
    )
    assert out.mode == "harmonic"
    assert out.entropy_array.shape == (2,)


def test_free_energy_output_accepts_qha_mode():
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out = FreeEnergyOutput(
        mode="qha",
        reference_phase="solid",
        free_energy=-3.0,
        free_energy_error=0.0,
        temperature=0.0,
        pressure=0.0,
        n_atoms=4,
        elements=["Al"],
        simfolder="/tmp",
        report={},
        volumes=np.array([15.0, 16.0, 17.0]),
        free_energy_volume_array=np.zeros((2, 3)),
        equilibrium_volume_array=np.array([16.0, 16.1]),
        gibbs_free_energy_array=np.array([-3.0, -3.1]),
        bulk_modulus_array=np.array([70.0, 68.0]),
        thermal_expansion_array=np.array([0.0, 2e-5]),
    )
    assert out.bulk_modulus_array[0] == 70.0


def test_free_energy_output_accepts_anharmonic_dynaphopy_modes():
    import numpy as np
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    out_single = FreeEnergyOutput(
        mode="anharmonic_dynaphopy",
        reference_phase="solid",
        free_energy=-2.9,
        free_energy_error=0.0,
        temperature=300.0,
        pressure=0.0,
        n_atoms=4,
        elements=["Al"],
        simfolder="/tmp",
        report={},
        harmonic_frequencies=np.zeros((1, 12)),
        renormalised_frequencies=np.zeros((1, 12)),
        linewidths=np.zeros((1, 12)),
        q_mesh=(7, 7, 7),
    )
    assert out_single.q_mesh == (7, 7, 7)

    out_tdi = FreeEnergyOutput(
        mode="anharmonic_dynaphopy_tdi",
        reference_phase="solid",
        free_energy=-2.9,
        free_energy_error=0.0,
        temperature=200.0,
        pressure=0.0,
        n_atoms=4,
        elements=["Al"],
        simfolder="/tmp",
        report={},
        temperature_array=np.array([200.0, 400.0]),
        free_energy_array=np.array([-2.9, -3.0]),
        renormalised_frequencies_per_T=np.zeros((2, 1, 12)),
        linewidths_per_T=np.zeros((2, 1, 12)),
    )
    assert out_tdi.renormalised_frequencies_per_T.shape == (2, 1, 12)


def test_to_dict_excludes_handle_fields():
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    sentinel = object()
    out = FreeEnergyOutput(
        mode="harmonic",
        reference_phase="solid",
        free_energy=-3.0,
        free_energy_error=0.0,
        temperature=0.0,
        pressure=0.0,
        n_atoms=4,
        elements=["Al"],
        simfolder="/tmp",
        report={},
        phonopy_handle=sentinel,
        qha_handle=sentinel,
        dynaphopy_handle=sentinel,
    )
    d = out.to_dict()
    assert "phonopy_handle" not in d
    assert "qha_handle" not in d
    assert "dynaphopy_handle" not in d
    assert d["mode"] == "harmonic"
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/unit/physics/test_free_energy.py::test_free_energy_output_accepts_harmonic_mode tests/unit/physics/test_free_energy.py::test_free_energy_output_accepts_qha_mode tests/unit/physics/test_free_energy.py::test_free_energy_output_accepts_anharmonic_dynaphopy_modes tests/unit/physics/test_free_energy.py::test_to_dict_excludes_handle_fields -v`

Expected: all four FAIL. The first three fail because `mode` Literal does not include the new values and the new fields don't exist; the fourth fails because the current `to_dict()` (a plain `asdict`) includes every field.

- [ ] **Step 3: Extend `FreeEnergyOutput`**

Open `pyiron_workflow_atomistics/physics/free_energy/outputs.py`. Replace its current body with:

```python
"""Structured result of a free-energy calculation.

Same shape across calphy + phonopy harmonic + phonopy QHA + dynaphopy modes;
per-mode arrays are None when not applicable. Pickleable for plain-data
fields; phonopy/dynaphopy handle fields (carried in memory when
``keep_handles=True``) are excluded from ``to_dict()``.

Unit conventions
----------------
``free_energy`` / ``free_energy_error`` / ``free_energy_array`` / ``gibbs_free_energy_array``
/ ``einstein_free_energy``: eV/atom (calphy native).
``temperature`` / ``temperature_array``: K.
``pressure`` for calphy modes (fe, ts, tscale, pscale, melting_temperature,
alchemy, composition_scaling): bar (calphy native).
``pressure`` for ``qha``: GPa (phonopy.qha native).
``bulk_modulus_array``: GPa.
``thermal_expansion_array``: 1/K.
``volumes`` / ``equilibrium_volume_array``: Å³/atom.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np

_HANDLE_FIELDS = frozenset(
    {"phonopy_handle", "qha_handle", "dynaphopy_handle", "band_structure", "phonon_dos"}
)


@dataclass
class FreeEnergyOutput:
    """Result of one free-energy calculation across calphy / phonopy / dynaphopy modes."""

    mode: Literal[
        "fe",
        "ts",
        "tscale",
        "pscale",
        "melting_temperature",
        "alchemy",
        "composition_scaling",
        "harmonic",
        "qha",
        "anharmonic_dynaphopy",
        "anharmonic_dynaphopy_tdi",
    ]
    reference_phase: Literal["solid", "liquid", "both"]
    free_energy: float
    free_energy_error: float
    temperature: float
    pressure: float
    n_atoms: int
    elements: list[str]
    simfolder: str
    report: dict[str, Any]

    # existing optional fields
    temperature_array: np.ndarray | None = None
    free_energy_array: np.ndarray | None = None
    pressure_array: np.ndarray | None = None
    melting_temperature: float | None = None
    melting_temperature_error: float | None = None
    composition_path: list[dict[str, int]] | None = None
    einstein_free_energy: float | None = None

    # NEW — harmonic + dynaphopy (single T)
    entropy: float | None = None
    heat_capacity: float | None = None

    # NEW — harmonic + qha + dynaphopy TDI (arrays over temperature_array)
    entropy_array: np.ndarray | None = None
    heat_capacity_array: np.ndarray | None = None

    # NEW — qha specific
    volumes: np.ndarray | None = None
    free_energy_volume_array: np.ndarray | None = None
    equilibrium_volume_array: np.ndarray | None = None
    gibbs_free_energy_array: np.ndarray | None = None
    bulk_modulus_array: np.ndarray | None = None
    thermal_expansion_array: np.ndarray | None = None

    # NEW — dynaphopy specific
    harmonic_frequencies: np.ndarray | None = None
    renormalised_frequencies: np.ndarray | None = None
    linewidths: np.ndarray | None = None
    renormalised_frequencies_per_T: np.ndarray | None = None
    linewidths_per_T: np.ndarray | None = None
    q_mesh: tuple[int, int, int] | None = None

    # NEW — handles (always excluded from to_dict)
    phonopy_handle: Any | None = None
    qha_handle: Any | None = None
    dynaphopy_handle: Any | None = None
    band_structure: dict | None = None
    phonon_dos: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every non-handle field."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in _HANDLE_FIELDS
        }
```

- [ ] **Step 4: Run the four new tests to verify they pass**

Run: `pytest tests/unit/physics/test_free_energy.py::test_free_energy_output_accepts_harmonic_mode tests/unit/physics/test_free_energy.py::test_free_energy_output_accepts_qha_mode tests/unit/physics/test_free_energy.py::test_free_energy_output_accepts_anharmonic_dynaphopy_modes tests/unit/physics/test_free_energy.py::test_to_dict_excludes_handle_fields -v`

Expected: all four PASS.

- [ ] **Step 5: Run the full pre-existing test_free_energy.py to check for regressions**

Run: `pytest tests/unit/physics/test_free_energy.py -v`

Expected: all existing tests still pass. If a pre-existing test relied on handles appearing in `to_dict()` (it shouldn't — `_HANDLE_FIELDS` are all newly introduced), update that test to assert their absence and document the change in the commit message.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/outputs.py tests/unit/physics/test_free_energy.py
git commit -m "feat(free_energy): extend FreeEnergyOutput for harmonic/QHA/dynaphopy modes"
```

---

## Task 2 — `_free_energy_from_spectrum` (pure helper)

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py`
- Test: `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py`

- [ ] **Step 1: Write the failing test (Einstein-mode closed form)**

Create `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py`:

```python
"""Tests for pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy."""

from __future__ import annotations

import numpy as np
import pytest


def _einstein_free_energy_per_mode(omega_THz: float, T_K: float) -> float:
    """Closed-form F per mode for an Einstein oscillator. eV.

    F = ℏω/2 + k_B T ln(1 − exp(−ℏω / k_B T))
    """
    import scipy.constants as c

    omega_rad_s = omega_THz * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV
    if T_K == 0:
        return 0.5 * hbar_omega_eV
    kT_eV = c.Boltzmann * T_K / c.eV
    x = hbar_omega_eV / kT_eV
    return 0.5 * hbar_omega_eV + kT_eV * np.log1p(-np.exp(-x))


def test_free_energy_from_spectrum_matches_einstein_closed_form():
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        _free_energy_from_spectrum,
    )

    omega_THz = 5.0
    frequencies = np.full((1, 3), omega_THz)  # 1 q, 3 bands, all identical
    q_weights = np.array([1.0])
    F, S, Cv = _free_energy_from_spectrum.node_function(
        frequencies=frequencies,
        q_weights=q_weights,
        temperature=300.0,
        n_atoms_primitive=1,
    )

    expected_per_atom = 3 * _einstein_free_energy_per_mode(omega_THz, 300.0)
    assert F == pytest.approx(expected_per_atom, rel=1e-8)
    assert Cv > 0


def test_free_energy_from_spectrum_rejects_imaginary_modes():
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        _free_energy_from_spectrum,
    )

    frequencies = np.array([[5.0, -1.0, 3.0]])  # one imaginary
    q_weights = np.array([1.0])
    with pytest.raises(ValueError, match="imaginary modes"):
        _free_energy_from_spectrum.node_function(
            frequencies=frequencies,
            q_weights=q_weights,
            temperature=300.0,
            n_atoms_primitive=1,
        )
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_free_energy_from_spectrum_matches_einstein_closed_form tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_free_energy_from_spectrum_rejects_imaginary_modes -v`

Expected: FAIL (module does not exist yet).

- [ ] **Step 3: Create `anharmonic_dynaphopy.py` with `_free_energy_from_spectrum`**

```python
"""Anharmonic free energy via dynaphopy MD-projection — single T and TDI over T."""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf


@pwf.as_function_node("free_energy_per_atom", "entropy_per_atom", "cv_per_atom")
def _free_energy_from_spectrum(
    frequencies: np.ndarray,         # (n_q, n_band) THz
    q_weights: np.ndarray,           # (n_q,), sums to 1
    temperature: float,              # K
    n_atoms_primitive: int,
) -> tuple[float, float, float]:
    """Harmonic free energy / entropy / Cv on a discrete (q, band) frequency grid.

    F = sum_q w_q * sum_b [ ℏω_qb/2 + k_B T ln(1 − exp(−ℏω_qb / k_B T)) ]

    Acoustic modes at Γ are zeroed upstream by dynaphopy's
    `_project_with_dynaphopy` (positions[:3]=0 at q==0); any ω ≤ 0 remaining
    is treated as imaginary and rejected.

    Units
    -----
    Frequencies in THz. Returned F / S / Cv per primitive-cell atom, in
    eV / (eV/K) / (eV/K) respectively.
    """
    import scipy.constants as c

    freqs = np.asarray(frequencies, dtype=float)
    weights = np.asarray(q_weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"q_weights must sum to 1, got {weights.sum()}")
    if freqs.ndim != 2 or freqs.shape[0] != weights.shape[0]:
        raise ValueError(
            f"frequencies shape {freqs.shape} incompatible with q_weights shape "
            f"{weights.shape}"
        )

    # Acoustic-at-Γ modes (already zeroed upstream) are dropped from the sum.
    is_acoustic_gamma = freqs == 0.0
    active = freqs[~is_acoustic_gamma]
    imag_mask = active < 0
    if imag_mask.any():
        raise ValueError(
            f"Spectrum has {int(imag_mask.sum())} imaginary modes; "
            "harmonic free energy is undefined for an unstable spectrum."
        )

    omega_rad_s = freqs * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV  # (n_q, n_band)
    if temperature <= 0:
        # T=0: only zero-point energy contributes; S and Cv are zero.
        F_modes = 0.5 * hbar_omega_eV
        F_modes = np.where(is_acoustic_gamma, 0.0, F_modes)
        F = (weights[:, None] * F_modes).sum() / n_atoms_primitive
        return float(F), 0.0, 0.0

    kT_eV = c.Boltzmann * temperature / c.eV
    x = hbar_omega_eV / kT_eV
    # ln(1 − exp(−x)) — stable; modes with x huge → ln(1) = 0.
    log_term = np.where(is_acoustic_gamma, 0.0, np.log1p(-np.exp(-x)))
    F_modes = 0.5 * hbar_omega_eV + kT_eV * log_term
    F_modes = np.where(is_acoustic_gamma, 0.0, F_modes)

    # Entropy per mode:  S = k_B [ x/(exp(x)−1) − ln(1 − exp(−x)) ]
    kB_eV_per_K = c.Boltzmann / c.eV
    with np.errstate(over="ignore", invalid="ignore"):
        S_modes = kB_eV_per_K * (
            np.where(is_acoustic_gamma, 0.0, x / np.expm1(x)) - log_term
        )

    # Cv per mode:  C_v = k_B (x^2 exp(x)) / (exp(x)−1)^2
    with np.errstate(over="ignore", invalid="ignore"):
        denom = np.expm1(x) ** 2
        num = (x**2) * np.exp(x)
        Cv_modes = kB_eV_per_K * np.where(
            is_acoustic_gamma | (denom == 0), 0.0, num / denom
        )

    F = (weights[:, None] * F_modes).sum() / n_atoms_primitive
    S = (weights[:, None] * S_modes).sum() / n_atoms_primitive
    Cv = (weights[:, None] * Cv_modes).sum() / n_atoms_primitive
    return float(F), float(S), float(Cv)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_free_energy_from_spectrum_matches_einstein_closed_form tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_free_energy_from_spectrum_rejects_imaginary_modes -v`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py
git commit -m "feat(free_energy): add _free_energy_from_spectrum helper"
```

---

## Task 3 — `_produce_fc2_view` helper + `harmonic_free_energy` macro

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/harmonic.py`
- Test: `tests/unit/physics/test_free_energy_harmonic.py`

- [ ] **Step 1: Write failing Tier-2 EMT smoke test**

Create `tests/unit/physics/test_free_energy_harmonic.py`:

```python
"""Tests for pyiron_workflow_atomistics.physics.free_energy.harmonic."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_harmonic_free_energy_emt_al_2x2x2(tmp_path):
    pytest.importorskip("phonopy", reason="phonopy not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
        harmonic_free_energy,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = harmonic_free_energy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(0.0, 300.0),
        working_directory=str(tmp_path),
        subdir="harmonic",
    )
    out = wf.run()

    assert out.mode == "harmonic"
    assert out.reference_phase == "solid"
    # ZPE > 0 at T=0
    assert out.free_energy_array[0] > 0.0
    # F decreases with T (entropy dominates)
    assert out.free_energy_array[1] < out.free_energy_array[0]
    # Entropy at T=0 is zero
    assert out.entropy_array[0] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `pytest tests/unit/physics/test_free_energy_harmonic.py::test_harmonic_free_energy_emt_al_2x2x2 -v -m slow`

Expected: FAIL (`harmonic_free_energy` not importable).

- [ ] **Step 3: Implement `harmonic.py`**

Create `pyiron_workflow_atomistics/physics/free_energy/harmonic.py`:

```python
"""Harmonic free energy via phonopy FC2 — single user-facing entry point.

Built on top of phonopy via thin wrappers around the FC2 helpers in
`physics.phonons.harmonic`. The κ(T) workflow continues to own those
helpers; we import them here without behavioural change.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from numpy.typing import ArrayLike

from pyiron_workflow_atomistics.engine import Engine
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.phonons._compat import require_phonopy
from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
    _check_all_converged,
    _evaluate_supercells,
    _stack_forces,
)
from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _ase_to_phonopy,
    _compute_harmonic_observables,
    _generate_fc2_supercells,
    _normalise_supercell_matrix,
)


@pwf.as_function_node("phonopy_view")
def _produce_fc2_view(
    structure: Atoms,
    fc2_supercell_matrix: ArrayLike,
    fc2_engine_outputs: list,
    displacement_distance: float,
    is_plusminus,
):
    """Build a phonopy.Phonopy with FC2 fitted from supercell forces."""
    require_phonopy()
    import phonopy

    _check_all_converged(fc2_engine_outputs, label="FC2")
    sc = _normalise_supercell_matrix(fc2_supercell_matrix)

    unitcell = _ase_to_phonopy(structure)
    phonon = phonopy.Phonopy(
        unitcell=unitcell, supercell_matrix=sc, primitive_matrix="auto"
    )
    phonon.generate_displacements(
        distance=displacement_distance, is_plusminus=is_plusminus
    )
    forces = _stack_forces(fc2_engine_outputs)
    if forces.shape[0] != len(phonon.supercells_with_displacements):
        raise RuntimeError(
            f"FC2 force/supercell mismatch: {forces.shape[0]} forces vs "
            f"{len(phonon.supercells_with_displacements)} expected supercells. "
            "displacement kwargs likely drifted between generation and synthesis."
        )
    phonon.forces = forces
    phonon.produce_force_constants()
    return phonon


@pwf.as_function_node("free_energy_output")
def _pack_harmonic_output(
    structure: Atoms,
    phonopy_view,
    temperatures: ArrayLike,
    fc2_supercell_matrix: ArrayLike,
    displacement_distance: float,
    simfolder: str,
    keep_handles: bool,
) -> FreeEnergyOutput:
    """Compute thermal properties from the FC2 view and pack into FreeEnergyOutput."""
    T = np.asarray(temperatures, dtype=float)
    band_structure, dos, free_energy_dict = _compute_harmonic_observables(
        ph3=_Phono3pyShim(phonopy_view), temperatures=T
    )
    F = np.asarray(free_energy_dict["F"])
    S = np.asarray(free_energy_dict["S"])
    Cv = np.asarray(free_energy_dict["Cv"])

    elements = list(dict.fromkeys(structure.get_chemical_symbols()))
    return FreeEnergyOutput(
        mode="harmonic",
        reference_phase="solid",
        free_energy=float(F[0]),
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=0.0,
        n_atoms=len(structure),
        elements=elements,
        simfolder=simfolder,
        report={
            "method": "harmonic",
            "fc2_supercell_matrix": _normalise_supercell_matrix(
                fc2_supercell_matrix
            ).tolist(),
            "displacement_distance": float(displacement_distance),
        },
        temperature_array=T,
        free_energy_array=F,
        entropy=float(S[0]),
        heat_capacity=float(Cv[0]),
        entropy_array=S,
        heat_capacity_array=Cv,
        phonopy_handle=phonopy_view if keep_handles else None,
        band_structure=band_structure if keep_handles else None,
        phonon_dos=dos if keep_handles else None,
    )


class _Phono3pyShim:
    """Minimal adapter exposing the four attributes _compute_harmonic_observables reads.

    `_compute_harmonic_observables` was originally written against a Phono3py
    object; for the harmonic-only path we hand it a Phonopy view with the
    same four attributes. Keeps the helper signature unchanged.
    """

    def __init__(self, phonopy_view) -> None:
        self.phonon_primitive = phonopy_view.primitive
        self.phonon_supercell_matrix = phonopy_view.supercell_matrix
        self.fc2 = phonopy_view.force_constants


@pwf.api.as_macro_node("free_energy_output")
def harmonic_free_energy(
    wf,
    *,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0),
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    working_directory: str = ".",
    subdir: str = "harmonic_free_energy",
    keep_handles: bool = False,
):
    """Helmholtz free energy F(T), entropy S(T), heat capacity Cv(T) at fixed volume.

    Returns
    -------
    FreeEnergyOutput
        ``mode="harmonic"``, ``reference_phase="solid"``. The scalar
        ``free_energy`` is the value at the *lowest* T in ``temperatures``
        (typically T=0, i.e. zero-point energy). Curves are in
        ``temperature_array`` / ``free_energy_array`` / ``entropy_array`` /
        ``heat_capacity_array``.

    See spec: docs/design/specs/2026-05-15-free-energy-consolidation-design.md
    """
    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)

    sub_engine = engine.with_working_directory(simfolder)

    wf.fc2_supercells = _generate_fc2_supercells(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    wf.fc2_eval = _evaluate_supercells(
        supercells=wf.fc2_supercells.outputs.fc2_supercells,
        engine=sub_engine,
        prefix="fc2_disp_",
    )
    wf.fc2_view = _produce_fc2_view(
        structure=structure,
        fc2_supercell_matrix=fc2_supercell_matrix,
        fc2_engine_outputs=wf.fc2_eval.outputs.engine_outputs,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
    )
    wf.synthesis = _pack_harmonic_output(
        structure=structure,
        phonopy_view=wf.fc2_view.outputs.phonopy_view,
        temperatures=temperatures,
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=displacement_distance,
        simfolder=simfolder,
        keep_handles=keep_handles,
    )
    return wf.synthesis.outputs.free_energy_output
```

- [ ] **Step 4: Run the EMT smoke test to verify pass**

Run: `pytest tests/unit/physics/test_free_energy_harmonic.py::test_harmonic_free_energy_emt_al_2x2x2 -v -m slow`

Expected: PASS. Runtime ~60-90 s.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/harmonic.py tests/unit/physics/test_free_energy_harmonic.py
git commit -m "feat(free_energy): add harmonic_free_energy macro"
```

---

## Task 4 — `_fit_qha` + `_check_qha_volume_range` helpers

**Files:**
- Create: `pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py`
- Test: `tests/unit/physics/test_free_energy_qha.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/unit/physics/test_free_energy_qha.py`:

```python
"""Tests for pyiron_workflow_atomistics.physics.free_energy.quasiharmonic."""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_qha_inputs(n_T=5, n_V=7):
    """Build a synthetic well-behaved QHA input grid (Vinet-like E(V), Einstein phonons)."""
    V0 = 16.0  # Å³/atom
    B0 = 70.0  # GPa, converted later
    Bp = 4.0
    volumes = V0 * np.linspace(0.95, 1.05, n_V)
    # Murnaghan-like E(V) parametrisation in eV/atom
    energies = (
        -3.5
        + 9 * V0 * B0 / 1602.176 / Bp / (Bp - 1)
        * (volumes / V0) ** (1 - Bp) * ((volumes / V0) ** Bp - 1)
    )
    # Per-volume Einstein-like F(T): F(T,V) = -3 k_B T ln(...) — keep simple.
    temperatures = np.linspace(0, 400, n_T)
    F_TV = np.zeros((n_T, n_V))
    S_TV = np.zeros((n_T, n_V))
    Cv_TV = np.zeros((n_T, n_V))
    for j, V in enumerate(volumes):
        omega_THz = 5.0 * (V0 / V) ** 1.5
        from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
            _free_energy_from_spectrum,
        )

        for i, T in enumerate(temperatures):
            F_TV[i, j], S_TV[i, j], Cv_TV[i, j] = (
                _free_energy_from_spectrum.node_function(
                    frequencies=np.full((1, 3), omega_THz),
                    q_weights=np.array([1.0]),
                    temperature=T,
                    n_atoms_primitive=1,
                )
            )
    return energies, volumes, temperatures, F_TV, S_TV, Cv_TV


def test_fit_qha_produces_finite_arrays():
    pytest.importorskip("phonopy.qha", reason="phonopy.qha not installed")
    from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import _fit_qha

    energies, volumes, T, F_TV, S_TV, Cv_TV = _synthetic_qha_inputs()
    result = _fit_qha.node_function(
        energies=energies,
        volumes=volumes,
        free_energy_per_T_V=F_TV,
        entropy_per_T_V=S_TV,
        cv_per_T_V=Cv_TV,
        temperatures=T,
        pressure_GPa=0.0,
        eos_type="vinet",
    )
    for key in (
        "equilibrium_volume_array",
        "gibbs_free_energy_array",
        "bulk_modulus_array",
        "thermal_expansion_array",
    ):
        arr = result[key]
        assert arr.shape == T.shape, f"{key} shape {arr.shape} != {T.shape}"
        assert np.all(np.isfinite(arr[1:])), f"{key} has NaNs at finite T"


def test_check_qha_volume_range_raises_on_nan_volume():
    from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
        _check_qha_volume_range,
    )

    V_T = np.array([16.0, 16.1, np.nan, np.nan])
    T = np.array([0.0, 100.0, 200.0, 300.0])
    volumes = np.array([15.5, 16.0, 16.5])
    with pytest.raises(RuntimeError, match="QHA equilibrium volume undefined"):
        _check_qha_volume_range(V_T, T, strain_range=(-0.03, 0.03), volumes=volumes)


@pytest.mark.slow
def test_quasiharmonic_free_energy_emt_al(tmp_path):
    pytest.importorskip("phonopy.qha", reason="phonopy.qha not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
        quasiharmonic_free_energy,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = quasiharmonic_free_energy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(0.0, 300.0),
        strain_range=(-0.03, 0.03),
        num_volumes=5,
        working_directory=str(tmp_path),
        subdir="qha",
    )
    out = wf.run()

    assert out.mode == "qha"
    # Thermal expansion is positive on warming for Al/EMT
    assert out.equilibrium_volume_array[1] > out.equilibrium_volume_array[0]
    # Thermal expansion coefficient is finite and positive at 300 K
    assert np.isfinite(out.thermal_expansion_array[1])
    assert out.thermal_expansion_array[1] > 0
```

- [ ] **Step 2: Run the two unit tests; confirm both fail**

Run: `pytest tests/unit/physics/test_free_energy_qha.py::test_fit_qha_produces_finite_arrays tests/unit/physics/test_free_energy_qha.py::test_check_qha_volume_range_raises_on_nan_volume -v`

Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `quasiharmonic.py` skeleton (helpers only, macro after)**

Create `pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py`:

```python
"""Quasiharmonic free energy via phonopy.qha — single user-facing entry point.

QHA recipe: EOS volume sweep + per-volume harmonic free energy → phonopy.qha.QHA
gives G(T,P), V*(T,P), B(T,P), α(T,P). Reuses `harmonic_free_energy` per volume.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, calculate
from pyiron_workflow_atomistics.physics.bulk import generate_structures
from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
    harmonic_free_energy,
)
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.phonons._compat import require_phonopy


def _check_qha_volume_range(
    V_T: np.ndarray,
    temperatures: np.ndarray,
    strain_range: tuple[float, float],
    volumes: np.ndarray,
) -> None:
    """Raise if `phonopy.qha` returned NaN V*(T) anywhere — indicates V grid too narrow."""
    nan_mask = ~np.isfinite(V_T)
    if nan_mask.any():
        bad_T = np.asarray(temperatures)[nan_mask].tolist()
        raise RuntimeError(
            f"QHA equilibrium volume undefined at T={bad_T} K — widen "
            f"`strain_range` or extend it to include positive strain "
            f"(current range: {strain_range}, current V grid: {volumes.tolist()} Å³/atom)."
        )


@pwf.as_function_node("qha_results")
def _fit_qha(
    energies,
    volumes,
    free_energy_per_T_V,
    entropy_per_T_V,
    cv_per_T_V,
    temperatures,
    pressure_GPa: float,
    eos_type: str,
) -> dict:
    """Fit phonopy.qha.QHA on the (V, T) grid and return derived thermodynamics."""
    require_phonopy()
    from phonopy.qha import QHA

    E = np.asarray(energies, dtype=float)
    V = np.asarray(volumes, dtype=float)
    T = np.asarray(temperatures, dtype=float)
    F_TV = np.asarray(free_energy_per_T_V, dtype=float)
    S_TV = np.asarray(entropy_per_T_V, dtype=float)
    Cv_TV = np.asarray(cv_per_T_V, dtype=float)

    qha = QHA(
        volumes=V,
        electronic_energies=E,
        temperatures=T,
        free_energy=F_TV,
        cv=Cv_TV,
        entropy=S_TV,
        eos=eos_type,
        pressure=pressure_GPa,
    )
    qha.run()
    V_T = np.asarray(qha.get_volume_temperature())
    _check_qha_volume_range(V_T, T, strain_range=(V.min(), V.max()), volumes=V)

    return {
        "equilibrium_volume_array": V_T,
        "gibbs_free_energy_array": np.asarray(qha.get_gibbs_temperature()),
        "bulk_modulus_array": np.asarray(qha.get_bulk_modulus_temperature()),
        "thermal_expansion_array": np.asarray(qha.get_thermal_expansion()),
        "qha_handle": qha,
    }
```

- [ ] **Step 4: Run the two unit tests; verify they pass**

Run: `pytest tests/unit/physics/test_free_energy_qha.py::test_fit_qha_produces_finite_arrays tests/unit/physics/test_free_energy_qha.py::test_check_qha_volume_range_raises_on_nan_volume -v`

Expected: both PASS.

- [ ] **Step 5: Commit the helpers**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py tests/unit/physics/test_free_energy_qha.py
git commit -m "feat(free_energy): add _fit_qha + _check_qha_volume_range helpers"
```

---

## Task 5 — `quasiharmonic_free_energy` macro

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py`

- [ ] **Step 1: The Tier-2 EMT test from Task 4 currently FAILS — confirm**

Run: `pytest tests/unit/physics/test_free_energy_qha.py::test_quasiharmonic_free_energy_emt_al -v -m slow`

Expected: FAIL (`quasiharmonic_free_energy` not yet defined).

- [ ] **Step 2: Add per-volume helpers + the macro**

Append to `pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py`:

```python
@pwf.as_function_node("energies_per_volume", "volumes")
def _static_energies_per_volume(strained_structures: list[Atoms], engine: Engine):
    """One-shot static energy per strained cell. Returns (energies, volumes)."""
    energies, volumes = [], []
    for i, s in enumerate(strained_structures):
        sub_engine = engine.with_working_directory(f"vol_E_{i:03d}")
        out = calculate.node_function(structure=s, engine=sub_engine)
        if not out.converged:
            raise RuntimeError(
                f"Static-energy calc failed for strained cell {i} "
                f"(volume {s.get_volume():.3f} Å³)."
            )
        energies.append(float(out.final_energy))
        volumes.append(float(s.get_volume()) / len(s))
    return np.asarray(energies), np.asarray(volumes)


@pwf.as_function_node(
    "free_energy_per_T_V", "entropy_per_T_V", "cv_per_T_V"
)
def _harmonic_grid_over_volumes(
    strained_structures: list[Atoms],
    engine: Engine,
    fc2_supercell_matrix,
    temperatures,
    displacement_distance: float,
    is_plusminus,
    working_directory: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run harmonic_free_energy at each strained cell, stack F/S/Cv along V axis."""
    n_T = len(np.asarray(temperatures))
    n_V = len(strained_structures)
    F_TV = np.zeros((n_T, n_V))
    S_TV = np.zeros((n_T, n_V))
    Cv_TV = np.zeros((n_T, n_V))
    for j, s in enumerate(strained_structures):
        sub_engine = engine.with_working_directory(
            os.path.join(working_directory, f"vol_{j:03d}")
        )
        sub_wf = harmonic_free_energy(
            structure=s,
            engine=sub_engine,
            fc2_supercell_matrix=fc2_supercell_matrix,
            temperatures=temperatures,
            displacement_distance=displacement_distance,
            is_plusminus=is_plusminus,
            working_directory=os.path.join(working_directory, f"vol_{j:03d}"),
            subdir="harmonic",
        )
        out = sub_wf.run()
        F_TV[:, j] = out.free_energy_array
        S_TV[:, j] = out.entropy_array
        Cv_TV[:, j] = out.heat_capacity_array
    return F_TV, S_TV, Cv_TV


@pwf.as_function_node("free_energy_output")
def _pack_qha_output(
    structure: Atoms,
    qha_results: dict,
    energies: np.ndarray,
    volumes: np.ndarray,
    free_energy_per_T_V: np.ndarray,
    entropy_per_T_V: np.ndarray,
    cv_per_T_V: np.ndarray,
    temperatures: np.ndarray,
    pressure_GPa: float,
    simfolder: str,
    keep_handles: bool,
) -> FreeEnergyOutput:
    T = np.asarray(temperatures, dtype=float)
    elements = list(dict.fromkeys(structure.get_chemical_symbols()))
    return FreeEnergyOutput(
        mode="qha",
        reference_phase="solid",
        free_energy=float(qha_results["gibbs_free_energy_array"][0]),
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=float(pressure_GPa),
        n_atoms=len(structure),
        elements=elements,
        simfolder=simfolder,
        report={
            "method": "qha",
            "n_volumes": int(volumes.size),
            "pressure_GPa": float(pressure_GPa),
        },
        temperature_array=T,
        free_energy_array=qha_results["gibbs_free_energy_array"],
        entropy_array=entropy_per_T_V[:, entropy_per_T_V.shape[1] // 2],
        heat_capacity_array=cv_per_T_V[:, cv_per_T_V.shape[1] // 2],
        volumes=volumes,
        free_energy_volume_array=free_energy_per_T_V,
        equilibrium_volume_array=qha_results["equilibrium_volume_array"],
        gibbs_free_energy_array=qha_results["gibbs_free_energy_array"],
        bulk_modulus_array=qha_results["bulk_modulus_array"],
        thermal_expansion_array=qha_results["thermal_expansion_array"],
        qha_handle=qha_results["qha_handle"] if keep_handles else None,
    )


@pwf.api.as_macro_node("free_energy_output")
def quasiharmonic_free_energy(
    wf,
    *,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0),
    pressure: float = 0.0,
    strain_range: tuple[float, float] = (-0.03, 0.03),
    num_volumes: int = 7,
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    eos_type: str = "vinet",
    working_directory: str = ".",
    subdir: str = "quasiharmonic_free_energy",
    keep_handles: bool = False,
):
    """Gibbs free energy G(T,P), V*(T,P), B(T,P), α(T,P) via phonopy.qha.QHA.

    Pressure is in **GPa** (phonopy.qha native). At ``pressure=0.0`` the
    output is Helmholtz. See spec
    ``docs/design/specs/2026-05-15-free-energy-consolidation-design.md``.
    """
    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)

    wf.strained_structures = generate_structures(
        base_structure=structure,
        axes=["iso"],
        strain_range=strain_range,
        num_points=num_volumes,
    )
    wf.static_E = _static_energies_per_volume(
        strained_structures=wf.strained_structures.outputs.structure_list,
        engine=engine.with_working_directory(simfolder),
    )
    wf.harmonic_grid = _harmonic_grid_over_volumes(
        strained_structures=wf.strained_structures.outputs.structure_list,
        engine=engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperatures=temperatures,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        working_directory=simfolder,
    )
    wf.qha = _fit_qha(
        energies=wf.static_E.outputs.energies_per_volume,
        volumes=wf.static_E.outputs.volumes,
        free_energy_per_T_V=wf.harmonic_grid.outputs.free_energy_per_T_V,
        entropy_per_T_V=wf.harmonic_grid.outputs.entropy_per_T_V,
        cv_per_T_V=wf.harmonic_grid.outputs.cv_per_T_V,
        temperatures=temperatures,
        pressure_GPa=pressure,
        eos_type=eos_type,
    )
    wf.synthesis = _pack_qha_output(
        structure=structure,
        qha_results=wf.qha.outputs.qha_results,
        energies=wf.static_E.outputs.energies_per_volume,
        volumes=wf.static_E.outputs.volumes,
        free_energy_per_T_V=wf.harmonic_grid.outputs.free_energy_per_T_V,
        entropy_per_T_V=wf.harmonic_grid.outputs.entropy_per_T_V,
        cv_per_T_V=wf.harmonic_grid.outputs.cv_per_T_V,
        temperatures=temperatures,
        pressure_GPa=pressure,
        simfolder=simfolder,
        keep_handles=keep_handles,
    )
    return wf.synthesis.outputs.free_energy_output
```

- [ ] **Step 3: Run the EMT smoke test to verify pass**

Run: `pytest tests/unit/physics/test_free_energy_qha.py::test_quasiharmonic_free_energy_emt_al -v -m slow`

Expected: PASS. Runtime ~3-5 min (5 volumes × FC2 supercell pipeline).

- [ ] **Step 4: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/quasiharmonic.py
git commit -m "feat(free_energy): add quasiharmonic_free_energy macro"
```

---

## Task 6 — `anharmonic_free_energy_dynaphopy` macro (single T)

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py`
- Modify: `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py`

- [ ] **Step 1: Add the failing Tier-2 EMT smoke test**

Append to `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py`:

```python
@pytest.mark.slow
def test_anharmonic_free_energy_dynaphopy_emt_al(tmp_path):
    pytest.importorskip("dynaphopy", reason="dynaphopy not installed")
    pytest.importorskip("phonopy", reason="phonopy not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        anharmonic_free_energy_dynaphopy,
    )
    from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
        harmonic_free_energy,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    out_h = harmonic_free_energy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(300.0,),
        working_directory=str(tmp_path),
        subdir="harmonic_ref",
    ).run()

    out_a = anharmonic_free_energy_dynaphopy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        production_steps=2000,
        q_mesh=(5, 5, 5),
        working_directory=str(tmp_path),
        subdir="anharmonic_T300",
    ).run()

    assert out_a.mode == "anharmonic_dynaphopy"
    assert out_a.temperature == 300.0
    # Anharmonic and harmonic Al/EMT at 300 K should be within 50 meV/atom
    assert abs(out_a.free_energy - out_h.free_energy) < 0.05
    assert out_a.harmonic_frequencies.shape == out_a.renormalised_frequencies.shape
```

- [ ] **Step 2: Run the test; confirm failure**

Run: `pytest tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_anharmonic_free_energy_dynaphopy_emt_al -v -m slow`

Expected: FAIL.

- [ ] **Step 3: Append the macro to `anharmonic_dynaphopy.py`**

Append to `pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py`:

```python
import os

from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.phonons._compat import require_phonopy
from pyiron_workflow_atomistics.physics.phonons.harmonic import (
    _normalise_supercell_matrix,
)
from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
    calculate_phonon_md_renormalisation,
)


def _commensurate_q_points(structure: Atoms, q_mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_points, weights) on a commensurate Monkhorst-Pack mesh.

    Phonopy's `GridPoints` lays out the mesh on the *primitive* cell that
    phonopy infers via primitive_matrix="auto". We mirror that convention
    here so the mesh is consistent with the FC2 view dynaphopy projects into.
    Weights sum to 1.
    """
    require_phonopy()
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=list(structure.get_chemical_symbols()),
        positions=structure.get_positions(),
        cell=np.asarray(structure.get_cell()),
        masses=structure.get_masses(),
    )
    phonon = phonopy.Phonopy(
        unitcell=unitcell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
    )
    phonon.run_mesh(mesh=list(q_mesh), is_mesh_symmetry=True)
    md = phonon.get_mesh_dict()
    q_points = np.asarray(md["qpoints"])
    weights = np.asarray(md["weights"], dtype=float)
    weights = weights / weights.sum()
    return q_points, weights


@pwf.as_function_node("free_energy_output")
def _pack_anharmonic_dynaphopy_output(
    structure: Atoms,
    md_phonon_output,
    q_weights: np.ndarray,
    free_energy_per_atom: float,
    entropy_per_atom: float,
    cv_per_atom: float,
    temperature: float,
    q_mesh: tuple[int, int, int],
    simfolder: str,
    keep_handles: bool,
) -> FreeEnergyOutput:
    healthy, issues = md_phonon_output.check_md_health()
    elements = list(dict.fromkeys(structure.get_chemical_symbols()))
    return FreeEnergyOutput(
        mode="anharmonic_dynaphopy",
        reference_phase="solid",
        free_energy=float(free_energy_per_atom),
        free_energy_error=0.0,
        temperature=float(temperature),
        pressure=0.0,
        n_atoms=len(structure),
        elements=elements,
        simfolder=simfolder,
        report={
            "method": "anharmonic_dynaphopy",
            "n_md_steps": int(md_phonon_output.n_md_steps),
            "time_step_fs": float(md_phonon_output.time_step_fs),
            "md_temperature_mean": float(md_phonon_output.md_temperature_mean),
            "md_temperature_std": float(md_phonon_output.md_temperature_std),
            "md_health": {"healthy": bool(healthy), "issues": list(issues)},
        },
        entropy=float(entropy_per_atom),
        heat_capacity=float(cv_per_atom),
        harmonic_frequencies=np.asarray(md_phonon_output.harmonic_frequencies),
        renormalised_frequencies=np.asarray(
            md_phonon_output.renormalised_frequencies
        ),
        linewidths=np.asarray(md_phonon_output.linewidths),
        q_mesh=tuple(int(x) for x in q_mesh),
        dynaphopy_handle=(
            md_phonon_output.quasiparticle if keep_handles else None
        ),
        phonopy_handle=md_phonon_output.phonopy if keep_handles else None,
    )


@pwf.api.as_macro_node("free_energy_output")
def anharmonic_free_energy_dynaphopy(
    wf,
    *,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperature: float,
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,
    thermostat_time_constant: float = 100.0,
    seed: int | None = None,
    q_mesh=(11, 11, 11),
    phono3py_output=None,
    working_directory: str = ".",
    subdir: str = "anharmonic_free_energy_dynaphopy",
    keep_handles: bool = False,
):
    """Anharmonic free energy at one T via dynaphopy MD projection + harmonic-formula sum.

    Engine must expose ``.calculator`` (inherited from
    ``calculate_phonon_md_renormalisation``).
    """
    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)
    sub_engine = engine.with_working_directory(simfolder)

    q_points, q_weights = _commensurate_q_points(structure, q_mesh)

    wf.md_renorm = calculate_phonon_md_renormalisation(
        structure=structure,
        engine=sub_engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperature=temperature,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        time_step=time_step,
        thermostat_time_constant=thermostat_time_constant,
        seed=seed,
        q_points=q_points,
        phono3py_output=phono3py_output,
        power_spectra=False,
        keep_handles=True,  # we need .phonopy / .quasiparticle handles to extract data
    )
    wf.spectrum = _free_energy_from_spectrum(
        frequencies=wf.md_renorm.outputs.md_phonon_output.renormalised_frequencies,
        q_weights=q_weights,
        temperature=temperature,
        n_atoms_primitive=len(structure),  # primitive equals unitcell for fcc cubic; OK for v1
    )
    wf.synthesis = _pack_anharmonic_dynaphopy_output(
        structure=structure,
        md_phonon_output=wf.md_renorm.outputs.md_phonon_output,
        q_weights=q_weights,
        free_energy_per_atom=wf.spectrum.outputs.free_energy_per_atom,
        entropy_per_atom=wf.spectrum.outputs.entropy_per_atom,
        cv_per_atom=wf.spectrum.outputs.cv_per_atom,
        temperature=temperature,
        q_mesh=q_mesh,
        simfolder=simfolder,
        keep_handles=keep_handles,
    )
    return wf.synthesis.outputs.free_energy_output
```

Note: the `n_atoms_primitive=len(structure)` value is correct only when `structure` already *is* the primitive cell. For non-primitive inputs, phonopy would infer a smaller primitive automatically — but using `len(structure)` here keeps the units consistent because `_compute_harmonic_observables` upstream produces F per *unitcell* atom, not per *primitive* atom. The dynaphopy `renormalised_frequencies` shape `(n_q, 3*n_atoms_unitcell)` is per-unitcell-atom; dividing by `len(structure)` gives per-atom-of-input-cell. Documented in the function docstring.

- [ ] **Step 4: Run the EMT smoke; verify pass**

Run: `pytest tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_anharmonic_free_energy_dynaphopy_emt_al -v -m slow`

Expected: PASS. Runtime ~3-5 min (FC2 + MD + projection).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py
git commit -m "feat(free_energy): add anharmonic_free_energy_dynaphopy macro"
```

---

## Task 7 — `anharmonic_free_energy_dynaphopy_tdi` macro (T grid)

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py`
- Modify: `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py`

- [ ] **Step 1: Add failing unit test for `_stack_tdi_outputs` + Tier-2 EMT smoke**

Append to `tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py`:

```python
def test_stack_tdi_outputs_central_differences():
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        _stack_tdi_outputs,
    )
    from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

    # Synthetic F(T) = a + b T + c T^2  → S = -∂F/∂T = -(b + 2cT)
    a, b, c = -3.0, -1e-4, -1e-7
    Ts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    Fs = a + b * Ts + c * Ts**2
    per_T = [
        FreeEnergyOutput(
            mode="anharmonic_dynaphopy",
            reference_phase="solid",
            free_energy=float(F),
            free_energy_error=0.0,
            temperature=float(T),
            pressure=0.0,
            n_atoms=4,
            elements=["Al"],
            simfolder="/tmp",
            report={},
            harmonic_frequencies=np.zeros((1, 12)),
            renormalised_frequencies=np.zeros((1, 12)),
            linewidths=np.zeros((1, 12)),
            q_mesh=(7, 7, 7),
        )
        for T, F in zip(Ts, Fs)
    ]
    structure = type("FakeAtoms", (), {"__len__": lambda self: 4})()
    out = _stack_tdi_outputs.node_function(
        per_T_outputs=per_T,
        structure=structure,
        temperatures=Ts,
    )
    # interior central-difference S at T=300:  -(b + 2 c * 300)
    expected_S_300 = -(b + 2 * c * 300.0)
    assert out.entropy_array[2] == pytest.approx(expected_S_300, rel=5e-2)
    assert out.renormalised_frequencies_per_T.shape == (5, 1, 12)


@pytest.mark.slow
def test_anharmonic_free_energy_dynaphopy_tdi_emt_al(tmp_path):
    pytest.importorskip("dynaphopy", reason="dynaphopy not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
        anharmonic_free_energy_dynaphopy_tdi,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    out = anharmonic_free_energy_dynaphopy_tdi(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(200.0, 400.0),
        production_steps=2000,
        q_mesh=(5, 5, 5),
        working_directory=str(tmp_path),
        subdir="anharmonic_tdi",
    ).run()

    assert out.mode == "anharmonic_dynaphopy_tdi"
    assert out.temperature_array.shape == (2,)
    assert out.free_energy_array.shape == (2,)
    assert out.renormalised_frequencies_per_T.shape == (2, *out.renormalised_frequencies_per_T.shape[1:])
```

- [ ] **Step 2: Run both new tests; confirm failure**

Run: `pytest tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_stack_tdi_outputs_central_differences tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_anharmonic_free_energy_dynaphopy_tdi_emt_al -v`

Expected: both FAIL.

- [ ] **Step 3: Append `_stack_tdi_outputs` and the TDI macro**

Append to `pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py`:

```python
@pwf.as_function_node("free_energy_output")
def _stack_tdi_outputs(
    per_T_outputs: list,
    structure,
    temperatures,
) -> FreeEnergyOutput:
    """Aggregate independent-T dynaphopy free energies into one FreeEnergyOutput."""
    T = np.asarray(temperatures, dtype=float)
    F = np.asarray([o.free_energy for o in per_T_outputs], dtype=float)
    n_T = T.size

    if n_T >= 2:
        dF_dT = np.gradient(F, T)
        S = -dF_dT
    else:
        S = np.full(n_T, np.nan)
    if n_T >= 3:
        d2F_dT2 = np.gradient(dF_dT, T)
        Cv = -T * d2F_dT2
    else:
        Cv = np.full(n_T, np.nan)

    elements = list(dict.fromkeys(structure.get_chemical_symbols())) if hasattr(
        structure, "get_chemical_symbols"
    ) else ["?"]
    renorm = np.stack(
        [np.asarray(o.renormalised_frequencies) for o in per_T_outputs], axis=0
    )
    lw = np.stack([np.asarray(o.linewidths) for o in per_T_outputs], axis=0)

    derivative_warning = n_T < 3
    return FreeEnergyOutput(
        mode="anharmonic_dynaphopy_tdi",
        reference_phase="solid",
        free_energy=float(F[0]),
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=0.0,
        n_atoms=len(structure),
        elements=elements,
        simfolder=per_T_outputs[0].simfolder if per_T_outputs else "",
        report={
            "method": "anharmonic_dynaphopy_tdi",
            "derivative_warning": bool(derivative_warning),
            "per_T_md_health": [o.report.get("md_health") for o in per_T_outputs],
        },
        temperature_array=T,
        free_energy_array=F,
        entropy_array=S,
        heat_capacity_array=Cv,
        renormalised_frequencies_per_T=renorm,
        linewidths_per_T=lw,
    )


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
):
    """Anharmonic F_anharm(T) on a T grid — renormalised-harmonic at each T.

    See spec ``docs/design/specs/2026-05-15-free-energy-consolidation-design.md``
    (renormalised-harmonic-over-T, not full ⟨∂H/∂λ⟩ TI).
    """
    simfolder = os.path.abspath(os.path.join(working_directory, subdir))
    os.makedirs(simfolder, exist_ok=True)

    per_T_outputs: list = []
    for i, T in enumerate(temperatures):
        sub_wf = anharmonic_free_energy_dynaphopy(
            structure=structure,
            engine=engine,
            fc2_supercell_matrix=fc2_supercell_matrix,
            temperature=float(T),
            equilibration_steps=equilibration_steps,
            production_steps=production_steps,
            time_step=time_step,
            thermostat_time_constant=thermostat_time_constant,
            seed=(None if seed is None else seed + i),
            q_mesh=q_mesh,
            working_directory=simfolder,
            subdir=f"T_{i:03d}_{T:.1f}K",
            keep_handles=False,
        )
        per_T_outputs.append(sub_wf.run())

    setattr(wf, "stack", _stack_tdi_outputs(
        per_T_outputs=per_T_outputs,
        structure=structure,
        temperatures=temperatures,
    ))
    return wf.stack.outputs.free_energy_output
```

- [ ] **Step 4: Run both tests; verify pass**

Run: `pytest tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_stack_tdi_outputs_central_differences tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py::test_anharmonic_free_energy_dynaphopy_tdi_emt_al -v`

Expected: both PASS. The slow test runs ~6-10 min (2 × single-T cost).

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/anharmonic_dynaphopy.py tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py
git commit -m "feat(free_energy): add anharmonic_free_energy_dynaphopy_tdi macro"
```

---

## Task 8 — Public exports

**Files:**
- Modify: `pyiron_workflow_atomistics/physics/free_energy/__init__.py`
- Test: `tests/unit/physics/test_free_energy.py`

- [ ] **Step 1: Write failing import-surface test**

Append to `tests/unit/physics/test_free_energy.py`:

```python
def test_public_free_energy_exports():
    import pyiron_workflow_atomistics.physics.free_energy as fe

    expected = {
        "FreeEnergyOutput",
        "LammpsPotential",
        "alchemy",
        "composition_scaling",
        "free_energy",
        "melting_temperature",
        "reversible_scaling_pressure",
        "reversible_scaling_temperature",
        "harmonic_free_energy",
        "quasiharmonic_free_energy",
        "anharmonic_free_energy_dynaphopy",
        "anharmonic_free_energy_dynaphopy_tdi",
    }
    assert expected.issubset(set(fe.__all__))
    for name in expected:
        assert hasattr(fe, name), f"missing public export: {name}"
```

- [ ] **Step 2: Run; confirm failure**

Run: `pytest tests/unit/physics/test_free_energy.py::test_public_free_energy_exports -v`

Expected: FAIL.

- [ ] **Step 3: Update `__init__.py`**

Replace `pyiron_workflow_atomistics/physics/free_energy/__init__.py` with:

```python
"""Free-energy workflows: calphy + phonopy harmonic / QHA + dynaphopy anharmonic.

Public API
----------
Dataclasses:
    LammpsPotential  - calphy-only: pair_style + pair_coeff + optional potential_file
    FreeEnergyOutput - typed result of every node

calphy function-nodes (one per mode):
    free_energy, reversible_scaling_temperature, reversible_scaling_pressure,
    melting_temperature, alchemy, composition_scaling

Phonon free-energy macros (NEW):
    harmonic_free_energy             - phonopy FC2 at a fixed volume
    quasiharmonic_free_energy        - phonopy.qha.QHA on top of harmonic_free_energy
    anharmonic_free_energy_dynaphopy - dynaphopy renormalised harmonic at one T
    anharmonic_free_energy_dynaphopy_tdi - dynaphopy renormalised harmonic over a T grid

All node-and-adapter imports defer ``calphy`` / ``phonopy`` / ``dynaphopy`` /
``pyiron_workflow_lammps`` imports to node-body call time, so importing this
subpackage does not require any specific optional extra.
"""

from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
    anharmonic_free_energy_dynaphopy,
    anharmonic_free_energy_dynaphopy_tdi,
)
from pyiron_workflow_atomistics.physics.free_energy.calphy import (
    alchemy,
    composition_scaling,
    free_energy,
    melting_temperature,
    reversible_scaling_pressure,
    reversible_scaling_temperature,
)
from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
    harmonic_free_energy,
)
from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
    quasiharmonic_free_energy,
)

__all__ = [
    "FreeEnergyOutput",
    "LammpsPotential",
    "alchemy",
    "anharmonic_free_energy_dynaphopy",
    "anharmonic_free_energy_dynaphopy_tdi",
    "composition_scaling",
    "free_energy",
    "harmonic_free_energy",
    "melting_temperature",
    "quasiharmonic_free_energy",
    "reversible_scaling_pressure",
    "reversible_scaling_temperature",
]
```

- [ ] **Step 4: Run; verify pass**

Run: `pytest tests/unit/physics/test_free_energy.py::test_public_free_energy_exports -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_atomistics/physics/free_energy/__init__.py tests/unit/physics/test_free_energy.py
git commit -m "feat(free_energy): expose harmonic/QHA/dynaphopy entry points from the package"
```

---

## Task 9 — Notebook `free_energy_solid.ipynb`

**Files:**
- Create: `notebooks/free_energy_solid.ipynb`

- [ ] **Step 1: Create the notebook skeleton**

Run:

```bash
cd /home/liger/pyiron_workflow_atomistics/notebooks
jupyter nbconvert --to notebook --new free_energy_solid.ipynb 2>/dev/null || true
```

(If `jupyter nbconvert --new` is unavailable in your jupyter version, create an empty notebook with `jupyter notebook free_energy_solid.ipynb` and immediately close it, or write `{"cells":[],"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}` to that path.)

- [ ] **Step 2: Author the seven sections**

Open `notebooks/free_energy_solid.ipynb` and add cells in this order. Each markdown header is a separate markdown cell; each code block is a separate code cell.

```markdown
# Free energy of a solid: harmonic vs quasiharmonic vs anharmonic

This notebook computes the free energy of fcc Al with four methods:

1. **Harmonic** — phonopy FC2 at the reference volume
2. **Quasiharmonic** — phonopy.qha.QHA across a volume sweep
3. **Anharmonic (dynaphopy)** — MD-projected renormalised harmonic over a T grid
4. **Anharmonic (calphy)** — Frenkel–Ladd thermodynamic integration

All four are entry points under `pyiron_workflow_atomistics.physics.free_energy`.
```

```python
# Section 1 — Setup
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic

structure = bulk("Al", "fcc", a=4.05, cubic=True)
ase_engine = ASEEngine(
    EngineInput=CalcInputStatic(),
    calculator=EMT(),
    working_directory="_runs",
)
fc2_sc = 2 * np.eye(3, dtype=int)
T_grid = np.arange(0, 1001, 50)
```

```python
# Section 2 — Harmonic
from pyiron_workflow_atomistics.physics.free_energy import harmonic_free_energy

out_harm = harmonic_free_energy(
    structure=structure,
    engine=ase_engine,
    fc2_supercell_matrix=fc2_sc,
    temperatures=T_grid,
    working_directory="_runs",
    subdir="harmonic",
).run()

import matplotlib.pyplot as plt
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.5))
a1.plot(out_harm.temperature_array, out_harm.free_energy_array)
a1.set_xlabel("T (K)"); a1.set_ylabel("F (eV/atom)"); a1.set_title("Harmonic F(T)")
a2.plot(out_harm.temperature_array, out_harm.entropy_array)
a2.set_xlabel("T (K)"); a2.set_ylabel("S (eV/K/atom)"); a2.set_title("S(T)")
a3.plot(out_harm.temperature_array, out_harm.heat_capacity_array)
a3.set_xlabel("T (K)"); a3.set_ylabel("Cv (eV/K/atom)"); a3.set_title("Cv(T)")
plt.tight_layout(); plt.show()
```

```python
# Section 3 — Quasiharmonic (QHA)
from pyiron_workflow_atomistics.physics.free_energy import quasiharmonic_free_energy

out_qha = quasiharmonic_free_energy(
    structure=structure,
    engine=ase_engine,
    fc2_supercell_matrix=fc2_sc,
    temperatures=T_grid,
    strain_range=(-0.03, 0.03),
    num_volumes=7,
    pressure=0.0,   # GPa
    working_directory="_runs",
    subdir="qha",
).run()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
a1.plot(out_qha.temperature_array, out_qha.equilibrium_volume_array)
a1.set_xlabel("T (K)"); a1.set_ylabel("V* (Å³/atom)"); a1.set_title("QHA V*(T)")
a2.plot(out_harm.temperature_array, out_harm.free_energy_array, label="harmonic")
a2.plot(out_qha.temperature_array, out_qha.gibbs_free_energy_array, label="QHA G(T,P=0)")
a2.set_xlabel("T (K)"); a2.set_ylabel("F or G (eV/atom)"); a2.legend()
plt.tight_layout(); plt.show()
```

```python
# Section 4 — Anharmonic (dynaphopy) — single T
from pyiron_workflow_atomistics.physics.free_energy import (
    anharmonic_free_energy_dynaphopy,
)

out_anh_T = anharmonic_free_energy_dynaphopy(
    structure=structure,
    engine=ase_engine,
    fc2_supercell_matrix=fc2_sc,
    temperature=300.0,
    production_steps=5_000,   # bump to ≥30_000 for production
    q_mesh=(7, 7, 7),         # bump to (11,11,11) or denser
    working_directory="_runs",
    subdir="anharmonic_T300",
).run()

print(f"F_anharm(300 K) = {out_anh_T.free_energy:.4f} eV/atom")
print(f"F_harmonic(300 K) = {np.interp(300.0, out_harm.temperature_array, out_harm.free_energy_array):.4f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(out_anh_T.harmonic_frequencies.ravel(), out_anh_T.renormalised_frequencies.ravel(), "o", ms=3)
lim = (
    min(out_anh_T.harmonic_frequencies.min(), out_anh_T.renormalised_frequencies.min()),
    max(out_anh_T.harmonic_frequencies.max(), out_anh_T.renormalised_frequencies.max()),
)
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("ω_harmonic (THz)"); ax.set_ylabel("ω_renorm (THz)")
ax.set_title("Dynaphopy mode renormalisation at 300 K")
plt.tight_layout(); plt.show()
```

```python
# Section 5 — Anharmonic (dynaphopy) — TDI over T grid
from pyiron_workflow_atomistics.physics.free_energy import (
    anharmonic_free_energy_dynaphopy_tdi,
)

out_anh_tdi = anharmonic_free_energy_dynaphopy_tdi(
    structure=structure,
    engine=ase_engine,
    fc2_supercell_matrix=fc2_sc,
    temperatures=(200, 400, 600, 800),
    production_steps=5_000,
    q_mesh=(7, 7, 7),
    working_directory="_runs",
    subdir="anharmonic_tdi",
).run()
```

```python
# Section 6 — Anharmonic (calphy TI) — needs [free-energy] extras + lmp on PATH
try:
    from pyiron_workflow_lammps.engine import LammpsEngine
    from pyiron_workflow_atomistics.physics.free_energy import (
        LammpsPotential,
        reversible_scaling_temperature,
    )
    lammps_engine = LammpsEngine(command="lmp")
    potential = LammpsPotential(
        pair_style="eam/alloy",
        pair_coeff="* * Al99.eam.alloy Al",
        potential_file="Al99.eam.alloy",
    )
    out_calphy = reversible_scaling_temperature(
        structure=structure,
        lammps_engine=lammps_engine,
        potential=potential,
        temperature_range=(100.0, 800.0),
        reference_phase="solid",
        working_directory="_runs",
        subdir="calphy_ts",
    ).run()
except Exception as exc:
    print(f"calphy step skipped: {exc}")
    out_calphy = None
```

```python
# Section 7 — Composite overlay
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(out_harm.temperature_array, out_harm.free_energy_array, label="harmonic")
ax.plot(out_qha.temperature_array, out_qha.gibbs_free_energy_array, label="QHA")
ax.plot(out_anh_tdi.temperature_array, out_anh_tdi.free_energy_array, "o-", label="dynaphopy TDI")
if out_calphy is not None and out_calphy.temperature_array is not None:
    ax.plot(out_calphy.temperature_array, out_calphy.free_energy_array, label="calphy TI")
ax.set_xlabel("T (K)"); ax.set_ylabel("F or G (eV/atom)")
ax.legend(); ax.set_title("Free energy of fcc Al — four methods")
plt.tight_layout(); plt.show()
```

```markdown
## Notes

- **Harmonic** has a fixed reference volume — it overestimates F at high T because the lattice cannot expand.
- **Quasiharmonic** captures thermal expansion → typically falls below harmonic at high T.
- **Dynaphopy renormalised** captures soft-mode renormalisation (classical, fixed-V here — combine with QHA in a follow-up).
- **Calphy TI** is the reference: full anharmonicity, classical, most expensive.

For production-grade Al, bump `production_steps` to ≥30 000, `q_mesh` to (11, 11, 11) or denser, the supercell to 3×3×3 or 4×4×4, and `num_volumes` to ≥9.
```

- [ ] **Step 3: Run the notebook end-to-end (smoke)**

```bash
jupyter nbconvert --to notebook --execute notebooks/free_energy_solid.ipynb --output free_energy_solid.executed.ipynb --ExecutePreprocessor.timeout=900
```

Expected: notebook executes without errors. Calphy section may print a skip message if the `[free-energy]` extra or `lmp` is not installed. Delete the `.executed.ipynb` byproduct after verification — do not commit it.

```bash
rm notebooks/free_energy_solid.executed.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/free_energy_solid.ipynb
git commit -m "docs(notebooks): free-energy of a solid — harmonic / QHA / dynaphopy / calphy overlay"
```

---

## Task 10 — Final cross-check pass

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/unit/physics/test_free_energy.py \
       tests/unit/physics/test_free_energy_harmonic.py \
       tests/unit/physics/test_free_energy_qha.py \
       tests/unit/physics/test_free_energy_anharmonic_dynaphopy.py \
       -v
```

Expected: every test PASSES, including the four `@pytest.mark.slow` Tier-2 EMT smokes. Total runtime budget ≤15 min.

- [ ] **Step 2: Run the non-free-energy phonons tests to check no regressions**

```bash
pytest tests/unit/physics/test_phonons.py tests/unit/physics/test_phonons_helpers.py tests/unit/physics/test_bulk_workflows.py -v
```

Expected: no regressions. The phonons modules were not modified; this is a paranoia check.

- [ ] **Step 3: Lint / style check**

```bash
ruff check pyiron_workflow_atomistics/physics/free_energy/
ruff format --check pyiron_workflow_atomistics/physics/free_energy/ tests/unit/physics/test_free_energy_*.py
```

Expected: clean. Fix any reported issues and amend the relevant commit (do **not** use `git commit --amend` for already-merged commits — squash via a follow-up commit instead).

- [ ] **Step 4: Final commit (only if any lint fixes were needed)**

```bash
git add -A
git commit -m "style: ruff format pass on free-energy consolidation"
```

---

## Self-review

**1. Spec coverage:**
- `FreeEnergyOutput` extension → Task 1 ✓
- `_free_energy_from_spectrum` → Task 2 ✓
- `_produce_fc2_view` + `harmonic_free_energy` → Task 3 ✓
- `_fit_qha` + `_check_qha_volume_range` → Task 4 ✓
- `quasiharmonic_free_energy` → Task 5 ✓
- `anharmonic_free_energy_dynaphopy` + `_commensurate_q_points` + `_pack_anharmonic_dynaphopy_output` → Task 6 ✓
- `anharmonic_free_energy_dynaphopy_tdi` + `_stack_tdi_outputs` → Task 7 ✓
- `__init__.py` public exports → Task 8 ✓
- Demo notebook `free_energy_solid.ipynb` → Task 9 ✓
- Tests: per-method unit + Tier-2 EMT smokes ✓ (Tasks 1, 2, 3, 4, 5, 6, 7, 8)

**2. Placeholder scan:** Every step has concrete code blocks or commands. No "TODO" / "TBD" / "implement later". One legitimate non-placeholder is the calphy section in the notebook (Section 6), which falls back gracefully if extras are missing — that's intended behaviour, not a placeholder.

**3. Type consistency:**
- `FreeEnergyOutput.mode` Literal extended in Task 1 includes exactly the four new values used in Tasks 3, 5, 6, 7.
- `_free_energy_from_spectrum` returns `(F, S, Cv)` in Task 2 — consumed with that triple unpack by Task 6's `wf.spectrum.outputs.{free_energy_per_atom, entropy_per_atom, cv_per_atom}`.
- `_fit_qha` returns a dict in Task 4 — Task 5's `_pack_qha_output` reads the same keys.
- `_stack_tdi_outputs` populates `renormalised_frequencies_per_T` and `linewidths_per_T` — fields declared in Task 1's dataclass.
- Engine constraint string ("must expose `.calculator`") is inherited from `calculate_phonon_md_renormalisation` — not re-validated in the new macros to avoid double-error-message noise.

---

## Execution handoff

**Plan complete and saved to `docs/design/plans/2026-05-15-free-energy-consolidation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
