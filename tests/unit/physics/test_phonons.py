"""Tests for pyiron_workflow_atomistics.physics.phonons.

Tier 1 — cheap unit tests, run always (no phono3py needed).
Tier 2 — gated on `pytest.importorskip("phono3py")`; cover the full graph.
Tier 3 — determinism checks (gated on phono3py when needed).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import numpy as np
import pytest
from ase.build import bulk

# ---------------------------------------------------------------------------
# Tier 1 — PhononOutput dataclass shape
# ---------------------------------------------------------------------------


def test_phonon_output_is_a_dataclass():
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    assert is_dataclass(PhononOutput)


def test_phonon_output_required_fields_have_no_default():
    """Required fields must be the dataclass's positional-required ones."""
    from dataclasses import MISSING

    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    required_names = {
        "structure",
        "fc2_supercell_matrix",
        "fc3_supercell_matrix",
        "temperatures",
        "kappa",
        "converged",
    }
    by_name = {f.name: f for f in fields(PhononOutput)}
    for name in required_names:
        f = by_name[name]
        assert (
            f.default is MISSING and f.default_factory is MISSING
        ), f"{name} must be required (no default)"


def test_phonon_output_optional_fields_default_to_none():
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    optional_names = {
        "q_points",
        "frequencies",
        "group_velocities",
        "mode_kappa",
        "gamma",
        "gruneisen",
        "band_structure",
        "dos",
        "free_energy",
        "fc2",
        "fc3",
        "phono3py",
    }
    by_name = {f.name: f for f in fields(PhononOutput)}
    for name in optional_names:
        assert by_name[name].default is None, f"{name} must default to None"


def test_phonon_output_to_dict_round_trip():
    """to_dict() returns plain dict of all fields (per EngineOutput convention)."""
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = PhononOutput(
        structure=cu,
        fc2_supercell_matrix=np.eye(3, dtype=int) * 2,
        fc3_supercell_matrix=np.eye(3, dtype=int) * 2,
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
    )
    d = out.to_dict()
    assert d["temperatures"][0] == 300.0
    assert d["kappa"].shape == (1, 3, 3)
    assert d["converged"] is True
    assert d["q_points"] is None


# ---------------------------------------------------------------------------
# Tier 1 — lazy-import shims
# ---------------------------------------------------------------------------


def _patch_missing(monkeypatch, module_name):
    """Make `import <module_name>` raise ImportError inside the shim."""
    import sys

    monkeypatch.setitem(sys.modules, module_name, None)


def test_require_phonopy_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "phonopy")
    with pytest.raises(ImportError) as exc:
        _compat.require_phonopy()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons]" in msg
    assert "phonopy" in msg


def test_require_phono3py_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "phono3py")
    with pytest.raises(ImportError) as exc:
        _compat.require_phono3py()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons]" in msg
    assert "phono3py" in msg


def test_require_symfc_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "symfc")
    with pytest.raises(ImportError) as exc:
        _compat.require_symfc()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons]" in msg
    assert "symfc" in msg


# ---------------------------------------------------------------------------
# Tier 1 — require_dynaphopy lazy-import shim
# ---------------------------------------------------------------------------


def test_require_dynaphopy_missing_actionable(monkeypatch):
    from pyiron_workflow_atomistics.physics.phonons import _compat

    _patch_missing(monkeypatch, "dynaphopy")
    with pytest.raises(ImportError) as exc:
        _compat.require_dynaphopy()
    msg = str(exc.value)
    assert "pip install pyiron_workflow_atomistics[phonons-md]" in msg
    assert "dynaphopy" in msg


# ---------------------------------------------------------------------------
# Tier 1 — polar-material kwargs early exit
# ---------------------------------------------------------------------------


def test_born_charges_raises_not_implemented():
    """Passing born_charges raises before any phono3py import."""
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError) as exc:
        _check_polar_unsupported(born_charges=np.zeros((4, 3, 3)), epsilon_inf=None)
    msg = str(exc.value)
    assert "BORN" in msg or "Non-analytic" in msg
    assert "v1" in msg


def test_epsilon_inf_raises_not_implemented():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError):
        _check_polar_unsupported(born_charges=None, epsilon_inf=np.eye(3))


def test_no_polar_kwargs_returns_silently():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    # Should return without raising
    _check_polar_unsupported(born_charges=None, epsilon_inf=None)


# ---------------------------------------------------------------------------
# Tier 3 — displacement generation determinism (gated)
# ---------------------------------------------------------------------------


def _cu_fcc_primitive():
    return bulk("Cu", "fcc", a=3.6)


def _two_by_two_by_two():
    return (2 * np.eye(3)).astype(int)


@pytest.mark.slow
def test_fd_fc2_supercells_deterministic():
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    a = _generate_fc2_supercells.node_function(
        structure=_cu_fcc_primitive(),
        fc2_supercell_matrix=_two_by_two_by_two(),
        displacement_distance=0.03,
        is_plusminus="auto",
    )
    b = _generate_fc2_supercells.node_function(
        structure=_cu_fcc_primitive(),
        fc2_supercell_matrix=_two_by_two_by_two(),
        displacement_distance=0.03,
        is_plusminus="auto",
    )
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.get_positions(), y.get_positions())
        np.testing.assert_allclose(x.get_cell()[:], y.get_cell()[:])


# ---------------------------------------------------------------------------
# Tier 1 — _evaluate_supercells using mock engine
# ---------------------------------------------------------------------------


def test_evaluate_supercells_uses_with_working_directory(tmp_path):
    """Each supercell gets its own engine subdir; the node returns one
    EngineOutput per input supercell."""
    from dataclasses import replace as dc_replace

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    supercells = [cu.copy(), cu.copy(), cu.copy()]
    outs = _evaluate_supercells.node_function(
        supercells=supercells, engine=engine, prefix="fc2_disp_"
    )
    assert len(outs) == 3
    assert all(o.converged for o in outs)
    # Each supercell got its own working_directory under tmp_path
    for i in range(3):
        assert (tmp_path / f"fc2_disp_{i:04d}").exists()


@pytest.mark.slow
def test_fd_fc3_supercells_deterministic():
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _generate_fc3_supercells,
    )

    kwargs = dict(
        structure=_cu_fcc_primitive(),
        fc2_supercell_matrix=_two_by_two_by_two(),
        fc3_supercell_matrix=_two_by_two_by_two(),
        displacement_distance=0.03,
        is_plusminus="auto",
        cutoff_pair_distance=None,
        number_of_snapshots=None,
        random_seed=None,
    )
    a = _generate_fc3_supercells.node_function(**kwargs)
    b = _generate_fc3_supercells.node_function(**kwargs)
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.get_positions(), y.get_positions())


# ---------------------------------------------------------------------------
# Tier 2 — synthesis-node smoke (EMT-Cu, ~60s)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_phono3py_thermal_conductivity_emt_smoke(tmp_path):
    """End-to-end smoke through the synthesis node only (skipping macro plumbing).

    Calls the FC2/FC3 generation + evaluate nodes manually, hands the
    EngineOutputs to the synthesis node, and asserts a sensible
    PhononOutput comes out. ~60s with EMT.
    """
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
        _generate_fc3_supercells,
        _run_phono3py_thermal_conductivity,
    )
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )

    fc2_supercells = _generate_fc2_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc
    )
    fc3_supercells = _generate_fc3_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc, fc3_supercell_matrix=sc
    )
    fc2_outs = _evaluate_supercells.node_function(
        supercells=fc2_supercells, engine=engine, prefix="fc2_disp_"
    )
    fc3_outs = _evaluate_supercells.node_function(
        supercells=fc3_supercells, engine=engine, prefix="fc3_disp_"
    )
    out = _run_phono3py_thermal_conductivity.node_function(
        structure=cu,
        fc2_supercell_matrix=sc,
        fc3_supercell_matrix=sc,
        displacement_distance=0.03,
        is_plusminus="auto",
        cutoff_pair_distance=None,
        number_of_snapshots=None,
        random_seed=None,
        fc_calculator=None,
        fc2_engine_outputs=fc2_outs,
        fc3_engine_outputs=fc3_outs,
        temperatures=np.array([300.0]),
        q_mesh=(5, 5, 5),
        mode_resolved=False,
        harmonic_observables=False,
        keep_handles=False,
    )

    assert out.converged is True
    assert out.kappa.shape == (1, 3, 3)
    # Diagonal κ should be positive (BTE), trace > 0.
    diag = np.array([out.kappa[0, i, i] for i in range(3)])
    assert (diag > 0).all(), f"Non-positive diagonal κ: {diag}"
    # Optional fields all None at this tier.
    assert out.q_points is None
    assert out.band_structure is None
    assert out.fc2 is None


# ---------------------------------------------------------------------------
# Tier 2 — error guards (gated; need phono3py for _build_phono3py to work)
# ---------------------------------------------------------------------------


def _make_fake_engine_output(*, converged: bool, n_atoms: int = 32):
    """Minimal EngineOutput-shaped object for testing the synthesis-node guards."""
    from pyiron_workflow_atomistics.engine import EngineOutput

    return EngineOutput(
        final_structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        final_energy=-1.0,
        converged=converged,
        final_forces=np.zeros((n_atoms, 3)),
    )


@pytest.mark.slow
def test_synthesis_raises_when_force_calc_failed():
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _run_phono3py_thermal_conductivity,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)

    # Build minimal fake FC2 / FC3 force lists with one failed entry at index 3.
    fc2_outs = [_make_fake_engine_output(converged=(i != 3)) for i in range(6)]
    fc3_outs = [_make_fake_engine_output(converged=True) for _ in range(2)]

    with pytest.raises(RuntimeError) as exc:
        _run_phono3py_thermal_conductivity.node_function(
            structure=cu,
            fc2_supercell_matrix=sc,
            fc3_supercell_matrix=sc,
            displacement_distance=0.03,
            is_plusminus="auto",
            cutoff_pair_distance=None,
            number_of_snapshots=None,
            random_seed=None,
            fc_calculator=None,
            fc2_engine_outputs=fc2_outs,
            fc3_engine_outputs=fc3_outs,
            temperatures=np.array([300.0]),
            q_mesh=(5, 5, 5),
            mode_resolved=False,
            harmonic_observables=False,
            keep_handles=False,
        )
    msg = str(exc.value)
    assert "FC2" in msg
    assert "3" in msg  # the failed index


@pytest.mark.slow
def test_synthesis_raises_on_supercell_force_mismatch():
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _evaluate_supercells,
        _generate_fc3_supercells,
        _run_phono3py_thermal_conductivity,
    )
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)

    fc2_supercells = _generate_fc2_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc
    )
    fc3_supercells = _generate_fc3_supercells.node_function(
        structure=cu, fc2_supercell_matrix=sc, fc3_supercell_matrix=sc
    )
    n_fc2 = len(fc2_supercells[0])
    n_fc3 = len(fc3_supercells[0])
    fc2_outs = [
        _make_fake_engine_output(converged=True, n_atoms=n_fc2)
        for _ in range(len(fc2_supercells) - 1)  # ← one too few
    ]
    fc3_outs = [
        _make_fake_engine_output(converged=True, n_atoms=n_fc3) for _ in fc3_supercells
    ]

    with pytest.raises(RuntimeError) as exc:
        _run_phono3py_thermal_conductivity.node_function(
            structure=cu,
            fc2_supercell_matrix=sc,
            fc3_supercell_matrix=sc,
            displacement_distance=0.03,
            is_plusminus="auto",
            cutoff_pair_distance=None,
            number_of_snapshots=None,
            random_seed=None,
            fc_calculator=None,
            fc2_engine_outputs=fc2_outs,
            fc3_engine_outputs=fc3_outs,
            temperatures=np.array([300.0]),
            q_mesh=(5, 5, 5),
            mode_resolved=False,
            harmonic_observables=False,
            keep_handles=False,
        )
    msg = str(exc.value)
    assert "FC2 force/supercell mismatch" in msg
    assert str(len(fc2_supercells) - 1) in msg
    assert str(len(fc2_supercells)) in msg


# ---------------------------------------------------------------------------
# Tier 2 — full-macro smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calculate_phonon_thermal_conductivity_macro_emt(tmp_path):
    """End-to-end through the public macro. ~60s with EMT."""
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = (2 * np.eye(3)).astype(int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    out = calculate_phonon_thermal_conductivity(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=sc,
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
    ).run()

    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.converged is True
    assert out.kappa.shape == (1, 3, 3)
    # Engine got the per-supercell subdirs
    assert (tmp_path / "fc2_disp_0000").exists()
    assert (tmp_path / "fc3_disp_0000").exists()


# ---------------------------------------------------------------------------
# Tier 1 — seed auto-resolution helper
# ---------------------------------------------------------------------------


def test_resolve_random_seed_passthrough_when_explicit():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    assert _resolve_random_seed(number_of_snapshots=10, random_seed=42) == 42
    assert _resolve_random_seed(number_of_snapshots=None, random_seed=None) is None
    assert _resolve_random_seed(number_of_snapshots=None, random_seed=7) == 7


def test_resolve_random_seed_auto_fills_when_random_mode_without_seed():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    seed = _resolve_random_seed(number_of_snapshots=10, random_seed=None)
    assert seed is not None
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


# ---------------------------------------------------------------------------
# Tier 3 — random-displacement determinism (gated additionally on symfc)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_random_fc3_supercells_deterministic_with_seed():
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    pytest.importorskip("symfc", reason="symfc not installed")
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _generate_fc3_supercells,
    )

    kwargs = dict(
        structure=bulk("Cu", "fcc", a=3.6),
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        fc3_supercell_matrix=(2 * np.eye(3)).astype(int),
        displacement_distance=0.03,
        is_plusminus="auto",
        cutoff_pair_distance=None,
        number_of_snapshots=10,
        random_seed=42,
    )
    a = _generate_fc3_supercells.node_function(**kwargs)
    b = _generate_fc3_supercells.node_function(**kwargs)
    assert len(a) == len(b) == 10
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.get_positions(), y.get_positions())


# ---------------------------------------------------------------------------
# Tier 2 — random-mode end-to-end smoke (gated additionally on symfc)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_random_displacement_macro_emt(tmp_path):
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    pytest.importorskip("symfc", reason="symfc not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    # Al 3x3x3: symfc 1.7.0 requires ≥27-atom supercells for the FC3 O3 basis set;
    # the 2x2x2 FCC supercell (8 atoms) triggers an internal assertion failure in
    # symfc.utils.eig_tools_division that is fixed in later symfc releases.
    al = bulk("Al", "fcc", a=4.05)
    sc = (3 * np.eye(3)).astype(int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=al,
        engine=engine,
        fc2_supercell_matrix=sc,
        temperatures=[300.0],
        q_mesh=(3, 3, 3),
        number_of_snapshots=10,
        random_seed=0,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.converged is True
    assert np.all(np.isfinite(out.kappa))


# ---------------------------------------------------------------------------
# Tier 2 — output tiers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mode_resolved_off_by_default(tmp_path):
    """Without mode_resolved=True, all mode-resolved fields are None."""
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.q_points is None
    assert out.frequencies is None
    assert out.group_velocities is None
    assert out.mode_kappa is None
    assert out.gamma is None
    assert out.gruneisen is None


@pytest.mark.slow
def test_mode_resolved_on_populates_all_fields(tmp_path):
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
        mode_resolved=True,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out
    assert out.q_points is not None and out.q_points.shape[1] == 3
    assert out.frequencies is not None and out.frequencies.ndim == 2
    assert out.group_velocities is not None and out.group_velocities.shape[-1] == 3
    assert out.mode_kappa is not None
    assert out.gamma is not None
    n_q = out.q_points.shape[0]
    n_band = out.frequencies.shape[1]
    assert out.frequencies.shape == (n_q, n_band)
    assert out.mode_kappa.shape == (1, n_q, n_band, 6)


@pytest.mark.slow
def test_harmonic_observables_populates_bands_dos_freeenergy(tmp_path):
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0, 500.0],
        q_mesh=(5, 5, 5),
        harmonic_observables=True,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out

    assert out.band_structure is not None
    assert "q" in out.band_structure and "frequencies" in out.band_structure
    assert out.dos is not None
    assert out.dos["frequencies"].ndim == 1
    assert out.dos["dos"].shape == out.dos["frequencies"].shape
    assert out.free_energy is not None
    assert out.free_energy["temperatures"].shape == (2,)
    assert out.free_energy["F"].shape == (2,)
    assert out.free_energy["S"].shape == (2,)
    assert out.free_energy["Cv"].shape == (2,)


@pytest.mark.slow
def test_keep_handles_returns_fc2_fc3_and_phono3py_handle(tmp_path):
    pytest.importorskip("phono3py", reason="phonons extra not installed")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = calculate_phonon_thermal_conductivity(
        structure=bulk("Cu", "fcc", a=3.6),
        engine=engine,
        fc2_supercell_matrix=(2 * np.eye(3)).astype(int),
        temperatures=[300.0],
        q_mesh=(5, 5, 5),
        keep_handles=True,
    ).run()
    out = out["phonon_output"] if isinstance(out, dict) else out

    assert out.fc2 is not None
    assert out.fc2.ndim == 4 and out.fc2.shape[-1] == 3
    assert out.fc3 is not None
    assert out.fc3.ndim == 6 and out.fc3.shape[-1] == 3
    # phono3py handle is the live Phono3py object
    assert out.phono3py is not None
    assert hasattr(out.phono3py, "thermal_conductivity")


# ---------------------------------------------------------------------------
# Tier 1 — κ-solver non-convergence predicate
# ---------------------------------------------------------------------------


def test_kappa_convergence_predicate_matches_phono3py_message():
    """The predicate should flag phono3py's documented non-convergence text."""
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _is_kappa_not_converged,
    )

    assert (
        _is_kappa_not_converged(["Iteration is not converged.", "Other warning"])
        is True
    )
    assert _is_kappa_not_converged(["NOT CONVERGED in 100 iterations"]) is True
    assert _is_kappa_not_converged(["Successfully ran BTE"]) is False
    assert _is_kappa_not_converged([]) is False


# ---------------------------------------------------------------------------
# Tier 1 — public re-exports
# ---------------------------------------------------------------------------


def test_public_reexports():
    """All publicly-documented symbols are importable from the subpackage."""
    from pyiron_workflow_atomistics.physics.phonons import (
        PhononOutput,
        calculate_phonon_thermal_conductivity,
    )

    assert PhononOutput is not None
    assert callable(calculate_phonon_thermal_conductivity)


# ---------------------------------------------------------------------------
# Tier 1 — MdPhononOutput dataclass + check_md_health
# ---------------------------------------------------------------------------


def _make_md_output(
    *,
    temperature: float = 300.0,
    md_temperature_mean: float | None = None,
    md_temperature_std: float | None = None,
    n_atoms_supercell: int = 32,
):
    """Build a minimal MdPhononOutput for testing the dataclass shape + health checks."""
    from pyiron_workflow_atomistics.physics.phonons.output import MdPhononOutput

    if md_temperature_mean is None:
        md_temperature_mean = temperature
    if md_temperature_std is None:
        # Langevin expectation: T * sqrt(2 / (3 * N))
        md_temperature_std = temperature * np.sqrt(2.0 / (3.0 * n_atoms_supercell))

    cu = bulk("Cu", "fcc", a=3.6)
    return MdPhononOutput(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=temperature,
        q_points=np.zeros((1, 3)),
        harmonic_frequencies=np.array([[5.0, 5.0, 8.0]]),
        renormalised_frequencies=np.array([[4.9, 4.9, 7.8]]),
        linewidths=np.array([[0.1, 0.1, 0.2]]),
        converged=True,
        n_md_steps=2000,
        time_step_fs=1.0,
        md_temperature_mean=md_temperature_mean,
        md_temperature_std=md_temperature_std,
    )


def test_md_phonon_output_dataclass_shape():
    from dataclasses import MISSING, fields, is_dataclass

    from pyiron_workflow_atomistics.physics.phonons.output import MdPhononOutput

    assert is_dataclass(MdPhononOutput)

    required_names = {
        "structure",
        "fc2_supercell_matrix",
        "temperature",
        "q_points",
        "harmonic_frequencies",
        "renormalised_frequencies",
        "linewidths",
        "converged",
        "n_md_steps",
        "time_step_fs",
        "md_temperature_mean",
        "md_temperature_std",
    }
    optional_names = {
        "power_spectra",
        "frequency_grid",
        "quasiparticle",
        "dynamics",
        "phonopy",
    }
    by_name = {f.name: f for f in fields(MdPhononOutput)}
    for name in required_names:
        f = by_name[name]
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{name} must be required (no default)"
        )
    for name in optional_names:
        assert by_name[name].default is None, f"{name} must default to None"


def test_md_phonon_output_to_dict_round_trip():
    out = _make_md_output()
    d = out.to_dict()
    assert d["temperature"] == 300.0
    assert d["renormalised_frequencies"].shape == (1, 3)
    assert d["power_spectra"] is None


def test_check_md_health_passes_on_clean_run():
    out = _make_md_output()
    healthy, issues = out.check_md_health()
    assert healthy is True
    assert issues == []


def test_check_md_health_flags_temperature_drift():
    # 10% drift below requested
    out = _make_md_output(temperature=300.0, md_temperature_mean=270.0)
    healthy, issues = out.check_md_health()
    assert healthy is False
    assert any("drift" in i.lower() for i in issues)
    assert any("300" in i for i in issues)
    assert any("270" in i for i in issues)


def test_check_md_health_flags_anomalous_sigma():
    # σ_T way above Langevin expectation
    out = _make_md_output(temperature=300.0, md_temperature_std=200.0)
    healthy, issues = out.check_md_health()
    assert healthy is False
    assert any("σ" in i or "sigma" in i.lower() or "std" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# Tier 1 — _resolve_md_defaults argument coupling + auto-bandpath
# ---------------------------------------------------------------------------


def test_resolve_md_defaults_requires_at_least_one_fc2_source():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    with pytest.raises(ValueError) as exc:
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=None,
            phono3py_output=None,
            q_points=None,
            band_npoints=30,
            seed=42,
        )
    msg = str(exc.value)
    assert "fc2_supercell_matrix" in msg and "phono3py_output" in msg


def test_resolve_md_defaults_rejects_mismatched_supercells():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6)
    fake_phono3py_output = PhononOutput(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        fc3_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
        fc2=np.zeros((8, 8, 3, 3)),  # plausible FC2 shape for 2x2x2 Cu
    )

    with pytest.raises(ValueError) as exc:
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=3 * np.eye(3, dtype=int),  # MISMATCH
            phono3py_output=fake_phono3py_output,
            q_points=None,
            band_npoints=30,
            seed=42,
        )
    msg = str(exc.value)
    assert "disagree" in msg.lower() or "must match" in msg.lower()


def test_resolve_md_defaults_rejects_phono3py_output_without_fc2():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6)
    output_without_handles = PhononOutput(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        fc3_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
        # fc2 deliberately left None (i.e. keep_handles=False upstream)
    )

    with pytest.raises(ValueError) as exc:
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=None,
            phono3py_output=output_without_handles,
            q_points=None,
            band_npoints=30,
            seed=42,
        )
    msg = str(exc.value)
    assert "keep_handles=True" in msg
    assert "fc2" in msg.lower()


def test_resolve_md_defaults_auto_derives_band_path_when_qpoints_none():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    (
        resolved_fc2_supercell,
        resolved_q_points,
        resolved_seed,
        fc2_source_tag,
        fc2_array,
    ) = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=None,
        band_npoints=30,
        seed=42,
    )
    assert resolved_q_points.shape == (30, 3)
    assert fc2_source_tag == "recompute"
    assert fc2_array is None
    assert resolved_seed == 42


def test_resolve_md_defaults_band_path_is_deterministic():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    kwargs = dict(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=None,
        band_npoints=30,
        seed=None,  # auto-fill — but the q-points path should still match across calls
    )
    out_a = _resolve_md_defaults.node_function(**kwargs)
    out_b = _resolve_md_defaults.node_function(**kwargs)
    np.testing.assert_allclose(out_a[1], out_b[1])  # resolved_q_points identical


def test_resolve_md_defaults_passes_through_explicit_qpoints():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    user_q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    out = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=user_q,
        band_npoints=30,  # ignored because q_points is explicit
        seed=42,
    )
    np.testing.assert_allclose(out[1], user_q)


def test_resolve_md_defaults_seed_auto_filled_when_none():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    out = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        phono3py_output=None,
        q_points=np.zeros((1, 3)),
        band_npoints=30,
        seed=None,
    )
    resolved_seed = out[2]
    assert resolved_seed is not None
    assert isinstance(resolved_seed, int)
    assert 0 <= resolved_seed < 2**32


# ---------------------------------------------------------------------------
# Tier 1 — _multiplier_to_cell_vectors helper
# ---------------------------------------------------------------------------


def test_multiplier_to_cell_vectors_matches_ase_make_supercell():
    """The named helper must match ASE's `make_supercell(...).cell` exactly,
    so swapping the implicit ASE path for the explicit helper in
    `_run_nvt_trajectory` is a no-op on `trajectory_pack['supercell']`.
    """
    from ase.build import make_supercell

    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _multiplier_to_cell_vectors,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    multiplier = 2 * np.eye(3, dtype=int)

    expected = np.asarray(make_supercell(cu, multiplier).cell)
    got = _multiplier_to_cell_vectors(cu.cell, multiplier)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_multiplier_to_cell_vectors_accepts_1d_and_scalar_forms():
    """Helper composes with `_normalise_supercell_matrix` to accept the
    1d diag form, the (3,3) form, and the scalar-int form alike.
    """
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _multiplier_to_cell_vectors,
    )

    prim = np.array([[1.8, 1.8, 0.0], [0.0, 1.8, 1.8], [1.8, 0.0, 1.8]])
    from_scalar = _multiplier_to_cell_vectors(prim, 2)
    from_1d = _multiplier_to_cell_vectors(prim, [2, 2, 2])
    from_2d = _multiplier_to_cell_vectors(prim, 2 * np.eye(3, dtype=int))
    np.testing.assert_allclose(from_scalar, from_1d, atol=1e-12)
    np.testing.assert_allclose(from_1d, from_2d, atol=1e-12)
    # And it's the multiplier @ primitive convention (not primitive @ multiplier).
    expected = (2 * np.eye(3)).astype(float) @ prim
    np.testing.assert_allclose(from_scalar, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Tier 2 — _compute_fc2_from_scratch (gated on phonopy)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_fc2_from_scratch_produces_correct_shape(tmp_path):
    pytest.importorskip("phonopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _compute_fc2_from_scratch,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    fc2 = _compute_fc2_from_scratch.node_function(
        structure=cu,
        engine=engine,
        resolved_fc2_supercell=2 * np.eye(3, dtype=int),
    )
    # 2x2x2 of a Cu primitive (1 atom) → 8 supercell atoms
    assert fc2.shape == (8, 8, 3, 3)
    assert np.all(np.isfinite(fc2))
    # The fc2_disp_* directories should exist on disk
    assert (tmp_path / "fc2_disp_0000").exists()


# ---------------------------------------------------------------------------
# Tier 2 — _run_nvt_trajectory smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_nvt_trajectory_returns_expected_pack_shape(tmp_path):
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMD
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _run_nvt_trajectory,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputMD(),  # ignored; the node builds its own
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    pack = _run_nvt_trajectory.node_function(
        structure=cu,
        engine=engine,
        resolved_fc2_supercell=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=300,
        time_step=1.0,
        thermostat_time_constant=100.0,
        seed=42,
    )

    n_supercell_atoms = 8  # 2x2x2 of Cu FCC primitive
    assert pack["positions"].shape == (300, n_supercell_atoms, 3)
    assert pack["velocities"].shape == (300, n_supercell_atoms, 3)
    assert pack["time"].shape == (300,)
    assert pack["supercell"].shape == (3, 3)
    assert pack["n_md_steps"] == 300
    assert pack["time_step_fs"] == 1.0
    # ⟨T⟩ and σ_T were measured (any finite positive value)
    assert pack["md_temperature_mean"] > 0
    assert pack["md_temperature_std"] > 0


# ---------------------------------------------------------------------------
# Tier 2 — _project_with_dynaphopy synthesis smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_project_with_dynaphopy_emt_gamma_smoke(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _compute_fc2_from_scratch,
        _project_with_dynaphopy,
        _run_nvt_trajectory,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    fc2_supercell = 2 * np.eye(3, dtype=int)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    fc2_array = _compute_fc2_from_scratch.node_function(
        structure=cu, engine=engine, resolved_fc2_supercell=fc2_supercell
    )
    pack = _run_nvt_trajectory.node_function(
        structure=cu,
        engine=engine,
        resolved_fc2_supercell=fc2_supercell,
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        time_step=1.0,
        thermostat_time_constant=100.0,
        seed=42,
    )

    out = _project_with_dynaphopy.node_function(
        structure=cu,
        fc2_array=fc2_array,
        resolved_fc2_supercell=fc2_supercell,
        trajectory_pack=pack,
        resolved_q_points=np.zeros((1, 3)),  # Γ only
        temperature=300.0,
        power_spectra=False,
        keep_handles=False,
    )

    # 1 q-point × 3 bands (Cu primitive has 1 atom)
    assert out.renormalised_frequencies.shape == (1, 3)
    assert out.linewidths.shape == (1, 3)
    assert out.harmonic_frequencies.shape == (1, 3)
    assert out.power_spectra is None
    assert out.quasiparticle is None
    # All renormalised frequencies should be finite (EMT-Cu is well-behaved)
    assert np.all(np.isfinite(out.renormalised_frequencies))
    # Renormalisation should not drift wildly from the harmonic value. At
    # Gamma for a monatomic FCC primitive the three acoustic modes are
    # pinned to 0 by translation invariance; relative drift is undefined,
    # so mask those out and require absolute closeness instead.
    nonzero = np.abs(out.harmonic_frequencies) > 0.1  # THz
    if nonzero.any():
        rel_drift = np.abs(
            (out.renormalised_frequencies[nonzero]
             - out.harmonic_frequencies[nonzero])
            / out.harmonic_frequencies[nonzero]
        )
        assert (rel_drift < 0.5).all(), (
            f"Anomalous renormalisation: rel_drift = {rel_drift}"
        )
    # Acoustic modes at Gamma should be ~0 (within Lorentzian-fit noise).
    acoustic_mask = ~nonzero
    if acoustic_mask.any():
        assert np.allclose(
            out.renormalised_frequencies[acoustic_mask],
            out.harmonic_frequencies[acoustic_mask],
            atol=0.1,
        ), (
            "Acoustic-mode renormalisation deviates from harmonic ~0: "
            f"{out.renormalised_frequencies[acoustic_mask]}"
        )


# ---------------------------------------------------------------------------
# Tier 2 — full-macro smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calculate_phonon_md_renormalisation_macro_emt(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        time_step=1.0,
        q_points=[[0.0, 0.0, 0.0]],  # Gamma-only for runtime
        seed=42,
    )
    wf.run()
    out = wf.outputs.md_phonon_output.value

    assert out.converged is True
    assert out.renormalised_frequencies.shape == (1, 3)
    assert out.q_points.shape == (1, 3)
    # FC2 was recomputed -> fc2_disp_NNNN dirs on disk
    assert (tmp_path / "fc2_disp_0000").exists()


@pytest.mark.slow
def test_md_macro_reuses_fc2_from_phono3py_output(tmp_path):
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        calculate_phonon_thermal_conductivity,
    )
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc = 2 * np.eye(3, dtype=int)

    # Step 1: run the phono3py macro with keep_handles=True
    engine_phono3py = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "phono3py_run"),
    )
    wf_phono3py = calculate_phonon_thermal_conductivity(
        structure=cu,
        engine=engine_phono3py,
        fc2_supercell_matrix=sc,
        temperatures=[300.0],
        q_mesh=(3, 3, 3),
        keep_handles=True,
    )
    wf_phono3py.run()
    phono3py_out = wf_phono3py.outputs.phonon_output.value

    # Step 2: run dynaphopy macro reusing the FC2.
    engine_md = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "md_run"),
    )
    wf_md = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine_md,
        # fc2_supercell_matrix deliberately NOT passed → must derive from
        # phono3py_output
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        time_step=1.0,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
        phono3py_output=phono3py_out,
    )
    wf_md.run()
    out = wf_md.outputs.md_phonon_output.value

    # Reuse path → no fc2_disp_NNNN directories in the dynaphopy run's workdir.
    assert not (tmp_path / "md_run" / "fc2_disp_0000").exists()
    # FC2 supercell propagates from phono3py output.
    np.testing.assert_array_equal(
        out.fc2_supercell_matrix, phono3py_out.fc2_supercell_matrix
    )
    assert out.renormalised_frequencies.shape == (1, 3)


@pytest.mark.slow
def test_md_macro_warns_when_temperature_drifts(monkeypatch, tmp_path):
    """Monkey-patch the trajectory pack to fake a wildly drifted ⟨T⟩."""
    pytest.importorskip("dynaphopy")
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons import md_renormalised
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        calculate_phonon_md_renormalisation,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    # Wrap _run_nvt_trajectory.node_function so the returned pack reports a
    # bogus ⟨T⟩ that triggers the drift check. functools.wraps preserves the
    # original signature so pyiron_workflow's preview-build introspection
    # doesn't see a generic (*args, **kwargs) wrapper, and the staticmethod()
    # wrapper preserves the original (un-bound) descriptor behaviour so the
    # node doesn't pass itself as the first positional arg.
    import functools

    original_node_function = md_renormalised._run_nvt_trajectory.node_function

    @functools.wraps(original_node_function)
    def drifted_node_function(*args, **kwargs):
        pack = original_node_function(*args, **kwargs)
        pack["md_temperature_mean"] = 200.0  # >>3% drift from requested 300
        return pack

    monkeypatch.setattr(
        md_renormalised._run_nvt_trajectory,
        "node_function",
        staticmethod(drifted_node_function),
    )

    wf = calculate_phonon_md_renormalisation(
        structure=cu,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperature=300.0,
        equilibration_steps=200,
        production_steps=2000,
        q_points=[[0.0, 0.0, 0.0]],
        seed=42,
    )
    with pytest.warns(UserWarning, match=r"⟨T⟩ drift.*exceeds tolerance"):
        wf.run()
