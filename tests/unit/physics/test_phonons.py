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
