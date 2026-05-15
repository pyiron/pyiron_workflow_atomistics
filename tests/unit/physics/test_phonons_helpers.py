"""Tier 1 helpers for the phonons subpackage.

Covers the cheap pure-Python helpers in
``physics.phonons.harmonic`` / ``anharmonic`` / ``md_renormalised`` that
don't need phonopy or phono3py to be importable. Heavy paths live in the
sibling ``test_phonons.py`` module behind ``pytest.importorskip``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk

# ---------------------------------------------------------------------------
# harmonic._normalise_supercell_matrix
# ---------------------------------------------------------------------------


def test_normalise_supercell_matrix_scalar_returns_diag_identity():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _normalise_supercell_matrix,
    )

    out = _normalise_supercell_matrix(3)
    assert out.shape == (3, 3)
    assert out.dtype.kind in "iu"
    assert np.array_equal(out, 3 * np.eye(3, dtype=int))


def test_normalise_supercell_matrix_1d_list_to_diag():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _normalise_supercell_matrix,
    )

    out = _normalise_supercell_matrix([2, 3, 4])
    assert np.array_equal(out, np.diag([2, 3, 4]))


def test_normalise_supercell_matrix_2d_preserved():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _normalise_supercell_matrix,
    )

    src = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    out = _normalise_supercell_matrix(src)
    assert out.shape == (3, 3)
    assert np.array_equal(out, src)


def test_normalise_supercell_matrix_1d_wrong_shape_raises():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _normalise_supercell_matrix,
    )

    with pytest.raises(ValueError, match=r"1d shape must be \(3,\)"):
        _normalise_supercell_matrix([1, 2])


def test_normalise_supercell_matrix_2d_wrong_shape_raises():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _normalise_supercell_matrix,
    )

    with pytest.raises(ValueError, match=r"2d shape must be \(3,3\)"):
        _normalise_supercell_matrix(np.zeros((2, 3)))


def test_normalise_supercell_matrix_high_dim_raises():
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _normalise_supercell_matrix,
    )

    with pytest.raises(ValueError, match=r"must be int / \(3,\) / \(3,3\)"):
        _normalise_supercell_matrix(np.zeros((3, 3, 3)))


# ---------------------------------------------------------------------------
# md_renormalised._normalise_supercell_matrix (local copy, must match)
# ---------------------------------------------------------------------------


def test_md_normalise_supercell_matrix_matches_harmonic_for_all_forms():
    """The md_renormalised local copy must produce byte-identical results."""
    from pyiron_workflow_atomistics.physics.phonons import harmonic, md_renormalised

    for arg in [2, [2, 2, 2], np.diag([1, 2, 3])]:
        a = harmonic._normalise_supercell_matrix(arg)
        b = md_renormalised._normalise_supercell_matrix(arg)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# md_renormalised._auto_band_path
# ---------------------------------------------------------------------------


def test_auto_band_path_returns_npoints_by_3():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _auto_band_path,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    pts = _auto_band_path(cell=np.asarray(cu.cell), npoints=15)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.shape[0] >= 5  # ASE returns at least a few high-symmetry points


def test_auto_band_path_distinct_cells_give_different_paths():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _auto_band_path,
    )

    fcc = bulk("Cu", "fcc", a=3.6)
    bcc = bulk("Fe", "bcc", a=2.86)
    a = _auto_band_path(cell=np.asarray(fcc.cell), npoints=20)
    b = _auto_band_path(cell=np.asarray(bcc.cell), npoints=20)
    # The two paths cover different reciprocal regions.
    assert not np.allclose(a[: min(len(a), len(b))], b[: min(len(a), len(b))])


# ---------------------------------------------------------------------------
# md_renormalised._multiplier_to_cell_vectors
# ---------------------------------------------------------------------------


def test_multiplier_to_cell_vectors_scales_each_axis():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _multiplier_to_cell_vectors,
    )

    cell = np.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]])
    out = _multiplier_to_cell_vectors(cell, multiplier=[2, 3, 4])
    assert out.shape == (3, 3)
    np.testing.assert_allclose(np.diag(out), [6.0, 12.0, 20.0])


def test_multiplier_to_cell_vectors_with_3x3():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _multiplier_to_cell_vectors,
    )

    cell = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    P = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    out = _multiplier_to_cell_vectors(cell, multiplier=P)
    np.testing.assert_allclose(out, P)


# ---------------------------------------------------------------------------
# anharmonic._check_polar_unsupported (extra coverage beyond test_phonons.py)
# ---------------------------------------------------------------------------


def test_check_polar_unsupported_both_kwargs_raises():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError, match="NAC"):
        _check_polar_unsupported(
            born_charges=np.zeros((4, 3, 3)),
            epsilon_inf=np.eye(3),
        )


# ---------------------------------------------------------------------------
# anharmonic._resolve_random_seed
# ---------------------------------------------------------------------------


def test_resolve_random_seed_returns_user_value_when_set():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    assert _resolve_random_seed(number_of_snapshots=10, random_seed=42) == 42


def test_resolve_random_seed_returns_none_when_no_snapshots():
    """Without random mode, the seed isn't auto-generated — pass-through."""
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    assert _resolve_random_seed(number_of_snapshots=None, random_seed=None) is None


def test_resolve_random_seed_autofills_when_snapshots_set_and_seed_missing():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _resolve_random_seed,
    )

    seed = _resolve_random_seed(number_of_snapshots=10, random_seed=None)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


# ---------------------------------------------------------------------------
# anharmonic._is_kappa_not_converged
# ---------------------------------------------------------------------------


def test_is_kappa_not_converged_detects_phrase():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _is_kappa_not_converged,
    )

    assert _is_kappa_not_converged(["Iteration is NOT CONVERGED in 30 steps."])
    assert _is_kappa_not_converged(["foo", "NOT converged warning", "bar"])


def test_is_kappa_not_converged_returns_false_for_unrelated_messages():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _is_kappa_not_converged,
    )

    assert not _is_kappa_not_converged([])
    assert not _is_kappa_not_converged(["everything is fine", "kappa = 100"])


# ---------------------------------------------------------------------------
# anharmonic._check_all_converged
# ---------------------------------------------------------------------------


def _make_engine_output(converged: bool, wd: str = "/tmp/foo"):
    from ase.build import bulk

    cu = bulk("Cu", "fcc", a=3.6)
    cu.info["working_directory"] = wd
    return SimpleNamespace(
        converged=converged, final_structure=cu, final_forces=np.zeros((len(cu), 3))
    )


def test_check_all_converged_silent_when_all_pass():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_all_converged,
    )

    outs = [_make_engine_output(True) for _ in range(3)]
    _check_all_converged(outs, label="FC2")  # should not raise


def test_check_all_converged_raises_with_failed_indices_and_wd():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_all_converged,
    )

    outs = [
        _make_engine_output(True, "/tmp/a"),
        _make_engine_output(False, "/tmp/b"),
        _make_engine_output(False, "/tmp/c"),
    ]
    with pytest.raises(RuntimeError) as exc:
        _check_all_converged(outs, label="FC3")
    msg = str(exc.value)
    assert "FC3" in msg
    assert "1 (/tmp/b)" in msg
    assert "2 (/tmp/c)" in msg


# ---------------------------------------------------------------------------
# anharmonic._stack_forces
# ---------------------------------------------------------------------------


def test_stack_forces_returns_3d_array():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import _stack_forces

    n_atoms = 4
    outs = [SimpleNamespace(final_forces=np.ones((n_atoms, 3)) * i) for i in range(3)]
    out = _stack_forces(outs)
    assert out.shape == (3, n_atoms, 3)
    assert np.all(out[0] == 0)
    assert np.all(out[2] == 2)


# ---------------------------------------------------------------------------
# anharmonic._kappa_voigt_to_tensor
# ---------------------------------------------------------------------------


def test_kappa_voigt_to_tensor_round_trip_for_isotropic_case():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _kappa_voigt_to_tensor,
    )

    voigt = np.array([[10.0, 10.0, 10.0, 0.0, 0.0, 0.0]])
    tensor = _kappa_voigt_to_tensor(voigt)
    assert tensor.shape == (1, 3, 3)
    expected = 10.0 * np.eye(3)
    np.testing.assert_allclose(tensor[0], expected)


def test_kappa_voigt_to_tensor_general_shear_terms():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _kappa_voigt_to_tensor,
    )

    voigt = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    out = _kappa_voigt_to_tensor(voigt)[0]
    expected = np.array(
        [
            [1.0, 6.0, 5.0],
            [6.0, 2.0, 4.0],
            [5.0, 4.0, 3.0],
        ]
    )
    np.testing.assert_allclose(out, expected)
    # symmetry
    np.testing.assert_allclose(out, out.T)


def test_kappa_voigt_to_tensor_multiple_temperatures():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _kappa_voigt_to_tensor,
    )

    voigt = np.tile([10.0, 10.0, 10.0, 0.0, 0.0, 0.0], (5, 1))
    out = _kappa_voigt_to_tensor(voigt)
    assert out.shape == (5, 3, 3)
    for t in range(5):
        np.testing.assert_allclose(out[t], 10 * np.eye(3))


# ---------------------------------------------------------------------------
# md_renormalised._resolve_md_defaults — argument-coupling tests
# ---------------------------------------------------------------------------


def test_resolve_md_defaults_raises_when_both_inputs_missing():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    with pytest.raises(ValueError, match="Must supply"):
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=None,
            phono3py_output=None,
            q_points=None,
            band_npoints=10,
            seed=None,
        )


def test_resolve_md_defaults_recompute_branch_returns_expected_tags():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    sc, qpts, seed, tag, fc2 = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2,
        phono3py_output=None,
        q_points=None,
        band_npoints=10,
        seed=12345,
    )
    assert np.array_equal(sc, 2 * np.eye(3, dtype=int))
    assert qpts.shape[1] == 3
    assert seed == 12345
    assert tag == "recompute"
    assert fc2 is None


def test_resolve_md_defaults_seed_autofill_when_missing():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    *_, seed, _, _ = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2,
        phono3py_output=None,
        q_points=None,
        band_npoints=10,
        seed=None,
    )
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_resolve_md_defaults_explicit_qpoints_pass_through():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    explicit_q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    _, qpts, *_ = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=2,
        phono3py_output=None,
        q_points=explicit_q,
        band_npoints=10,
        seed=0,
    )
    np.testing.assert_allclose(qpts, explicit_q)


def test_resolve_md_defaults_qpoints_wrong_shape_raises():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6)
    with pytest.raises(ValueError, match="must be"):
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=2,
            phono3py_output=None,
            q_points=np.zeros((4, 4)),
            band_npoints=10,
            seed=0,
        )


def _fake_phonon_output(fc2_supercell, fc2=None):
    """Construct just enough of a PhononOutput for the coupling guard."""
    from pyiron_workflow_atomistics.physics.phonons.output import PhononOutput

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    return PhononOutput(
        structure=cu,
        fc2_supercell_matrix=fc2_supercell,
        fc3_supercell_matrix=fc2_supercell,
        temperatures=np.array([300.0]),
        kappa=np.zeros((1, 3, 3)),
        converged=True,
        fc2=fc2,
    )


def test_resolve_md_defaults_reuse_requires_fc2():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    upstream = _fake_phonon_output(fc2_supercell=2 * np.eye(3, dtype=int), fc2=None)
    with pytest.raises(ValueError, match="phono3py_output.fc2 is None"):
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=None,
            phono3py_output=upstream,
            q_points=None,
            band_npoints=10,
            seed=0,
        )


def test_resolve_md_defaults_reuse_succeeds_when_fc2_present():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    n = 4 * len(cu)
    fake_fc2 = np.zeros((n, n, 3, 3))
    upstream = _fake_phonon_output(fc2_supercell=2 * np.eye(3, dtype=int), fc2=fake_fc2)
    sc, _qpts, _seed, tag, fc2 = _resolve_md_defaults.node_function(
        structure=cu,
        fc2_supercell_matrix=None,
        phono3py_output=upstream,
        q_points=None,
        band_npoints=10,
        seed=0,
    )
    assert tag == "reuse"
    assert np.array_equal(sc, 2 * np.eye(3, dtype=int))
    assert fc2 is not None
    assert fc2.shape == fake_fc2.shape


def test_resolve_md_defaults_supercell_mismatch_raises():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _resolve_md_defaults,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    upstream = _fake_phonon_output(
        fc2_supercell=2 * np.eye(3, dtype=int),
        fc2=np.zeros((4, 4, 3, 3)),
    )
    with pytest.raises(ValueError, match="disagrees with"):
        _resolve_md_defaults.node_function(
            structure=cu,
            fc2_supercell_matrix=3 * np.eye(3, dtype=int),  # mismatch
            phono3py_output=upstream,
            q_points=None,
            band_npoints=10,
            seed=0,
        )


# ---------------------------------------------------------------------------
# md_renormalised._select_or_compute_fc2 — exercise the reuse and error paths
# ---------------------------------------------------------------------------


def test_select_or_compute_fc2_reuse_returns_array_as_ndarray():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _select_or_compute_fc2,
    )

    fake = [[[1.0, 2.0]]]  # nested list to make sure asarray triggers
    out = _select_or_compute_fc2.node_function(
        structure=None,
        engine=None,
        resolved_fc2_supercell=np.eye(3, dtype=int),
        fc2_source_tag="reuse",
        fc2_array_reused=fake,
    )
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.asarray(fake))


def test_select_or_compute_fc2_reuse_without_array_raises():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _select_or_compute_fc2,
    )

    with pytest.raises(RuntimeError, match="Internal error"):
        _select_or_compute_fc2.node_function(
            structure=None,
            engine=None,
            resolved_fc2_supercell=np.eye(3, dtype=int),
            fc2_source_tag="reuse",
            fc2_array_reused=None,
        )


def test_select_or_compute_fc2_unknown_tag_raises():
    from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
        _select_or_compute_fc2,
    )

    with pytest.raises(ValueError, match="Unknown fc2_source_tag"):
        _select_or_compute_fc2.node_function(
            structure=None,
            engine=None,
            resolved_fc2_supercell=np.eye(3, dtype=int),
            fc2_source_tag="bogus",
            fc2_array_reused=None,
        )


# ---------------------------------------------------------------------------
# Tier 2 — gated on phonopy/phono3py for the round-trip atoms helpers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ase_phonopy_round_trip_preserves_structure():
    pytest.importorskip("phonopy")
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _ase_to_phonopy,
        _phonopy_to_ase,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    p = _ase_to_phonopy(cu)
    back = _phonopy_to_ase(p)
    assert back.get_chemical_symbols() == cu.get_chemical_symbols()
    np.testing.assert_allclose(back.get_positions(), cu.get_positions(), atol=1e-10)
    np.testing.assert_allclose(np.asarray(back.cell), np.asarray(cu.cell), atol=1e-10)
    assert all(back.pbc)


@pytest.mark.slow
def test_build_phono3py_round_trip_carries_supercell_matrices():
    pytest.importorskip("phono3py")
    from pyiron_workflow_atomistics.physics.phonons.harmonic import _build_phono3py

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    sc = 2 * np.eye(3, dtype=int)
    ph3 = _build_phono3py(
        structure=cu,
        fc2_supercell_matrix=sc,
        fc3_supercell_matrix=sc,
    )
    np.testing.assert_array_equal(ph3.supercell_matrix, sc)
    np.testing.assert_array_equal(ph3.phonon_supercell_matrix, sc)
