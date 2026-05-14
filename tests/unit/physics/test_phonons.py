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
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{name} must be required (no default)"
        )


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
# Tier 1 — polar-material kwargs early exit
# ---------------------------------------------------------------------------


def test_born_charges_raises_not_implemented():
    """Passing born_charges raises before any phono3py import."""
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError) as exc:
        _check_polar_unsupported(
            born_charges=np.zeros((4, 3, 3)), epsilon_inf=None
        )
    msg = str(exc.value)
    assert "BORN" in msg or "Non-analytic" in msg
    assert "v1" in msg


def test_epsilon_inf_raises_not_implemented():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    with pytest.raises(NotImplementedError):
        _check_polar_unsupported(
            born_charges=None, epsilon_inf=np.eye(3)
        )


def test_no_polar_kwargs_returns_silently():
    from pyiron_workflow_atomistics.physics.phonons.anharmonic import (
        _check_polar_unsupported,
    )

    # Should return without raising
    _check_polar_unsupported(born_charges=None, epsilon_inf=None)


# ---------------------------------------------------------------------------
# Tier 3 — displacement generation determinism (gated)
# ---------------------------------------------------------------------------


phono3py = pytest.importorskip("phono3py", reason="phonons extra not installed")


def _cu_fcc_primitive():
    return bulk("Cu", "fcc", a=3.6)


def _two_by_two_by_two():
    return (2 * np.eye(3)).astype(int)


@pytest.mark.slow
def test_fd_fc2_supercells_deterministic():
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


@pytest.mark.slow
def test_fd_fc3_supercells_deterministic():
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
