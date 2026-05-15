"""Tests for pyiron_workflow_atomistics.physics.free_energy."""

from __future__ import annotations

import sys

import pytest


def test_require_calphy_raises_actionable_when_missing(monkeypatch):
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    monkeypatch.setitem(sys.modules, "calphy", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        _compat._require_calphy()
    assert "pip install 'pyiron_workflow_atomistics[free-energy]'" in str(exc.value)


def test_require_lammps_engine_raises_actionable_when_missing(monkeypatch):
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    monkeypatch.setitem(sys.modules, "pyiron_workflow_lammps", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        _compat._require_lammps_engine()
    assert "pip install 'pyiron_workflow_atomistics[free-energy]'" in str(exc.value)


def test_require_calphy_returns_module_when_present():
    from pyiron_workflow_atomistics.physics.free_energy import _compat

    calphy = pytest.importorskip("calphy")
    assert _compat._require_calphy() is calphy


# ---------------------------------------------------------------------------
# LammpsPotential
# ---------------------------------------------------------------------------


def test_lammps_potential_required_fields():
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /path/to/Cu.eam.alloy Cu")
    assert pot.pair_style == "eam/alloy"
    assert pot.pair_coeff == "* * /path/to/Cu.eam.alloy Cu"
    assert pot.potential_file is None


def test_lammps_potential_optional_file():
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(
        pair_style="pace",
        pair_coeff="* * /path/to/pot.yace Cu",
        potential_file="/path/to/extra.txt",
    )
    assert pot.potential_file == "/path/to/extra.txt"


def test_lammps_potential_picklable():
    import pickle
    from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential

    pot = LammpsPotential(pair_style="eam/alloy",
                          pair_coeff="* * /path/to/Cu.eam.alloy Cu")
    restored = pickle.loads(pickle.dumps(pot))
    assert restored == pot
