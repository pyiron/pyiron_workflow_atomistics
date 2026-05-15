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
