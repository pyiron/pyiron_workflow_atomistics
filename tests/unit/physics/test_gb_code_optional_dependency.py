"""`gb_code` is an optional dependency — the import error must be actionable.

`gb_code` (PyPI: ``lz-GB-code``) is not on conda-forge, so it lives in the
``[gb]`` extra rather than the mandatory requirements. That means a plain
install can reach the GB modules without it, and the resulting error is the
only thing telling the user what to do about it.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

MODULE = "pyiron_workflow_atomistics.physics._grain_boundary_code._gb_code"


def _hide_gb_code(monkeypatch):
    """Make any ``import gb_code...`` fail, as it would without the extra."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gb_code" or name.startswith("gb_code."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for name in [m for m in list(sys.modules) if m.startswith("gb_code")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_gb_code_names_the_extra(monkeypatch):
    _hide_gb_code(monkeypatch)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(MODULE)

    message = str(excinfo.value)
    # The install command is the whole point of the guard.
    assert "pyiron_workflow_atomistics[gb]" in message
    # Conda users can't get it from their channel, so say so explicitly.
    assert "conda-forge" in message
    # The underlying failure stays reachable for debugging.
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_gb_code_present_exports_the_expected_names():
    """With the extra installed, the guard is a transparent re-export."""
    module = pytest.importorskip(MODULE)

    assert module.csl_generator is not None
    assert callable(module.get_theta_m_n_list)
    assert module.GB_character is not None
