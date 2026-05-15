"""Lazy-import shims for the optional [free-energy] extra.

Mirrors the pattern in physics/phonons/_compat.py: every public node
calls one of these helpers at the top of its body so the error message
points the user at the install line, not a bare ``ImportError``.
"""

from __future__ import annotations

_INSTALL_HINT = (
    "pip install 'pyiron_workflow_atomistics[free-energy]'"
)


def _require_calphy():
    """Return the imported ``calphy`` module or raise an actionable error."""
    try:
        import calphy
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"calphy is required for free-energy workflows but is not "
            f"installed.\nInstall with: {_INSTALL_HINT}"
        ) from exc
    if calphy is None:  # monkeypatched in tests
        raise ModuleNotFoundError(
            f"calphy is required for free-energy workflows but is not "
            f"installed.\nInstall with: {_INSTALL_HINT}"
        )
    return calphy


def _require_lammps_engine():
    """Return ``LammpsEngine`` or raise an actionable error."""
    try:
        from pyiron_workflow_lammps.engine import LammpsEngine
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"pyiron_workflow_lammps is required for free-energy workflows "
            f"but is not installed.\nInstall with: {_INSTALL_HINT}"
        ) from exc
    import sys
    if sys.modules.get("pyiron_workflow_lammps") is None:
        raise ModuleNotFoundError(
            f"pyiron_workflow_lammps is required for free-energy workflows "
            f"but is not installed.\nInstall with: {_INSTALL_HINT}"
        )
    return LammpsEngine
