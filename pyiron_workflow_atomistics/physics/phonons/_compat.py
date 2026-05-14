"""Lazy-import shims for the optional phonopy / phono3py / symfc stack.

These let `import pyiron_workflow_atomistics.physics.phonons` succeed in
environments where the `[phonons]` extra isn't installed. The check fires
only when a workflow that needs the optional dep is actually invoked.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = "pip install pyiron_workflow_atomistics[phonons]"


def _require(module_name: str) -> Any:
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"{module_name} is required for this workflow. Install with: {_INSTALL_HINT}"
        ) from e


def require_phonopy() -> Any:
    """Return the imported phonopy module or raise an actionable ImportError."""
    return _require("phonopy")


def require_phono3py() -> Any:
    """Return the imported phono3py module or raise an actionable ImportError."""
    return _require("phono3py")


def require_symfc() -> Any:
    """Return the imported symfc module or raise an actionable ImportError.

    Only used when fc_calculator='symfc' (random-displacement mode).
    """
    return _require("symfc")
