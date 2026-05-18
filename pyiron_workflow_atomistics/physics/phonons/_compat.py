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


def require_dynaphopy() -> Any:
    """Return the imported dynaphopy module or raise an actionable ImportError.

    Used by md_renormalised.py for the MD-trajectory mode-projection workflow.
    The install hint references the [phonons-md] extras group (superset of
    [phonons] adding dynaphopy on top of phonopy + phono3py + symfc).
    """
    try:
        import importlib

        return importlib.import_module("dynaphopy")
    except ImportError as e:
        raise ImportError(
            "dynaphopy is required for this workflow. "
            "Install with: pip install pyiron_workflow_atomistics[phonons-md]"
        ) from e


def ir_qpoints_and_weights(
    *,
    mesh,
    phonopy_obj,
    is_gamma_center: bool = True,
    is_mesh_symmetry: bool = True,
) -> tuple[Any, Any]:
    """Return ir-q-points and weights on a Monkhorst-Pack mesh of the primitive cell.

    Bridges the phonopy v3 → v4 API split:

    * v4 (``phonopy>=4.0``) exposes ``phonopy.phonon.grid.get_ir_qpoints_and_weights``
      which takes the primitive lattice (row vectors) plus a ``Symmetry`` object.
    * v3 used ``phonopy.structure.grid_points.GridPoints``, which took the
      reciprocal lattice plus a raw rotations array.

    ``phonopy_obj`` is an already-built ``phonopy.Phonopy`` instance whose
    ``.primitive`` and ``.primitive_symmetry`` are used to seed the grid.
    Built before force constants exist so callers cannot rely on
    ``Phonopy.run_mesh``.
    """
    require_phonopy()
    import numpy as np

    mesh_arr = np.asarray(list(mesh), dtype="int64")
    primitive_lattice = np.asarray(phonopy_obj.primitive.cell, dtype="double")

    try:
        from phonopy.phonon.grid import get_ir_qpoints_and_weights  # v4
    except ImportError:
        from phonopy.structure.grid_points import GridPoints  # v3

        reciprocal_lattice = np.linalg.inv(primitive_lattice).T
        rotations = phonopy_obj.primitive_symmetry.pointgroup_operations
        gp = GridPoints(
            mesh_numbers=mesh_arr,
            reciprocal_lattice=reciprocal_lattice,
            rotations=rotations,
            is_mesh_symmetry=is_mesh_symmetry,
            is_gamma_center=is_gamma_center,
        )
        return np.asarray(gp.qpoints, dtype=float), np.asarray(gp.weights, dtype=float)

    q_points, weights = get_ir_qpoints_and_weights(
        mesh=mesh_arr,
        lattice=primitive_lattice,
        primitive_symmetry=phonopy_obj.primitive_symmetry,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
    )
    return np.asarray(q_points, dtype=float), np.asarray(weights, dtype=float)
