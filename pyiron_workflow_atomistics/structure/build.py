"""Constructors for crystalline / surface structures."""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.build import surface as ase_surface


@pwf.as_function_node("equil_struct")
def get_bulk(
    name: str,
    crystalstructure: str | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alpha: float | None = None,
    covera: float | None = None,
    u: float | None = None,
    orthorhombic: bool = False,
    cubic: bool = False,
    basis: list | None = None,
) -> Atoms:
    """Build a bulk crystal via ``ase.build.bulk``.

    Examples
    --------
    >>> cu = get_bulk.node_function("Cu", crystalstructure="fcc", a=3.6, cubic=True)
    >>> len(cu)
    4
    """
    equil_struct = ase_bulk(
        name=name,
        crystalstructure=crystalstructure,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        covera=covera,
        u=u,
        orthorhombic=orthorhombic,
        cubic=cubic,
        basis=basis,
    )
    return equil_struct


@pwf.as_function_node("surface_slab")
def create_surface_slab(
    bulk_structure: Atoms,
    miller_indices: tuple[int, int, int] | tuple[int, int, int, int] = (1, 1, 1),
    layers: int = 3,
    vacuum: float = 10.0,
    periodic: bool = True,
) -> Atoms:
    """Cut a slab from a bulk structure via ``ase.build.surface``.

    Examples
    --------
    >>> cu = get_bulk.node_function("Cu", crystalstructure="fcc", a=3.6, cubic=True)
    >>> slab = create_surface_slab.node_function(cu, miller_indices=(1, 1, 1), layers=3)
    >>> bool(slab.pbc.all())
    True
    """
    surface_slab = ase_surface(
        bulk_structure,
        indices=miller_indices,
        layers=layers,
        vacuum=vacuum,
        periodic=periodic,
    )
    return surface_slab
