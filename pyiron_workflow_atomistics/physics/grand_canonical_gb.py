"""Grand-canonical optimization of grain-boundary phases.

Public surface:
    - gco_search:           GCO sampling loop (added in Task 8).
    - build_bicrystal_slabs: convenience slab builder.
    - GCOConfig:            sampling configuration dataclass.

Algorithm: Chen, Heo, Wood, Asta, Frolov, *Nature Communications* **15**,
7049 (2024). DOI: 10.1038/s41467-024-51330-9.
"""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (
    GCOConfig,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.slabs import make_slabs

__all__ = ["GCOConfig", "build_bicrystal_slabs"]


@pwf.as_function_node("lower_slab", "upper_slab", "dlat")
def build_bicrystal_slabs(
    crystal: str,
    symbol: str,
    a: float,
    upper_dirs: list[list[int]],
    lower_dirs: list[list[int]],
    c: float = 0.0,
    cutoff: float = 35.0,
    size_z: int = 15,
) -> tuple[Atoms, Atoms, float]:
    """Build matched upper/lower slabs from a crystal type + tilt directions.

    Parameters
    ----------
    crystal
        One of ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"dc"``, ``"sc"``.
    symbol
        Chemical symbol (e.g. ``"Cu"``, ``"Ti"``).
    a
        ``a`` lattice constant in Å.
    upper_dirs, lower_dirs
        3×3 (or 4-index for HCP) lists of integer Miller indices defining
        each slab's orthogonal x/y/z axes.
    c
        ``c`` lattice constant in Å (HCP only).
    cutoff
        Max slab z-height in Å; ``0`` disables trimming.
    size_z
        Number of unit-cell replications along the z (GB-normal) axis
        before trimming. Increase if the trimmed slab is too thin for
        your `gb_thick + pad` window; default 15.

    Returns
    -------
    lower_slab, upper_slab, dlat
        Two ``ase.Atoms`` slabs plus the minimum normal-component
        lattice-vector spacing along z (Å). ``dlat`` is needed by
        ``gco_search`` to identify the GB plane.
    """
    lower_slab, upper_slab, dlat = make_slabs(
        crystal=crystal,
        symbol=symbol,
        a=a,
        c=c,
        upper_dirs=upper_dirs,
        lower_dirs=lower_dirs,
        cutoff=cutoff,
        size_z=size_z,
    )
    return lower_slab, upper_slab, dlat
