"""Slab construction utilities.

Ports of:
    - GRIP utils/utils.py:compute_dhkl   → compute_dhkl
    - GRIP utils/utils.py:make_crystals  → make_slabs

Differences from upstream:
    - GRIP wrote slabs to ``POSCAR_*`` when ``struct["write"]`` was true;
      we never write to disk from this function.
    - GRIP read existing slabs from disk when ``struct["user"]`` was true;
      callers pass in their own ``ase.Atoms`` for that path.
    - The ``struct``-dict argument is replaced by explicit kwargs.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.lattice.bravais import Lattice
from ase.lattice.cubic import (
    BodyCenteredCubic,
    Diamond,
    FaceCenteredCubic,
    SimpleCubic,
)
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked

logger = logging.getLogger(__name__)

# Small position shift applied to avoid edge-of-cell ties (matches GRIP).
_P_SHIFT = 1e-3
# z-direction tolerance used in plane masking (matches GRIP utils/constants.py).
_Z_THRESH = 1e-3


def compute_dhkl(crystal: str, plane: list[int], a: float, c: float = 0.0) -> float:
    """Interplanar spacing for (hkl) or (hkil)."""
    cs = crystal.lower()
    if cs in {"fcc", "bcc", "dc", "sc"}:
        return a / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    if cs == "hcp":
        return 1.0 / np.sqrt(
            4 / 3 * (plane[0] ** 2 + plane[0] * plane[1] + plane[1] ** 2) / a**2
            + plane[3] ** 2 / c**2
        )
    raise ValueError(f"Crystal structure '{crystal}' is not yet supported.")


_CRYSTAL_TYPES: dict[str, type] = {
    "fcc": FaceCenteredCubic,
    "bcc": BodyCenteredCubic,
    "dc": Diamond,
    "sc": SimpleCubic,
    "hcp": HexagonalClosedPacked,
}


def make_slabs(
    crystal: str,
    symbol: str,
    a: float,
    c: float,
    upper_dirs: list[list[int]],
    lower_dirs: list[list[int]],
    cutoff: float = 35.0,
    size_z: int = 15,
) -> tuple[Lattice, Lattice, float]:
    """Build upper and lower slabs + their interplanar spacing.

    Mirrors GRIP ``make_crystals`` with ``struct["user"]=False``.
    """
    cs = crystal.lower()
    if cs not in _CRYSTAL_TYPES:
        raise ValueError(f"Crystal structure '{crystal}' is not yet supported.")

    init_size = (1, 1, size_z)
    builder = _CRYSTAL_TYPES[cs]

    if cs == "hcp":
        upper = builder(
            symbol=symbol, latticeconstant=(a, c), directions=upper_dirs, size=init_size
        )
        lower = builder(
            symbol=symbol, latticeconstant=(a, c), directions=lower_dirs, size=init_size
        )
    else:
        upper = builder(
            symbol=symbol, latticeconstant=a, directions=upper_dirs, size=init_size
        )
        lower = builder(
            symbol=symbol, latticeconstant=a, directions=lower_dirs, size=init_size
        )

    # Nudge atoms slightly to avoid PBC edge ties, then wrap.
    upper.positions += [0, _P_SHIFT, _Z_THRESH]
    upper.wrap()
    lower.positions += [0, _P_SHIFT, _Z_THRESH]
    lower.wrap()

    # Trim excess z-height to ``cutoff`` (skip if cutoff=0).
    if cutoff > 0:
        for name, slab, dirs in [
            ("lower", lower, lower_dirs),
            ("upper", upper, upper_dirs),
        ]:
            if slab.cell[2, 2] > cutoff:
                nvec = dirs[2]
                dspace = compute_dhkl(cs, nvec, a, c)
                logger.debug("Interplanar spacing for %s: %.6f Å", name, dspace)
                zmax = (
                    (cutoff // dspace + 1) * dspace
                    - _Z_THRESH
                    + min(slab.positions[:, 2].round(6))
                )
                del slab[[atom.index for atom in slab if atom.position[2] > zmax]]
                slab.cell[2, 2] = zmax

    # Compute dlat (minimum normal lattice-vector component along z).
    if cs in {"fcc", "bcc", "sc"}:
        unique_z = sorted(set(lower.positions[:, 2].round(6)))
    elif cs == "dc":
        parent = FaceCenteredCubic(
            symbol=symbol, latticeconstant=a, directions=lower_dirs, size=init_size
        )
        unique_z = sorted(set(parent.positions[:, 2].round(6)))
    elif cs == "hcp":
        parent = Hexagonal(
            symbol=symbol, latticeconstant=(a, c), directions=lower_dirs, size=init_size
        )
        unique_z = sorted(set(parent.positions[:, 2].round(6)))
    else:  # unreachable
        raise ValueError(f"Unhandled crystal '{crystal}'")

    dlat = abs(unique_z[1] - unique_z[0])

    if upper.cell[2, 2] < 20:
        logger.warning(
            "Upper slab z=%.2f Å is small; results may be inaccurate.",
            upper.cell[2, 2],
        )
    if lower.cell[2, 2] < 20:
        logger.warning(
            "Lower slab z=%.2f Å is small; results may be inaccurate.",
            lower.cell[2, 2],
        )

    return lower, upper, dlat
