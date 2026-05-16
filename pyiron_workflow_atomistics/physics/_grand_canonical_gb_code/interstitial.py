"""Interstitial site dataclass.

Direct port of GRIP's ``core/interstitial.py`` (no behavioural change).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Interstitial:
    """A candidate interstitial site identified by Voronoi analysis.

    Attributes
    ----------
    p : np.ndarray
        xyz position of the site (Å).
    symbol : str, optional
        Atomic species placed at this site, if any.
    nn : int, optional
        Number of nearest neighbours.
    nnd : np.ndarray, optional
        Distances to the ``nn`` nearest neighbours.
    label : str, optional
        Geometry label (e.g. ``"octahedral0"``, ``"tetrahedral1"``).
    """

    def __init__(
        self,
        p: Sequence[float],
        symbol: str | None = None,
        nn: int | None = None,
        nnd: Sequence[float] | None = None,
        label: str | None = None,
    ) -> None:
        self.p = np.asarray(p)
        self.symbol = symbol
        self.nn = nn
        self.nnd = np.asarray(nnd) if nnd is not None else None
        self.label = label

    @classmethod
    def from_df(cls, df) -> list[Interstitial]:
        """Construct a list of sites from a DataFrame with ``x, y, z, nn, nnd, label`` columns."""
        sites: list[Interstitial] = []
        for row in df.itertuples():
            sites.append(
                cls(p=[row.x, row.y, row.z], nn=row.nn, nnd=row.nnd, label=row.label)
            )
        return sites

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol}, p={self.p})"

    def position(self) -> np.ndarray:
        return self.p
