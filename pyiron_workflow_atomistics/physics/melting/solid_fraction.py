from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from sklearn.neighbors import KernelDensity
from structuretoolkit.analyse import get_adaptive_cna_descriptors


@pwf.as_function_node("solid_fraction")
def solid_fraction_kde(structure, crystalstructure: str, threshold: float = 0.1) -> float:
    """Fraction of the z-extent occupied by the crystalline phase.

    Per-atom CNA labels the target phase; a 1-D KDE of those atoms' z-positions
    gives the solid slab width relative to the cell. Mirrors the interface
    method's ``plot_solid_liquid_ratio`` (minus plotting).
    """
    target = crystalstructure.lower()
    labels = np.array(
        get_adaptive_cna_descriptors(
            structure=structure, mode="str", ovito_compatibility=False
        )
    )
    z = structure.get_positions()[:, 2]
    mask = labels == target
    if mask.sum() <= 0.05 * len(structure):
        ratio = 0.0
    else:
        bandwidth = (structure.get_volume() / len(structure)) ** (1.0 / 3.0)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            z[mask].reshape(-1, 1)
        )
        grid = np.linspace(z.min(), z.max(), 1000)
        g = np.exp(kde.score_samples(grid.reshape(-1, 1)))
        g = g / g.max()
        above = grid[g > threshold]
        below = grid[g < threshold]
        span = grid.max() - grid.min()
        ratio_above = (above.max() - above.min()) / span if len(above) else 1.0
        ratio_below = 1.0 - (below.max() - below.min()) / span if len(below) else 0.0
        if ratio_below == 0.0:
            ratio = ratio_above
        elif ratio_above == 1.0:
            ratio = ratio_below
        else:
            ratio = min(ratio_below, ratio_above)
    return float(ratio)
