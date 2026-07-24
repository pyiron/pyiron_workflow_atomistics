"""Common-neighbour-analysis and Voronoi structure descriptors (general)."""

from __future__ import annotations

import operator

import numpy as np
import pyiron_workflow as pwf
from structuretoolkit.analyse import (
    get_adaptive_cna_descriptors,
    get_voronoi_volumes,
)


@pwf.atomic("counts")
def cna_fractions(structure) -> dict:
    """Adaptive CNA counts, lowercase keys {'fcc','bcc','hcp','ico','others'}."""
    counts_obj = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=False
    )
    counts = dict(counts_obj)
    return counts


@pwf.atomic("key_max", "n_atoms", "distribution_half")
def analyse_reference_structure(structure):
    """Dominant CNA phase, atom count, and half its population fraction.

    ``distribution_half`` is the solid/liquid threshold: a structure counts as
    *solid* while the dominant-phase fraction stays above this value.
    """
    counts = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=False
    )
    key_max = max(counts.items(), key=operator.itemgetter(1))[0]
    n_atoms = len(structure)
    distribution_half = (counts[key_max] / n_atoms) / 2.0
    return key_max, n_atoms, distribution_half


@pwf.atomic("is_solid")
def classify_solid(structure, key_max: str, distribution_half: float) -> bool:
    """True if the dominant-phase fraction exceeds ``distribution_half``."""
    counts = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=False
    )
    fraction = counts.get(key_max, 0) / len(structure)
    is_solid = bool(fraction > distribution_half)
    return is_solid


@pwf.atomic("max_volume", "mean_volume")
def voronoi_max_mean(structure):
    """Max and mean per-atom Voronoi volume (A^3)."""
    volumes = get_voronoi_volumes(structure)
    max_volume = float(np.max(volumes))
    mean_volume = float(np.mean(volumes))
    return max_volume, mean_volume


@pwf.atomic("keep_mask")
def holes_mask(max_volumes, mean_volumes, factor: float = 2.0) -> list:
    """Per-entry True where no cavity: max_volume < factor * mean(mean_volumes)."""
    threshold = factor * float(np.mean(mean_volumes))
    keep_mask = [bool(m < threshold) for m in max_volumes]
    return keep_mask
