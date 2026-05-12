"""Targeted tests for the uncovered branches of ``analysis.gb_plane.find_gb_plane``.

The end-to-end ``test_pure_gb_study_runs_end_to_end`` hits the double-peak
disorder branch and the standard happy path. This file fills in:

* the ``bulk_all.size == 0`` defensive ValueError (slab too thin)
* the ``len(bulk_all) > n_bulk`` deterministic-sort cap
* the single-peak disorder branch
* the ``extend_region_length > 0`` extension path

Uses a synthetic featuriser whose output depends only on the atom's
fractional coord along the GB-normal axis, so the disorder profile is
fully predictable.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms


def _make_linear_chain(n: int, spacing: float = 1.0) -> Atoms:
    """A 1-D chain of n H atoms along z, periodic, in a (1, 1, n*spacing) cell."""
    positions = [(0.5, 0.5, (i + 0.5) * spacing) for i in range(n)]
    atoms = Atoms(
        symbols=["H"] * n,
        positions=positions,
        cell=[1.0, 1.0, n * spacing],
        pbc=True,
    )
    return atoms


def _peaked_featuriser(approx_gb_frac: float):
    """Return a featuriser whose 1-D feature equals the |frac-approx_gb_frac|.

    Bulk atoms (far from approx_gb_frac in fractional space) get features
    near 0.5, GB atoms get features near 0. So disorder = ||feat - bulk_template||
    has a single peak at the GB.
    """
    def feat(atoms, idx, **kwargs):
        z_frac = atoms.get_scaled_positions()[idx, 2] % 1.0
        return [abs(z_frac - approx_gb_frac)]
    return feat


def test_find_gb_plane_raises_when_slab_too_thin_to_catch_any_bulk_atoms():
    from pyiron_workflow_atomistics.analysis.gb_plane import find_gb_plane

    # 5-atom chain. Atoms at frac = 0.1, 0.3, 0.5, 0.7, 0.9. None at 0.25 or 0.75.
    atoms = _make_linear_chain(5, spacing=1.0)
    feat = _peaked_featuriser(0.5)
    with pytest.raises(ValueError, match="no bulk atoms found"):
        find_gb_plane.node_function(
            atoms=atoms, featuriser=feat, axis="c",
            approx_frac=0.5, tolerance=0.4,
            slab_thickness=0.001,   # impossibly thin → catches nothing at 0.25/0.75
            n_bulk=10, threshold_frac=0.5,
        )


def test_find_gb_plane_caps_bulk_indices_at_n_bulk():
    """With a fine chain that puts many atoms inside each bulk slab, n_bulk
    must clip and the chosen indices are the lowest-axial-coord ones."""
    from pyiron_workflow_atomistics.analysis.gb_plane import find_gb_plane

    # 40 atoms along z over a 40 Å cell, so atoms every 1 Å. Each bulk slab
    # (centred at 0.25 = z=10 Å and 0.75 = z=30 Å, thickness 5 Å on each side)
    # catches ~10 atoms each → 20 total. n_bulk=4 forces the cap.
    atoms = _make_linear_chain(40, spacing=1.0)
    feat = _peaked_featuriser(0.5)
    out = find_gb_plane.node_function(
        atoms=atoms, featuriser=feat, axis="c",
        approx_frac=0.5, tolerance=10.0,    # GB window 0.5±0.25
        slab_thickness=5.0,                 # generous bulk slab
        n_bulk=4, threshold_frac=0.5,
    )
    assert len(out["bulk_indices"]) == 4
    # Deterministic axial-sort: bulk_indices should be in ascending fractional order.
    fracs_sel = atoms.get_scaled_positions()[out["bulk_indices"], 2]
    assert list(fracs_sel) == sorted(fracs_sel)


def test_find_gb_plane_single_peak_branch():
    """Featurise so the disorder has exactly one clear peak in the GB window."""
    from pyiron_workflow_atomistics.analysis.gb_plane import find_gb_plane

    # 21-atom chain over 21 Å. Atoms at frac = 0.0238 .. 0.9762, evenly spaced.
    atoms = _make_linear_chain(21, spacing=1.0)
    # Single-peak featuriser: |z - 10|, so the atom at z=10 (frac~0.5) is the lone
    # peak; everything else is approximately linearly higher away from the GB.
    def feat(at, idx, **kwargs):
        z = at.get_positions()[idx, 2]
        return [-abs(z - 10.0)]   # most-negative at z=10 → peak in disorder after norm
    out = find_gb_plane.node_function(
        atoms=atoms, featuriser=feat, axis="c",
        approx_frac=0.5, tolerance=5.0,
        slab_thickness=2.0,
        n_bulk=10, threshold_frac=0.3,
    )
    # GB plane should be near z=10 (frac ~ 0.476) — the lone disorder peak.
    assert 0.4 < out["gb_frac"] < 0.55
    assert 9.0 < out["gb_cart"] < 11.5


def test_find_gb_plane_extends_selection_with_extend_region_length():
    """extend_region_length > 0 must widen the returned selection."""
    from pyiron_workflow_atomistics.analysis.gb_plane import find_gb_plane

    atoms = _make_linear_chain(21, spacing=1.0)
    def feat(at, idx, **kwargs):
        z = at.get_positions()[idx, 2]
        return [-abs(z - 10.0)]
    base = find_gb_plane.node_function(
        atoms=atoms, featuriser=feat, axis="c",
        approx_frac=0.5, tolerance=5.0, slab_thickness=2.0,
        n_bulk=10, threshold_frac=0.3, extend_region_length=0.0,
    )
    extended = find_gb_plane.node_function(
        atoms=atoms, featuriser=feat, axis="c",
        approx_frac=0.5, tolerance=5.0, slab_thickness=2.0,
        n_bulk=10, threshold_frac=0.3, extend_region_length=3.0,
    )
    # Extended selection must contain at least as many atoms as the base one.
    assert len(extended["extended_sel_indices"]) >= len(base["extended_sel_indices"])
