"""Tests for `physics._grain_boundary_code.searcher`.

Tier 1 — synthetic DataFrame tests for the pure dedup / negation / structure
duplicate-check helpers. These run fast and need only pandas + numpy + pymatgen.
Tier 2 — gated on ``gb_code`` for the actual GB DataFrame generator and
the parallel multi-axis driver. Marked slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Tier 1 — pure helpers
# ---------------------------------------------------------------------------


def test_deduplicate_miller_equivalent_collapses_orientation_variants():
    """``canonical_gb`` should treat (1,1,1)/(1,-1,-1) and its permutations as
    the same orbit under the cubic point group + plane-swap symmetry."""
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _deduplicate_gbcode_df_miller_indices_equivalent,
    )

    df = pd.DataFrame(
        {
            "Sigma": [3, 3, 3],
            # The second row is the first row with both planes negated (axis flip)
            # — should be canonicalised to the same fingerprint.
            "GB1": [(1, 1, 1), (-1, -1, -1), (2, 0, 0)],
            "GB2": [(1, -1, -1), (-1, 1, 1), (0, 2, 0)],
            "n_atoms": [4, 4, 8],
        }
    )

    out = _deduplicate_gbcode_df_miller_indices_equivalent(df)
    # The (2,0,0)/(0,2,0) row is a distinct orbit -> still present.
    # One of the two (1,1,1)/(1,-1,-1) rows is dropped.
    assert len(out) == 2
    assert "canon" in out.columns
    assert "dupe" in out.columns


def test_deduplicate_keeps_all_rows_when_orbits_are_distinct():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _deduplicate_gbcode_df_miller_indices_equivalent,
    )

    df = pd.DataFrame(
        {
            "Sigma": [3, 5],
            "GB1": [(1, 1, 1), (3, 1, 0)],
            "GB2": [(1, -1, 0), (1, -3, 0)],
            "n_atoms": [4, 12],
        }
    )

    out = _deduplicate_gbcode_df_miller_indices_equivalent(df)
    assert len(out) == 2


def test_rid_negative_duplicates_drops_one_per_negation_pair():
    """Rows whose GB1+GB2 are component-wise negations and have identical
    ``n_atoms`` should be collapsed within each Sigma group."""
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _rid_negative_duplicates,
    )

    df = pd.DataFrame(
        {
            "Sigma": [5, 5, 5],
            "GB1": [(1, 0, 0), (-1, 0, 0), (3, 1, 0)],
            "GB2": [(0, 1, 0), (0, -1, 0), (1, -3, 0)],
            "n_atoms": [10, 10, 20],
        }
    )

    out = _rid_negative_duplicates(df)
    assert len(out) == 2
    # The two negation-pair rows collapse to one; the third (different) row stays.
    assert (out["n_atoms"] == 20).sum() == 1


def test_rid_negative_duplicates_does_not_cross_sigma_groups():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _rid_negative_duplicates,
    )

    df = pd.DataFrame(
        {
            "Sigma": [3, 5],
            "GB1": [(1, 0, 0), (-1, 0, 0)],
            "GB2": [(0, 1, 0), (0, -1, 0)],
            "n_atoms": [10, 10],
        }
    )

    out = _rid_negative_duplicates(df)
    assert len(out) == 2  # different Sigma → never collapsed


def test_rid_negative_duplicates_requires_same_n_atoms():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _rid_negative_duplicates,
    )

    df = pd.DataFrame(
        {
            "Sigma": [5, 5],
            "GB1": [(1, 0, 0), (-1, 0, 0)],
            "GB2": [(0, 1, 0), (0, -1, 0)],
            "n_atoms": [10, 12],  # mismatch breaks the negation rule
        }
    )

    out = _rid_negative_duplicates(df)
    assert len(out) == 2


def test_check_duplicates_in_group_drops_identical_structures():
    """``_check_duplicates_in_group`` uses pymatgen's StructureMatcher; two
    identical FCC Cu cells should be reported as duplicates."""
    from ase.build import bulk

    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _check_duplicates_in_group,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    group = pd.DataFrame({"structure": [cu, cu.copy(), cu.copy()]})
    # Pandas index after construction is 0, 1, 2 — the helper drops by index.
    drops = _check_duplicates_in_group(group)
    assert sorted(drops) == [1, 2]


def test_check_duplicates_in_group_keeps_distinct_structures():
    from ase.build import bulk

    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _check_duplicates_in_group,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    ni = bulk("Ni", "fcc", a=3.6, cubic=True)
    group = pd.DataFrame({"structure": [cu, ni]})
    assert _check_duplicates_in_group(group) == []


def test_remove_duplicate_structures_groups_by_n_atoms():
    """The dedup step only compares structures with matching atom counts;
    a 4-atom and an 8-atom cell are never considered duplicates."""
    from ase.build import bulk

    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _remove_duplicate_structures,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    cu2 = cu * (2, 1, 1)
    df = pd.DataFrame({"structure": [cu, cu.copy(), cu2]})
    out = _remove_duplicate_structures(df, max_atoms=100, max_workers=1)
    # 4-atom group had 2 identical Cu cells → one drop. 8-atom group → keep.
    assert len(out) == 2


def test_remove_duplicate_structures_skips_groups_above_max_atoms():
    from ase.build import bulk

    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _remove_duplicate_structures,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    big = cu * (2, 2, 2)  # 32 atoms; above max_atoms
    df = pd.DataFrame({"structure": [big, big.copy()]})
    out = _remove_duplicate_structures(df, max_atoms=10, max_workers=1)
    assert len(out) == 2  # neither group was actually compared


# ---------------------------------------------------------------------------
# Tier 2 — gated on gb_code (always present in the test env, but importorskip
# makes this resilient to lean CI). Marked slow because each call spins up a
# multiprocessing pool.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_get_gbcode_df_for_axis_returns_expected_schema():
    pytest.importorskip("gb_code")
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        _get_gbcode_df,
    )

    df = _get_gbcode_df(
        axis=np.array([1, 0, 0]),
        basis="fcc",
        sigma_limit=10,
        lim_plane_index=1,
        max_atoms=200,
        max_workers=1,
    )
    assert {
        "Axis",
        "Sigma",
        "m",
        "n",
        "GB1",
        "GB2",
        "Theta (deg)",
        "Type",
        "n_atoms",
    } <= set(df.columns)
    assert (df["n_atoms"] <= 200).all()
    assert len(df) > 0


@pytest.mark.slow
def test_get_gb_code_df_macro_runs_and_deduplicates():
    pytest.importorskip("gb_code")
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        get_gb_code_df,
    )

    out = get_gb_code_df.node_function(
        axes_list=[np.array([1, 0, 0])],
        basis="fcc",
        sigma_limit=6,
        lim_plane_index=1,
        max_atoms=200,
        max_workers=1,
        deduplicate=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert "canon" in out.columns
    assert len(out) >= 1


@pytest.mark.slow
def test_get_gb_code_df_macro_no_dedup_returns_more_rows():
    """``deduplicate=False`` short-circuits the orientation collapsing step,
    so the no-dedup output has ≥ the dedup output rows."""
    pytest.importorskip("gb_code")
    from pyiron_workflow_atomistics.physics._grain_boundary_code.searcher import (
        get_gb_code_df,
    )

    raw = get_gb_code_df.node_function(
        axes_list=[np.array([1, 0, 0])],
        basis="fcc",
        sigma_limit=6,
        lim_plane_index=1,
        max_atoms=200,
        max_workers=1,
        deduplicate=False,
    )
    dedup = get_gb_code_df.node_function(
        axes_list=[np.array([1, 0, 0])],
        basis="fcc",
        sigma_limit=6,
        lim_plane_index=1,
        max_atoms=200,
        max_workers=1,
        deduplicate=True,
    )
    assert len(raw) >= len(dedup)
