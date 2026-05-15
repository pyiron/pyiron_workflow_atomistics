"""Unit tests for slab construction utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.slabs import (
    compute_dhkl,
    make_slabs,
)


def test_compute_dhkl_fcc_111():
    d = compute_dhkl("fcc", plane=[1, 1, 1], a=3.6)
    np.testing.assert_allclose(d, 3.6 / np.sqrt(3), rtol=1e-9)


def test_compute_dhkl_bcc_110():
    d = compute_dhkl("bcc", plane=[1, 1, 0], a=2.87)
    np.testing.assert_allclose(d, 2.87 / np.sqrt(2), rtol=1e-9)


def test_compute_dhkl_hcp_basal():
    # (0001) plane spacing = c
    d = compute_dhkl("hcp", plane=[0, 0, 0, 1], a=2.95, c=4.68)
    np.testing.assert_allclose(d, 4.68, rtol=1e-9)


def test_compute_dhkl_unknown_crystal_raises():
    with pytest.raises(Exception, match="not yet supported"):
        compute_dhkl("triclinic", plane=[1, 1, 1], a=3.6)


def test_make_slabs_fcc_returns_two_slabs_and_dlat():
    lower, upper, dlat = make_slabs(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        c=0.0,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    # dlat is the interplanar spacing along z; positive
    assert dlat > 0
    # Cell z-axis is shorter than uncut after cutoff trim
    assert lower.cell[2, 2] <= 22.0
    assert upper.cell[2, 2] <= 22.0


def test_make_slabs_hcp_requires_c():
    lower, upper, dlat = make_slabs(
        crystal="hcp",
        symbol="Ti",
        a=2.95,
        c=4.68,
        upper_dirs=[[5, 2, -7, 0], [0, 0, 0, -1], [-3, 4, -1, 0]],
        lower_dirs=[[7, -2, -5, 0], [0, 0, 0, -1], [-1, 4, -3, 0]],
        cutoff=25.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    assert dlat > 0


def test_make_slabs_bcc():
    lower, upper, dlat = make_slabs(
        crystal="bcc",
        symbol="Fe",
        a=2.87,
        c=0.0,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    assert dlat > 0


def test_make_slabs_unknown_crystal_raises():
    with pytest.raises(Exception, match="not yet supported"):
        make_slabs(
            crystal="triclinic",
            symbol="X",
            a=3.0,
            c=0.0,
            upper_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            lower_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )


def test_make_slabs_cutoff_zero_disables_trim():
    # Use a small size so the uncut z dimension is small enough to verify.
    lower, upper, _ = make_slabs(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        c=0.0,
        upper_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        lower_dirs=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        cutoff=0.0,
    )
    # With FCC [100] directions, z-height = a * size_z = 3.6 * 15 = 54 Å
    np.testing.assert_allclose(lower.cell[2, 2], 3.6 * 15, rtol=1e-6)


def test_build_bicrystal_slabs_node_callable():
    from pyiron_workflow_atomistics.physics.grand_canonical_gb import (
        build_bicrystal_slabs,
    )

    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    assert len(lower) > 0
    assert len(upper) > 0
    assert dlat > 0
