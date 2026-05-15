"""Unit tests for the Interstitial site dataclass."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.interstitial import (
    Interstitial,
)


def test_basic_construction():
    site = Interstitial(p=[1.0, 2.0, 3.0], symbol="Ti", nn=6, nnd=[2.5, 2.5, 2.5],
                       label="octahedral0")
    np.testing.assert_array_equal(site.p, np.array([1.0, 2.0, 3.0]))
    assert site.symbol == "Ti"
    assert site.nn == 6
    np.testing.assert_array_equal(site.nnd, np.array([2.5, 2.5, 2.5]))
    assert site.label == "octahedral0"


def test_position_returns_numpy_array():
    site = Interstitial(p=[0.5, 1.5, 2.5])
    pos = site.position()
    assert isinstance(pos, np.ndarray)
    np.testing.assert_array_equal(pos, np.array([0.5, 1.5, 2.5]))


def test_from_df_roundtrips():
    df = pd.DataFrame({
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
        "z": [5.0, 6.0],
        "nn": [4, 6],
        "nnd": [[2.0, 2.0, 2.0, 2.0], [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]],
        "label": ["tetrahedral0", "octahedral0"],
    })
    sites = Interstitial.from_df(df)
    assert len(sites) == 2
    np.testing.assert_array_equal(sites[0].p, np.array([1.0, 3.0, 5.0]))
    assert sites[0].nn == 4
    assert sites[0].label == "tetrahedral0"
    assert sites[1].nn == 6


def test_repr_contains_class_name_and_position():
    site = Interstitial(p=[1.0, 2.0, 3.0], symbol="Ti")
    r = repr(site)
    assert "Interstitial" in r
    assert "Ti" in r
