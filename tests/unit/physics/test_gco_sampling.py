"""Unit tests for sampling utilities (translations, replications, MD params)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import GCOConfig
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.sampling import (
    compute_weights,
    sample_md_steps,
    sample_md_temperature,
    sample_xy_replications,
    sample_xy_translation,
)


@pytest.fixture
def small_slab() -> Atoms:
    return Atoms(
        "H4",
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        cell=[2.5, 3.7, 5.0],
    )


def test_compute_weights_uniform():
    cfg = GCOConfig(size0=(1, 1, 1), size=(3, 4, 1), reps_mode=2)
    w = compute_weights(cfg)
    np.testing.assert_array_equal(w["nx"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(w["ny"], np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(w["wx"].sum(), 1.0)
    np.testing.assert_allclose(w["wy"].sum(), 1.0)
    # Uniform weights
    np.testing.assert_allclose(w["wx"], np.full(3, 1 / 3))
    np.testing.assert_allclose(w["wy"], np.full(4, 1 / 4))


def test_compute_weights_exact():
    cfg = GCOConfig(size0=(1, 1, 1), size=(3, 4, 1), reps_mode=1)
    w = compute_weights(cfg)
    # Exact mode puts all weight on the maximum
    np.testing.assert_array_equal(w["wx"], np.array([0, 0, 1]))
    np.testing.assert_array_equal(w["wy"], np.array([0, 0, 0, 1]))


def test_compute_weights_exp_small_favors_small():
    cfg = GCOConfig(size0=(1, 1, 1), size=(4, 4, 1), reps_mode=3)
    w = compute_weights(cfg)
    assert w["wx"][0] > w["wx"][-1]
    assert w["wy"][0] > w["wy"][-1]
    np.testing.assert_allclose(w["wx"].sum(), 1.0)


def test_compute_weights_exp_large_favors_large():
    cfg = GCOConfig(size0=(1, 1, 1), size=(4, 4, 1), reps_mode=4)
    w = compute_weights(cfg)
    assert w["wx"][-1] > w["wx"][0]
    assert w["wy"][-1] > w["wy"][0]


def test_sample_xy_replications_within_bounds():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(size0=(1, 1, 1), size=(3, 5, 1), reps_mode=2)
    weights = compute_weights(cfg)
    for _ in range(20):
        rx, ry = sample_xy_replications(rng, weights)
        assert 1 <= rx <= 3
        assert 1 <= ry <= 5


def test_sample_xy_translation_within_cell(small_slab):
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        dx, dy = sample_xy_translation(small_slab, rng, ngrid=10)
        assert 0.0 <= dx <= small_slab.cell[0, 0]
        assert 0.0 <= dy <= small_slab.cell[1, 1]


def test_sample_xy_translation_deterministic(small_slab):
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    assert sample_xy_translation(small_slab, rng1, ngrid=10) == sample_xy_translation(
        small_slab, rng2, ngrid=10
    )


def test_sample_md_temperature_within_bounds_and_multiple_of_100():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(t_min=300, t_max=1200)
    seen = set()
    for _ in range(50):
        T = sample_md_temperature(cfg, rng)
        assert 300 <= T <= 1200
        assert T % 100 == 0
        seen.add(T)
    assert len(seen) > 1  # some variety


def test_sample_md_steps_exact():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(md_min_steps=5000, md_max_steps=500_000, md_step_sampling="exact")
    for _ in range(5):
        assert sample_md_steps(cfg, rng) == 5000


def test_sample_md_steps_linear_within_bounds():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(md_min_steps=5000, md_max_steps=500_000, md_step_sampling="linear")
    for _ in range(20):
        n = sample_md_steps(cfg, rng)
        assert 5000 <= n <= 500_000
        assert n % 1000 == 0  # rounded to nearest 1000


def test_sample_md_steps_exponential_within_bounds():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(
        md_min_steps=5000, md_max_steps=500_000, md_step_sampling="exponential"
    )
    for _ in range(20):
        n = sample_md_steps(cfg, rng)
        assert 5000 <= n <= 500_000


def test_sample_md_steps_linear_respects_min_when_not_multiple_of_1000():
    """Regression: rounding to nearest 1000 must not drop below md_min_steps."""
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(
        md_min_steps=4500,
        md_max_steps=8500,
        md_step_sampling="linear",
    )
    for _ in range(50):
        n = sample_md_steps(cfg, rng)
        assert n >= 4500
        assert n <= 8500


def test_sample_md_steps_exponential_respects_min_when_not_multiple_of_1000():
    rng = np.random.default_rng(seed=0)
    cfg = GCOConfig(
        md_min_steps=4500,
        md_max_steps=8500,
        md_step_sampling="exponential",
    )
    for _ in range(50):
        n = sample_md_steps(cfg, rng)
        assert n >= 4500
        assert n <= 8500
