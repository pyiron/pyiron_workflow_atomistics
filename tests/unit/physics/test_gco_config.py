"""Unit tests for GCOConfig dataclass and its validation."""

from __future__ import annotations

import dataclasses

import pytest

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (
    GCOConfig,
    validate_gco_config,
)


def test_gco_config_defaults_match_grip_baseline():
    cfg = GCOConfig()

    # Geometry
    assert cfg.gb_thick == 10.0
    assert cfg.pad == 10.0
    assert cfg.gb_gap == 0.3
    assert cfg.vacuum == 1.0

    # Sampling
    assert cfg.ngrid == 100
    assert cfg.size0 == (1, 1, 1)
    assert cfg.size == (2, 4, 15)
    assert cfg.reps_mode == 2

    # Vacancy fraction
    assert cfg.frac_min == 0.0
    assert cfg.frac_max == 1.0

    # Perturbation
    assert cfg.perturb_u == 0.0
    assert cfg.perturb_l == 0.0

    # Interstitials
    assert cfg.inter_p == 0.0
    assert cfg.inter_n == 0
    assert cfg.inter_t == 1.5
    assert cfg.inter_u is False
    assert cfg.inter_r is True

    # MD
    assert cfg.md_run_probability == 0.0
    assert cfg.t_min == 300
    assert cfg.t_max == 1200
    assert cfg.md_min_steps == 5000
    assert cfg.md_max_steps == 500_000
    assert cfg.md_step_sampling == "exponential"

    # Filtering
    assert cfg.e_mult == 2.0
    assert cfg.dedup_every == 50


def test_gco_config_is_frozen():
    cfg = GCOConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.gb_thick = 5.0  # type: ignore[misc]


def test_validate_accepts_defaults():
    # Should not raise.
    validate_gco_config(GCOConfig())


def test_validate_rejects_inverted_frac_bounds():
    cfg = GCOConfig(frac_min=0.8, frac_max=0.2)
    with pytest.raises(ValueError, match="frac_min"):
        validate_gco_config(cfg)


def test_validate_rejects_frac_outside_unit_interval():
    with pytest.raises(ValueError, match="frac_min"):
        validate_gco_config(GCOConfig(frac_min=-0.1))
    with pytest.raises(ValueError, match="frac_max"):
        validate_gco_config(GCOConfig(frac_max=1.5))


def test_validate_rejects_e_mult_below_one():
    with pytest.raises(ValueError, match="e_mult"):
        validate_gco_config(GCOConfig(e_mult=0.5))


def test_validate_rejects_invalid_md_step_sampling():
    with pytest.raises(ValueError, match="md_step_sampling"):
        validate_gco_config(GCOConfig(md_step_sampling="invalid"))


def test_validate_rejects_t_min_above_t_max():
    with pytest.raises(ValueError, match="t_min"):
        validate_gco_config(GCOConfig(t_min=1500, t_max=1200))


def test_validate_rejects_md_min_above_md_max():
    with pytest.raises(ValueError, match="md_min_steps"):
        validate_gco_config(GCOConfig(md_min_steps=1000, md_max_steps=500))


def test_validate_rejects_inter_p_out_of_range():
    with pytest.raises(ValueError, match="inter_p"):
        validate_gco_config(GCOConfig(inter_p=-0.1))
    with pytest.raises(ValueError, match="inter_p"):
        validate_gco_config(GCOConfig(inter_p=1.1))


def test_validate_rejects_md_run_probability_out_of_range():
    with pytest.raises(ValueError, match="md_run_probability"):
        validate_gco_config(GCOConfig(md_run_probability=-0.1))
    with pytest.raises(ValueError, match="md_run_probability"):
        validate_gco_config(GCOConfig(md_run_probability=1.1))


def test_validate_rejects_invalid_reps_mode():
    with pytest.raises(ValueError, match="reps_mode"):
        validate_gco_config(GCOConfig(reps_mode=5))


def test_validate_warns_on_thin_gb_thick(caplog):
    with caplog.at_level("WARNING"):
        validate_gco_config(GCOConfig(gb_thick=3.0))
    assert "gb_thick" in caplog.text


def test_validate_warns_on_thin_pad(caplog):
    with caplog.at_level("WARNING"):
        validate_gco_config(GCOConfig(pad=2.0))
    assert "pad" in caplog.text
