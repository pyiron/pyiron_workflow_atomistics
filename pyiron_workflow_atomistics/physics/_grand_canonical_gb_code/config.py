"""GCOConfig — all algorithmic knobs for one ``gco_search`` invocation.

Field names align with GRIP's ``params.yaml`` where possible; the
following are renamed for PEP 8 / clarity:

    ``MD_run``       → ``md_run_probability``
    ``Tmin``/``Tmax``→ ``t_min``/``t_max``
    ``MD_min``/``MD_max`` → ``md_min_steps``/``md_max_steps``
    ``var_steps``    → ``md_step_sampling``  ("exact"|"linear"|"exponential")
    ``Emult``        → ``e_mult``
    ``reps``         → ``reps_mode``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_VALID_MD_STEP_SAMPLING = {"exact", "linear", "exponential"}
_VALID_REPS_MODES = {1, 2, 3, 4}


@dataclass(frozen=True)
class GCOConfig:
    """All algorithmic knobs for one ``gco_search`` invocation."""

    # --- Bicrystal geometry ------------------------------------------------
    gb_thick: float = 10.0
    pad: float = 10.0
    gb_gap: float = 0.3
    vacuum: float = 1.0

    # --- In-plane translation + replication sampling -----------------------
    ngrid: int = 100
    size0: tuple[int, int, int] = (1, 1, 1)
    size: tuple[int, int, int] = (2, 4, 15)
    reps_mode: int = 2  # 1=exact, 2=uniform, 3=exp-small, 4=exp-large

    # --- Vacancy fraction on GB plane --------------------------------------
    frac_min: float = 0.0
    frac_max: float = 1.0

    # --- Atom perturbation -------------------------------------------------
    perturb_u: float = 0.0
    perturb_l: float = 0.0

    # --- Voronoi interstitial swap -----------------------------------------
    inter_p: float = 0.0
    inter_n: int = 0
    inter_t: float = 1.5
    inter_u: bool = False
    inter_r: bool = True

    # --- MD (consulted only if md_engine is supplied) ----------------------
    md_run_probability: float = 0.0
    t_min: int = 300
    t_max: int = 1200
    md_min_steps: int = 5000
    md_max_steps: int = 500_000
    md_step_sampling: str = "exponential"

    # --- Filtering / dedup -------------------------------------------------
    e_mult: float = 2.0
    dedup_every: int = 50


def validate_gco_config(cfg: GCOConfig) -> None:
    """Raise ``ValueError`` on invalid combinations; warn on sketchy ones.

    Pure side-effect function: raises or emits warnings. Caller is the
    ``gco_search`` workflow's pre-loop validation block.
    """
    # --- hard rejects ------------------------------------------------------
    if not 0.0 <= cfg.frac_min <= 1.0:
        raise ValueError(f"frac_min must be in [0, 1]; got {cfg.frac_min}")
    if not 0.0 <= cfg.frac_max <= 1.0:
        raise ValueError(f"frac_max must be in [0, 1]; got {cfg.frac_max}")
    if cfg.frac_min > cfg.frac_max:
        raise ValueError(
            f"frac_min ({cfg.frac_min}) must be <= frac_max ({cfg.frac_max})"
        )
    if cfg.e_mult < 1.0:
        raise ValueError(
            f"e_mult must be >= 1.0 (1.0 disables filter); got {cfg.e_mult}"
        )
    if cfg.reps_mode not in _VALID_REPS_MODES:
        raise ValueError(
            f"reps_mode must be one of {sorted(_VALID_REPS_MODES)}; got {cfg.reps_mode}"
        )
    if cfg.md_step_sampling not in _VALID_MD_STEP_SAMPLING:
        raise ValueError(
            f"md_step_sampling must be one of {sorted(_VALID_MD_STEP_SAMPLING)}; "
            f"got {cfg.md_step_sampling!r}"
        )
    if cfg.t_min > cfg.t_max:
        raise ValueError(f"t_min ({cfg.t_min}) must be <= t_max ({cfg.t_max})")
    if cfg.md_min_steps > cfg.md_max_steps:
        raise ValueError(
            f"md_min_steps ({cfg.md_min_steps}) must be <= md_max_steps "
            f"({cfg.md_max_steps})"
        )
    if not 0.0 <= cfg.inter_p <= 1.0:
        raise ValueError(f"inter_p must be in [0, 1]; got {cfg.inter_p}")
    if not 0.0 <= cfg.md_run_probability <= 1.0:
        raise ValueError(
            f"md_run_probability must be in [0, 1]; got {cfg.md_run_probability}"
        )

    # --- soft warnings -----------------------------------------------------
    if cfg.gb_thick < 5.0:
        logger.warning(
            "gb_thick=%.2f Å is thin; recommend >= 5.0 Å to capture GB region",
            cfg.gb_thick,
        )
    if cfg.pad < 3.0:
        logger.warning(
            "pad=%.2f Å is thin; recommend >= 3.0 Å around GB region",
            cfg.pad,
        )
