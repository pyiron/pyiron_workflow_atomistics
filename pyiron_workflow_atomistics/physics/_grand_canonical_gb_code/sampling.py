"""RNG-driven parameter sampling for grand-canonical GB optimization.

Ports of:
    - GRIP utils/utils.py:compute_weights        → compute_weights
    - GRIP utils/utils.py:get_xy_translation     → sample_xy_translation
    - GRIP utils/utils.py:get_xy_replications    → sample_xy_replications
    - GRIP core/simulation.py:Simulation.sample_params
        → sample_md_temperature, sample_md_steps

Differences from upstream:
    - sample_xy_translation drops SLURM/PBS env-var branches and the
      ``pid``-based y-binning. Upstream split the y axis across MPI
      ranks so each rank explored its own band; here every search
      samples the full y range and different seeds give different
      draws. ``ngrid`` is retained for forward compatibility (future
      work may reintroduce stratified y-sampling at the workflow level).
    - sample_md_steps/sample_md_temperature take a GCOConfig instead of
      reading from a dict-based "algo" param block.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .config import GCOConfig


def compute_weights(cfg: GCOConfig) -> dict[str, np.ndarray]:
    """Replication weight vectors for the x and y axes.

    Returns a dict with keys ``nx``, ``ny`` (allowed replication counts) and
    ``wx``, ``wy`` (probabilities), suitable for ``np.random.Generator.choice``.

    ``reps_mode``:
        1 → exact (probability 1 at the maximum)
        2 → uniform
        3 → exponential negative (favours smaller cells)
        4 → exponential positive (favours larger cells)
    """
    nx = np.arange(cfg.size0[0], cfg.size[0] + 1)
    ny = np.arange(cfg.size0[1], cfg.size[1] + 1)
    nnx = cfg.size[0] - cfg.size0[0] + 1
    nny = cfg.size[1] - cfg.size0[1] + 1

    if cfg.reps_mode == 1:  # exact (max)
        wx = np.zeros(nnx)
        wx[-1] = 1.0
        wy = np.zeros(nny)
        wy[-1] = 1.0
    elif cfg.reps_mode == 2:  # uniform
        wx = np.ones(nnx) / nnx
        wy = np.ones(nny) / nny
    elif cfg.reps_mode in {3, 4}:  # exp-small or exp-large
        wx = np.exp(-nx)
        wx = wx / wx.sum()
        wy = np.exp(-ny)
        wy = wy / wy.sum()
        if cfg.reps_mode == 4:
            wx = wx[::-1]
            wy = wy[::-1]
    else:  # validate_gco_config should have prevented this
        raise ValueError(f"Invalid reps_mode: {cfg.reps_mode}")

    return {"nx": nx, "wx": wx, "ny": ny, "wy": wy}


def sample_xy_replications(
    rng: np.random.Generator,
    weights: dict[str, np.ndarray],
) -> tuple[int, int]:
    """Draw ``(rx, ry)`` from the replication-weight distributions."""
    rx = int(rng.choice(weights["nx"], p=weights["wx"]))
    ry = int(rng.choice(weights["ny"], p=weights["wy"]))
    return rx, ry


def sample_xy_translation(
    slab: Atoms,
    rng: np.random.Generator,
    ngrid: int,  # noqa: ARG001 — retained for future stratified sampling
) -> tuple[float, float]:
    """Uniform random translation in the (x, y) plane of ``slab``.

    Returns ``(dx, dy)`` with ``dx`` in ``[0, slab.cell[0,0]]`` and
    ``dy`` in ``[0, slab.cell[1,1]]``.
    """
    dx = float(rng.uniform(0.0, slab.cell[0, 0]))
    dy = float(rng.uniform(0.0, slab.cell[1, 1]))
    return dx, dy


def sample_md_temperature(cfg: GCOConfig, rng: np.random.Generator) -> int:
    """Random temperature in multiples of 100 K within ``[t_min, t_max]``."""
    return int(rng.choice(np.arange(cfg.t_min, cfg.t_max + 1, 100)))


def sample_md_steps(cfg: GCOConfig, rng: np.random.Generator) -> int:
    """Random MD step count within ``[md_min_steps, md_max_steps]``.

    ``md_step_sampling``:
        "exact"       → always ``md_min_steps``
        "linear"      → uniform in [md_min_steps, md_max_steps], rounded to 1000
        "exponential" → exponentially distributed, rounded to 1000
    """
    if cfg.md_step_sampling == "exact":
        return cfg.md_min_steps
    if cfg.md_step_sampling == "linear":
        n = int(rng.integers(cfg.md_min_steps, cfg.md_max_steps, endpoint=True))
        rounded = int(np.round(n, -3))
        return max(cfg.md_min_steps, min(cfg.md_max_steps, rounded))
    if cfg.md_step_sampling == "exponential":
        if cfg.md_min_steps <= 0:
            return 0
        c = np.log(cfg.md_max_steps / cfg.md_min_steps)
        n = cfg.md_min_steps * np.exp(c * rng.random())
        rounded = int(np.round(n, -3))
        return max(cfg.md_min_steps, min(cfg.md_max_steps, rounded))
    raise ValueError(f"Invalid md_step_sampling: {cfg.md_step_sampling}")
