"""Grand-canonical optimization of grain-boundary phases.

Public surface:
    - gco_search:           GCO sampling loop.
    - build_bicrystal_slabs: convenience slab builder.
    - GCOConfig:            sampling configuration dataclass.

Algorithm: Chen, Heo, Wood, Asta, Frolov, *Nature Communications* **15**,
7049 (2024). DOI: 10.1038/s41467-024-51330-9.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine.inputs import CalcInputMD, CalcInputMinimize
from pyiron_workflow_atomistics.engine.protocol import Engine, calculate
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.bicrystal import (
    Bicrystal,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (
    GCOConfig,
    validate_gco_config,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.energies import (
    gb_energy,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.sampling import (
    compute_weights,
    sample_md_steps,
    sample_md_temperature,
    sample_xy_replications,
    sample_xy_translation,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.slabs import make_slabs
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.store import dedup

__all__ = ["GCOConfig", "build_bicrystal_slabs", "gco_search"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public node: build_bicrystal_slabs
# ---------------------------------------------------------------------------


@pwf.as_function_node("lower_slab", "upper_slab", "dlat")
def build_bicrystal_slabs(
    crystal: str,
    symbol: str,
    a: float,
    upper_dirs: list[list[int]],
    lower_dirs: list[list[int]],
    c: float = 0.0,
    cutoff: float = 35.0,
    size_z: int = 15,
) -> tuple[Atoms, Atoms, float]:
    """Build matched upper/lower slabs from a crystal type + tilt directions.

    Parameters
    ----------
    crystal
        One of ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"dc"``, ``"sc"``.
    symbol
        Chemical symbol (e.g. ``"Cu"``, ``"Ti"``).
    a
        ``a`` lattice constant in Å.
    upper_dirs, lower_dirs
        3×3 (or 4-index for HCP) lists of integer Miller indices defining
        each slab's orthogonal x/y/z axes.
    c
        ``c`` lattice constant in Å (HCP only).
    cutoff
        Max slab z-height in Å; ``0`` disables trimming.
    size_z
        Number of unit-cell replications along the z (GB-normal) axis
        before trimming. Increase if the trimmed slab is too thin for
        your ``gb_thick + pad`` window; default 15.

    Returns
    -------
    lower_slab, upper_slab, dlat
        Two ``ase.Atoms`` slabs plus the minimum normal-component
        lattice-vector spacing along z (Å). ``dlat`` is needed by
        ``gco_search`` to identify the GB plane.
    """
    lower_slab, upper_slab, dlat = make_slabs(
        crystal=crystal,
        symbol=symbol,
        a=a,
        c=c,
        upper_dirs=upper_dirs,
        lower_dirs=lower_dirs,
        cutoff=cutoff,
        size_z=size_z,
    )
    return lower_slab, upper_slab, dlat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_atoms_in_gb_region(structure: Atoms, bounds: np.ndarray) -> int:
    """Atom count inside the (pad-extended) GB region.

    Port of GRIP ``Simulation.get_gb_energy``'s z-mask, used to populate
    ``n_gb_atoms`` for the energy formula.
    """
    lowerb, upperb, pad = bounds
    z = structure.positions[:, 2]
    z_min, z_max = z.min(), z.max()
    lower_cutoff = z_min + lowerb - pad
    upper_cutoff = z_max - upperb + pad
    mask = (z >= lower_cutoff) & (z <= upper_cutoff)
    return int(mask.sum())


def _make_iter_md_engine(
    md_engine: Engine,
    temperature: int,
    n_ionic_steps: int,
) -> Engine:
    """Return a per-iteration MD engine with the chosen T and step count.

    Relies on the convention (already in force for ASEEngine) that the
    engine is a dataclass with a dataclass ``EngineInput``.
    """
    new_input = dataclasses.replace(
        md_engine.EngineInput,
        temperature=temperature,
        n_ionic_steps=n_ionic_steps,
    )
    return dataclasses.replace(md_engine, EngineInput=new_input)


def _validate_workflow_inputs(
    minimize_engine: Engine,
    md_engine: Engine | None,
    config: GCOConfig,
    n_iters: int,
    e_cohesive: float,
) -> None:
    if n_iters <= 0:
        raise ValueError(f"n_iters must be > 0; got {n_iters}")
    if not isinstance(minimize_engine.EngineInput, CalcInputMinimize):
        raise ValueError(
            "minimize_engine.EngineInput must be a CalcInputMinimize; "
            f"got {type(minimize_engine.EngineInput).__name__}"
        )
    if config.md_run_probability > 0.0 and md_engine is None:
        raise ValueError("config.md_run_probability > 0 requires a non-None md_engine")
    if md_engine is not None and not isinstance(md_engine.EngineInput, CalcInputMD):
        raise ValueError(
            "md_engine.EngineInput must be a CalcInputMD; "
            f"got {type(md_engine.EngineInput).__name__}"
        )
    if e_cohesive > 0:
        logger.warning(
            "e_cohesive=%.4f eV is positive; cohesive energies are conventionally "
            "negative — check sign.",
            e_cohesive,
        )
    validate_gco_config(config)


# ---------------------------------------------------------------------------
# Public node: gco_search
# ---------------------------------------------------------------------------


# GCOConfig is frozen, so a module-level singleton is safe as a default.
_DEFAULT_GCO_CONFIG = GCOConfig()


@pwf.as_function_node("results", "best_structures")
def gco_search(
    minimize_engine: Engine,
    lower_slab: Atoms,
    upper_slab: Atoms,
    e_cohesive: float,
    config: GCOConfig = _DEFAULT_GCO_CONFIG,
    n_iters: int = 100,
    md_engine: Engine | None = None,
    seed: int = 0,
    dlat: float = 0.0,
) -> tuple[pd.DataFrame, list[Atoms]]:
    """Grand-canonical sampling of GB microstructures.

    Per iteration: build bicrystal → sample translation/replication →
    vacancies on GB plane → perturb → optional Voronoi interstitial
    swap → optional MD-at-T → minimize → compute Egb → conditional
    keep (running ``E_min × e_mult`` gate) → periodic dedup.

    Parameters
    ----------
    minimize_engine
        pwa ``Engine`` carrying a ``CalcInputMinimize`` — runs at the end
        of every iteration.
    lower_slab, upper_slab
        ASE Atoms; build via ``build_bicrystal_slabs`` or supply your own.
    e_cohesive
        Bulk cohesive energy per atom in eV (negative). Used in the
        GB-energy formula.
    config
        ``GCOConfig`` with all algorithmic knobs.
    n_iters
        Number of iterations (per search).
    md_engine
        Optional pwa ``Engine`` carrying a ``CalcInputMD`` — runs before
        the minimize when ``rng.random() < config.md_run_probability``.
        ``EngineInput.temperature`` and ``n_ionic_steps`` are overridden
        per iteration via ``dataclasses.replace``.
    seed
        RNG seed for reproducibility.
    dlat
        Minimum lattice-vector z-component spacing; identifies the GB
        plane in the upper slab. ``0`` triggers GRIP's planar-slice
        fallback.

    Returns
    -------
    results : pandas.DataFrame
        Columns: ``Egb, n, dx, dy, rx, ry, T, n_md_steps, iter, converged``.
        One row per kept iteration. Empty DataFrame if no iteration
        produced a converged, gate-passing result.
    best_structures : list[ase.Atoms]
        Same order as ``results`` rows.

    Notes
    -----
    MD-mode GCO requires both engines to be dataclasses with dataclass
    ``EngineInput`` (``ASEEngine`` satisfies this). Engines that don't
    will fail at the per-iteration ``dataclasses.replace`` call.
    """
    _validate_workflow_inputs(
        minimize_engine=minimize_engine,
        md_engine=md_engine,
        config=config,
        n_iters=n_iters,
        e_cohesive=e_cohesive,
    )

    rng = np.random.default_rng(seed)
    weights = compute_weights(config)
    best_egb = float("inf")
    kept_rows: list[dict] = []
    kept_atoms: list[Atoms] = []
    voronoi_warned = False

    for i in range(n_iters):
        # ---- structure sampling ----------------------------------------
        bc = Bicrystal(
            lower=lower_slab,
            upper=upper_slab,
            config=config,
            dlat=dlat,
            make_copy=True,  # Bicrystal.copy_ul() copies lower0/upper0 into lower/upper
        )

        dx, dy = sample_xy_translation(upper_slab, rng, config.ngrid)
        bc.shift_upper(dx, dy)
        bc.get_bounds(config)
        rx, ry = sample_xy_replications(rng, weights)
        bc.replicate(rx, ry)
        bc.get_gbplane_atoms_u()
        bc.defect_upper(config, rng)
        bc.perturb_atoms(rng)
        bc.join_gb(config)

        try:
            bc.find_and_swap_inters(rng)
        except Exception as exc:
            if not voronoi_warned:
                logger.warning(
                    "Voronoi swap failed at iter %d: %s; suppressing "
                    "further warnings (each iteration is retried).",
                    i,
                    exc,
                )
                voronoi_warned = True

        atoms = bc.gb
        bounds = bc.bounds
        n_frac = bc.n if bc.n is not None else 0.0

        # ---- optional MD -----------------------------------------------
        T, n_md = 0, 0
        if md_engine is not None and rng.random() < config.md_run_probability:
            T = sample_md_temperature(config, rng)
            n_md = sample_md_steps(config, rng)
            iter_md = _make_iter_md_engine(md_engine, T, n_md).with_working_directory(
                f"iter_{i:05d}/md"
            )
            try:
                out_md = calculate.node_function(structure=atoms, engine=iter_md)
            except Exception as exc:
                logger.warning("iter %d MD failed: %s; skipping.", i, exc)
                continue
            atoms = out_md.final_structure

        # ---- minimize --------------------------------------------------
        iter_min = minimize_engine.with_working_directory(f"iter_{i:05d}/min")
        try:
            out = calculate.node_function(structure=atoms, engine=iter_min)
        except Exception as exc:
            logger.warning("iter %d minimize failed: %s; skipping.", i, exc)
            continue
        if not out.converged:
            logger.debug("iter %d did not converge; skipping store.", i)
            continue

        # ---- score + keep ----------------------------------------------
        n_gb = _count_atoms_in_gb_region(out.final_structure, bounds)
        area = float(out.final_structure.cell[0, 0]) * float(
            out.final_structure.cell[1, 1]
        )
        egb = gb_energy(
            final_energy_ev=out.final_energy,
            n_gb_atoms=n_gb,
            gb_area_a2=area,
            e_cohesive_ev=e_cohesive,
        )
        if egb < best_egb * config.e_mult:
            if egb < best_egb:
                best_egb = egb
            kept_rows.append(
                {
                    "Egb": egb,
                    "n": n_frac,
                    "dx": dx,
                    "dy": dy,
                    "rx": rx,
                    "ry": ry,
                    "T": T,
                    "n_md_steps": n_md,
                    "iter": i,
                    "converged": True,
                }
            )
            kept_atoms.append(out.final_structure)

        # ---- periodic dedup --------------------------------------------
        if config.dedup_every and (i + 1) % config.dedup_every == 0:
            kept_rows, kept_atoms = dedup(kept_rows, kept_atoms)

    results = pd.DataFrame(kept_rows)
    best_structures = kept_atoms
    return results, best_structures
