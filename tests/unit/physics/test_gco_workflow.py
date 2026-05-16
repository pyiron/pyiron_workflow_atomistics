"""Unit tests for the gco_search workflow against a stub Engine.

The real EMT integration lives in tests/integration/test_gco_emt.py.
Here we use a deterministic stub that returns canned EngineOutputs so
unit tests stay fast (<1 s) and engine-independent.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import pandas as pd
import pytest
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic

from pyiron_workflow_atomistics.engine.inputs import CalcInputMD, CalcInputMinimize
from pyiron_workflow_atomistics.engine.protocol import EngineOutput
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import GCOConfig
from pyiron_workflow_atomistics.physics.grand_canonical_gb import gco_search


@dataclass
class _FakeMinimizeEngine:
    """Stub Engine that returns the input structure with a deterministic energy."""

    EngineInput: CalcInputMinimize
    working_directory: str = "."
    base_energy: float = -3.6  # eV/atom; matches Cu EMT roughly

    def get_calculate_fn(self, structure: Atoms):
        # Compute n*E_coh plus a tiny offset that varies per structure so Egb is positive
        # but small. Deterministic w.r.t. atom count so tests are stable.
        n = len(structure)

        def _fn(structure: Atoms) -> EngineOutput:
            return EngineOutput(
                final_structure=structure.copy(),
                final_energy=n * (-3.6) + 0.005 * n,  # ~5 meV/atom above bulk
                converged=True,
            )

        return _fn, {}

    def with_working_directory(self, subdir: str) -> "_FakeMinimizeEngine":
        return dataclasses.replace(
            self, working_directory=os.path.join(self.working_directory, subdir)
        )


@dataclass
class _FakeMDEngine:
    EngineInput: CalcInputMD
    working_directory: str = "."

    def get_calculate_fn(self, structure: Atoms):
        def _fn(structure: Atoms) -> EngineOutput:
            return EngineOutput(
                final_structure=structure.copy(),
                final_energy=0.0,  # MD output is intermediate; only structure consumed
                converged=True,
            )

        return _fn, {}

    def with_working_directory(self, subdir: str) -> "_FakeMDEngine":
        return dataclasses.replace(
            self, working_directory=os.path.join(self.working_directory, subdir)
        )


@pytest.fixture
def cu_slabs():
    lower = FaceCenteredCubic(
        symbol="Cu", latticeconstant=3.6,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(1, 1, 4),
    )
    upper = FaceCenteredCubic(
        symbol="Cu", latticeconstant=3.6,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(1, 1, 4),
    )
    return lower, upper


def test_gco_search_returns_dataframe_and_atoms_list(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMinimize(),
        working_directory=str(tmp_path),
    )
    cfg = GCOConfig(
        frac_min=0.7, frac_max=1.0,
        ngrid=10, size0=(1, 1, 1), size=(1, 2, 5),
        md_run_probability=0.0, dedup_every=0,
    )
    df, atoms_list = gco_search.node_function(
        minimize_engine=engine,
        lower_slab=lower, upper_slab=upper,
        e_cohesive=-3.6,
        config=cfg, n_iters=3, seed=0, dlat=1.8,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(atoms_list, list)
    assert len(df) == len(atoms_list)
    if not df.empty:
        for col in ("Egb", "n", "dx", "dy", "rx", "ry", "T", "n_md_steps",
                    "iter", "converged"):
            assert col in df.columns


def test_gco_search_with_md_engine_invokes_both(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    min_engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMinimize(),
        working_directory=str(tmp_path / "min"),
    )
    md_engine = _FakeMDEngine(
        EngineInput=CalcInputMD(mode="NVT", temperature=500.0, n_ionic_steps=100),
        working_directory=str(tmp_path / "md"),
    )
    cfg = GCOConfig(
        frac_min=1.0, frac_max=1.0,  # no vacancies for stability
        ngrid=10, size0=(1, 1, 1), size=(1, 1, 1), reps_mode=1,
        md_run_probability=1.0,
        t_min=300, t_max=300,  # fixed T
        md_min_steps=1000, md_max_steps=1000, md_step_sampling="exact",
        dedup_every=0,
    )
    df, _ = gco_search.node_function(
        minimize_engine=min_engine, md_engine=md_engine,
        lower_slab=lower, upper_slab=upper,
        e_cohesive=-3.6, config=cfg, n_iters=2, seed=0, dlat=1.8,
    )
    # MD should have run for every kept row (md_run_probability=1.0); rows
    # should exist (frac_min=frac_max=1.0 with a converged stub engine).
    assert not df.empty, "expected at least one kept row from MD path"
    assert (df["T"] == 300).all()
    assert (df["n_md_steps"] == 1000).all()


def test_gco_search_rejects_missing_md_engine_when_probability_positive(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(EngineInput=CalcInputMinimize(),
                                 working_directory=str(tmp_path))
    cfg = GCOConfig(md_run_probability=0.5)
    with pytest.raises(ValueError, match="md_engine"):
        gco_search.node_function(
            minimize_engine=engine, md_engine=None,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=cfg, n_iters=1, seed=0, dlat=1.8,
        )


def test_gco_search_rejects_wrong_minimize_engine_input_type(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMD(mode="NVT"),  # wrong type
        working_directory=str(tmp_path),
    )
    with pytest.raises(ValueError, match="minimize_engine"):
        gco_search.node_function(
            minimize_engine=engine,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=GCOConfig(), n_iters=1, seed=0, dlat=1.8,
        )


def test_gco_search_rejects_md_engine_with_wrong_input_type(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    min_engine = _FakeMinimizeEngine(
        EngineInput=CalcInputMinimize(), working_directory=str(tmp_path),
    )
    md_engine = _FakeMDEngine(
        EngineInput=CalcInputMinimize(),  # wrong: should be CalcInputMD
        working_directory=str(tmp_path),
    )
    cfg = GCOConfig(md_run_probability=1.0)
    with pytest.raises(ValueError, match="md_engine"):
        gco_search.node_function(
            minimize_engine=min_engine, md_engine=md_engine,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=cfg, n_iters=1, seed=0, dlat=1.8,
        )


def test_gco_search_rejects_zero_iterations(cu_slabs, tmp_path):
    lower, upper = cu_slabs
    engine = _FakeMinimizeEngine(EngineInput=CalcInputMinimize(),
                                 working_directory=str(tmp_path))
    with pytest.raises(ValueError, match="n_iters"):
        gco_search.node_function(
            minimize_engine=engine,
            lower_slab=lower, upper_slab=upper,
            e_cohesive=-3.6, config=GCOConfig(), n_iters=0, seed=0, dlat=1.8,
        )


def test_gco_search_handles_failed_minimize(cu_slabs, tmp_path):
    """A non-converged or exception-raising minimize should not abort the search."""
    lower, upper = cu_slabs

    @dataclass
    class _RaisingEngine:
        EngineInput: CalcInputMinimize
        working_directory: str = "."

        def get_calculate_fn(self, structure):
            def _fn(structure):
                raise RuntimeError("simulated engine crash")
            return _fn, {}

        def with_working_directory(self, subdir):
            return dataclasses.replace(
                self, working_directory=os.path.join(self.working_directory, subdir)
            )

    engine = _RaisingEngine(EngineInput=CalcInputMinimize(),
                            working_directory=str(tmp_path))
    df, atoms_list = gco_search.node_function(
        minimize_engine=engine,
        lower_slab=lower, upper_slab=upper,
        e_cohesive=-3.6, config=GCOConfig(frac_min=1.0, frac_max=1.0, dedup_every=0),
        n_iters=3, seed=0, dlat=1.8,
    )
    # All iterations failed; df is empty but workflow did not raise
    assert df.empty
    assert atoms_list == []
