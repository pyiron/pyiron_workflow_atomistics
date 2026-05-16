"""End-to-end gco_search tests against ASEEngine + EMT.

Pulled to integration/ so they're marked slow. Each test is a tiny GCO
search (handful of iters, small cell) — should complete in < 30 s.
"""

from __future__ import annotations

import pytest

ase_emt = pytest.importorskip("ase.calculators.emt")

# These imports must follow importorskip so environments without EMT cleanly
# skip rather than fail at collection time.
from ase.calculators.emt import EMT  # noqa: E402
from ase.optimize import BFGS  # noqa: E402

from pyiron_workflow_atomistics.engine import (  # noqa: E402
    ASEEngine,
    CalcInputMD,
    CalcInputMinimize,
)
from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.config import (  # noqa: E402
    GCOConfig,
)
from pyiron_workflow_atomistics.physics.grand_canonical_gb import (  # noqa: E402
    build_bicrystal_slabs,
    gco_search,
)


@pytest.mark.slow
def test_gco_search_emt_minimize_only(tmp_path):
    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1,
            max_iterations=50,
        ),
        calculator=EMT(),
        optimizer_class=BFGS,
        working_directory=str(tmp_path / "min"),
        write_to_disk=False,
    )

    df, atoms_list = gco_search.node_function(
        minimize_engine=engine,
        lower_slab=lower,
        upper_slab=upper,
        dlat=dlat,
        e_cohesive=-3.59,
        config=GCOConfig(
            frac_min=0.7,
            frac_max=1.0,
            ngrid=10,
            size0=(1, 1, 1),
            size=(1, 2, 5),
            reps_mode=2,
            md_run_probability=0.0,
            dedup_every=0,
        ),
        n_iters=5,
        seed=0,
    )

    assert len(df) > 0, "Expected at least one iteration to converge and be kept"
    assert (df["Egb"] >= 0).all()
    assert all(len(a) > 0 for a in atoms_list)
    assert len(df) == len(atoms_list)


@pytest.mark.slow
def test_gco_search_emt_with_md(tmp_path):
    lower, upper, dlat = build_bicrystal_slabs.node_function(
        crystal="fcc",
        symbol="Cu",
        a=3.6,
        upper_dirs=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
        lower_dirs=[[1, -1, 0], [0, 0, 1], [-1, -1, 0]],
        cutoff=20.0,
    )
    minimize_engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1,
            max_iterations=20,
        ),
        calculator=EMT(),
        optimizer_class=BFGS,
        working_directory=str(tmp_path / "min"),
        write_to_disk=False,
    )
    md_engine = ASEEngine(
        EngineInput=CalcInputMD(
            mode="NVT",
            thermostat="langevin",
            temperature=400.0,
            n_ionic_steps=20,
            time_step=1.0,
            seed=0,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path / "md"),
        write_to_disk=False,
    )

    df, atoms_list = gco_search.node_function(
        minimize_engine=minimize_engine,
        md_engine=md_engine,
        lower_slab=lower,
        upper_slab=upper,
        dlat=dlat,
        e_cohesive=-3.59,
        config=GCOConfig(
            frac_min=1.0,
            frac_max=1.0,
            ngrid=10,
            size0=(1, 1, 1),
            size=(1, 1, 3),
            reps_mode=1,
            md_run_probability=1.0,
            t_min=400,
            t_max=400,
            md_min_steps=20,
            md_max_steps=20,
            md_step_sampling="exact",
            dedup_every=0,
        ),
        n_iters=2,
        seed=0,
    )

    # MD path may or may not store every iteration depending on convergence;
    # we just assert the workflow ran end-to-end without raising.
    assert len(df) == len(atoms_list)
    if not df.empty:
        # All MD-storing rows ran at 400 K with 20 steps
        assert (df["T"] == 400).all()
        assert (df["n_md_steps"] == 20).all()
