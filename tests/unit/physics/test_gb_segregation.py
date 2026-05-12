"""Smoke test for physics.grain_boundary.calculate_substitutional_segregation_GB.

Uses ``LennardJones`` because ``write_df`` pickles the result dataframe
to disk and the for_node output retains the ``engine`` column. ASE's EMT
and EAM calculators both attach local closures (``EMT.initialize.<locals>.<lambda>``
and ``EAM.deriv.<locals>.d_spline``) which are unpicklable; LennardJones
has no such closures. The workflow is element-agnostic at this layer so
the choice of calculator does not change what's exercised.

Indirectly covers the segregation branch of ``physics/grain_boundary.py``:
``calculate_substitutional_segregation_GB``,
``create_seg_structure_and_output_dir``, ``get_df_col_as_list``,
``_make_engines_from_dirs``, and ``write_df``.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones


@pytest.mark.slow
def test_calculate_substitutional_segregation_GB_runs(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.grain_boundary import (
        calculate_substitutional_segregation_GB,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    defect_sites = [0, 5, 12]

    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.5, max_iterations=5
        ),
        calculator=LennardJones(),
        working_directory=str(tmp_path),
    )

    wf = calculate_substitutional_segregation_GB(
        structure=structure,
        defect_sites=defect_sites,
        element="Ag",
        structure_basename="Cu32",
        engine=engine,
        unique_sites_df=pd.DataFrame({"site_id": defect_sites}),
        parent_dir=str(tmp_path),
        df_filename="seg_df.pkl",
    )
    out = wf.run()
    df = out["gb_seg_calcs_df"]

    assert isinstance(df, pd.DataFrame)
    # One row per defect site, columns from both the unique_sites_df concat
    # and the for_node output.
    assert len(df) == len(defect_sites)
    assert "site_id" in df.columns, "unique_sites_df was not concatenated in"

    # Each site's structure was swapped to Ag and the relaxation produced an
    # engine_output with a finite final_energy.
    assert "engine_output" in df.columns
    energies = [out.final_energy for out in df["engine_output"]]
    assert all(np.isfinite(e) for e in energies)

    # write_df pickled the result to parent_dir/df_filename.
    pickled = tmp_path / "seg_df.pkl"
    assert pickled.exists(), "write_df did not produce the expected pickle"
    df_from_disk = pickle.loads(pickled.read_bytes())
    assert len(df_from_disk) == len(defect_sites)


def test_calculate_substitutional_segregation_GB_graph_constructs(tmp_path):
    """Graph-only test: catches type-hint / channel-shape regressions in
    create_seg_structure_and_output_dir, _make_engines_from_dirs, write_df,
    and the surrounding for_node wiring, without running any EMT step.
    """
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.grain_boundary import (
        calculate_substitutional_segregation_GB,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(force_convergence_tolerance=0.5, max_iterations=2),
        calculator=LennardJones(),
        working_directory=str(tmp_path),
    )

    wf = calculate_substitutional_segregation_GB(
        structure=structure,
        defect_sites=[0, 1],
        element="Ag",
        structure_basename="Cu32",
        engine=engine,
        unique_sites_df=pd.DataFrame({"site_id": [0, 1]}),
        parent_dir=str(tmp_path),
    )
    assert wf is not None


def test_create_seg_structure_and_output_dir_swaps_one_site(tmp_path):
    """Unit test for the inner helper, no workflow indirection."""
    from pyiron_workflow_atomistics.physics.grain_boundary import (
        create_seg_structure_and_output_dir,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    seg, out_dir = create_seg_structure_and_output_dir.node_function(
        structure=structure,
        defect_site=7,
        element="Ag",
        structure_basename="Cu32",
        parent_dir=str(tmp_path),
    )
    # Exactly one Ag swap was applied; everything else is Cu.
    symbols = list(seg.get_chemical_symbols())
    assert symbols[7] == "Ag"
    assert symbols.count("Ag") == 1
    assert symbols.count("Cu") == len(structure) - 1
    # Output dir path follows the documented naming convention.
    assert out_dir == os.path.join(str(tmp_path), "Cu32_Ag_7")


def test_get_df_col_as_list_extracts_column():
    from pyiron_workflow_atomistics.physics.grain_boundary import get_df_col_as_list

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = get_df_col_as_list.node_function(df=df, col="a")
    assert out == [1, 2, 3]


def test__make_engines_from_dirs_returns_one_per_subdir(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.grain_boundary import (
        _make_engines_from_dirs,
    )

    base = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=LennardJones(),
        working_directory=str(tmp_path),
    )
    dirs = [str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "c")]
    engines = _make_engines_from_dirs.node_function(engine=base, output_dirs=dirs)

    assert len(engines) == 3
    assert [e.working_directory for e in engines] == dirs
    # Original engine is untouched.
    assert base.working_directory == str(tmp_path)
