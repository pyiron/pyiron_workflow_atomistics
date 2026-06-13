"""Smoke test for physics.grain_boundary.pure_gb_study.

Builds a small BCC-Fe S3-RA110 bicrystal (same recipe as
``notebooks/pure_grain_boundary_study.ipynb``), runs the full ``pure_gb_study``
macro under EAM (Al-Fe.eam.fs is shipped with the notebooks), and asserts
that every downstream output is finite. Marked ``slow``: ~30 s of EAM
relaxations.

The test indirectly exercises the entire ``pure_gb_study`` call graph:
``full_gb_length_optimization``, ``gb_length_optimiser``, ``add_vacuum``,
``find_gb_plane``, ``rigid_and_relaxed_cleavage_study``,
``calc_cleavage_GB``, ``cleave_gb_structure``, and ~15 internal helpers
(``_make_engines_with_subdirs``, ``get_extended_struct_list``,
``get_min_energy_structure_from_forloop_df``, ``fit_polynomial_extremum``,
``get_GB_energy``, ``get_GB_exc_volume``, ``_get_area``,
``_get_surface_energy``, ``get_min_energy_from_cleavage_study``,
``generate_deepcopy``, ``get_length``, ``get_concat_df``,
``get_gb_length_optimiser_plot``, ``get_extended_names``,
``modify_dataclass``).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

EAM_PATH = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "Al-Fe.eam.fs"


def _build_s3_ra110_bcc_fe_bicrystal(size_z: int = 3) -> tuple:
    """Recipe lifted from notebooks/pure_grain_boundary_study.ipynb."""
    from ase.build import stack
    from ase.lattice.cubic import BodyCenteredCubic as bcc

    surface1, surface2 = [1, 1, 1], [1, 1, -1]
    rotation_axis = [1, -1, 0]
    lc = 2.85
    v1 = list(-np.cross(rotation_axis, surface1))
    v2 = list(-np.cross(rotation_axis, surface2))
    slab1 = bcc(
        symbol="Fe",
        latticeconstant=lc,
        directions=[rotation_axis, v1, surface1],
        size=[1, 1, size_z],
    )
    slab2 = bcc(
        symbol="Fe",
        latticeconstant=lc,
        directions=[rotation_axis, v2, surface2],
        size=[1, 1, size_z],
    )
    return stack(slab1, slab2), lc


def _bulk_fe_reference(lc: float):
    """Return (e0_per_atom, v0_per_atom) for unit-cell BCC-Fe under the EAM."""
    from ase.calculators.eam import EAM
    from ase.lattice.cubic import BodyCenteredCubic as bcc

    fe = bcc(
        symbol="Fe",
        latticeconstant=lc,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=[1, 1, 1],
    )
    fe.calc = EAM(potential=str(EAM_PATH))
    return fe.get_potential_energy() / len(fe), fe.get_volume() / len(fe)


@pytest.mark.slow
@pytest.mark.skipif(not EAM_PATH.exists(), reason=f"EAM file missing: {EAM_PATH}")
def test_pure_gb_study_runs_end_to_end(tmp_path):
    """End-to-end smoke test: graph constructs, all stages execute, outputs finite."""
    from ase.calculators.eam import EAM

    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputMinimize,
        CalcInputStatic,
    )
    from pyiron_workflow_atomistics.physics._grain_boundary_helpers.dataclass_storage import (
        CleaveGBStructureInput,
        PlotCleaveInput,
    )
    from pyiron_workflow_atomistics.physics.grain_boundary import pure_gb_study

    gb, lc = _build_s3_ra110_bcc_fe_bicrystal(size_z=3)
    e0, v0 = _bulk_fe_reference(lc)
    assert e0 < 0
    assert v0 > 0

    eng_min = ASEEngine(
        # max_iterations must actually be honoured now (the engine no longer
        # silently caps at 10_000 — see ASEEngine.max_steps). The cleavage
        # relaxations here need a few more than 10 BFGS steps to converge at
        # fmax=0.5; 50 keeps the smoke test fast while letting them finish.
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.5, max_iterations=50
        ),
        calculator=EAM(potential=str(EAM_PATH)),
        working_directory=str(tmp_path),
    )
    eng_static = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EAM(potential=str(EAM_PATH)),
        working_directory=str(tmp_path),
    )

    wf = pure_gb_study(
        gb_structure=gb,
        equil_bulk_volume=v0,
        equil_bulk_energy=e0,
        extensions_stage1=[-0.1, 0.0, 0.1],
        extensions_stage2=[-0.05, 0.0, 0.05],
        engine=eng_min,
        static_engine=eng_static,
        length_interpolate_min_n_points=3,
        gb_normal_axis="c",
        vacuum_length=8.0,
        min_inplane_cell_lengths=[None, None, None],
        slab_thickness=3.0,
        tolerance=4.0,
        CleaveGBStructure_Input=CleaveGBStructureInput(
            axis_to_cleave="c", cleave_region_halflength=2.0, separation=3.0
        ),
        PlotCleave_Input=PlotCleaveInput(),
    )
    out = wf.run()

    # --- final length-optimised structure ---
    final_struct = out["final_pure_grain_boundary_structure"]
    final_energy = out["final_pure_grain_boundary_structure_energy"]
    assert len(final_struct) == len(gb)
    assert np.isfinite(final_energy)

    # results_df is a concat of the two stage dataframes (3 + 3 rows)
    assert len(out["grain_boundary_length_optimisation_df"]) == 6

    # GB energy and excess volume from length opt (interpolated, J/m^2 and A^3/A^2).
    assert np.isfinite(out["grain_boundary_energy"])
    assert np.isfinite(out["grain_boundary_excess_volume"])

    # --- vacuum-relaxed structure (used for surface and cleavage) ---
    vac_struct = out["pure_grain_boundary_structure_vacuum"]
    vac_energy = out["pure_grain_boundary_structure_vacuum_energy"]
    assert len(vac_struct) == len(gb)
    assert (
        vac_struct.cell[2, 2] > final_struct.cell[2, 2]
    ), "vacuum should make the c-vector longer than the bulk bicrystal"
    assert np.isfinite(vac_energy)

    # Surface energy from rigid cleavage of the vacuum slab (J/m^2)
    assert np.isfinite(out["surface_energy"])

    # --- GB plane analyser must return a viable dict with a Cartesian coord ---
    gb_dict = out["gb_plane_analysis_dict"]
    assert isinstance(gb_dict, dict)
    assert "gb_cart" in gb_dict and np.isfinite(gb_dict["gb_cart"])
    assert "gb_frac" in gb_dict and 0.0 <= gb_dict["gb_frac"] <= 1.0
    assert (
        len(gb_dict["bulk_indices"]) > 0
    ), "bulk-template sampling found zero atoms — slab_thickness too thin?"

    # --- cleavage stage: both rigid and relaxed dfs are populated ---
    rigid_df = out["work_of_separation_rigid_df"]
    relax_df = out["work_of_separation_relaxed_df"]
    assert len(rigid_df) >= 1, "no cleavage planes were evaluated rigidly"
    assert len(relax_df) >= 1
    assert len(rigid_df) == len(
        relax_df
    ), "rigid and relax must process the same planes"

    # Minimum cleavage energies are finite scalars
    assert np.isfinite(out["work_of_separation_rigid"])
    assert np.isfinite(out["work_of_separation_relaxed"])


@pytest.mark.skipif(not EAM_PATH.exists(), reason=f"EAM file missing: {EAM_PATH}")
def test_pure_gb_study_constructs_graph_without_running():
    """Graph construction alone catches type-hint and channel-shape regressions
    without paying for any EAM relaxation. Runs in milliseconds.
    """
    from ase.calculators.eam import EAM

    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputMinimize,
        CalcInputStatic,
    )
    from pyiron_workflow_atomistics.physics._grain_boundary_helpers.dataclass_storage import (
        CleaveGBStructureInput,
        PlotCleaveInput,
    )
    from pyiron_workflow_atomistics.physics.grain_boundary import pure_gb_study

    gb, lc = _build_s3_ra110_bcc_fe_bicrystal(size_z=2)

    eng_min = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.5, max_iterations=5
        ),
        calculator=EAM(potential=str(EAM_PATH)),
        working_directory=".",
    )
    eng_static = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EAM(potential=str(EAM_PATH)),
        working_directory=".",
    )

    # Graph construction wires up every channel; passes only if every type hint
    # and channel-shape contract along the path is honoured.
    wf = pure_gb_study(
        gb_structure=gb,
        equil_bulk_volume=11.0,
        equil_bulk_energy=-4.0,
        extensions_stage1=[-0.05, 0.0, 0.05],
        extensions_stage2=[-0.02, 0.0, 0.02],
        engine=eng_min,
        static_engine=eng_static,
        CleaveGBStructure_Input=CleaveGBStructureInput(axis_to_cleave="c"),
        PlotCleave_Input=PlotCleaveInput(),
    )
    # Don't run — just check the macro was assembled.
    assert wf is not None
