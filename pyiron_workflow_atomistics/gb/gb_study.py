# Standard library imports
from typing import Callable, Any

# Local imports
import pyiron_workflow as pwf
from pyiron_workflow_atomistics.featurisers import voronoiSiteFeaturiser


@pwf.api.as_function_node("surface_energy")
def _get_surface_energy(total_energy_gb_vac, total_energy_gb_novac, area):
    surface_energy = (
        (total_energy_gb_vac - total_energy_gb_novac) / area * 16.021766208 / 2
    )
    return surface_energy


@pwf.api.as_function_node("area")
def _get_area(gb_with_vacuum_rel, axis="c"):
    from pyiron_workflow_atomistics.gb.utils import axis_to_index

    axis = axis_to_index(axis)
    area = gb_with_vacuum_rel.cell.volume / gb_with_vacuum_rel.cell[axis][axis]
    return area


@pwf.api.as_macro_node(
    "final_pure_grain_boundary_structure",
    "final_pure_grain_boundary_structure_energy",
    "grain_boundary_length_optimisation_df",
    "grain_boundary_energy",
    "grain_boundary_excess_volume",
    "surface_energy",
    "pure_grain_boundary_structure_vacuum",
    "pure_grain_boundary_structure_vacuum_energy",
    "gb_plane_analysis_dict",
    "work_of_separation_rigid",
    "work_of_separation_rigid_df",
    "work_of_separation_relaxed",
    "work_of_separation_relaxed_df",
)
def pure_gb_study(
    wf,
    gb_structure,
    equil_bulk_volume,
    equil_bulk_energy,
    extensions_stage1,
    extensions_stage2,
    calculation_engine,
    static_calculation_engine,
    calc_structure_fn=None,
    calc_structure_fn_kwargs=None,
    static_calc_structure_fn=None,
    static_calc_structure_fn_kwargs=None,
    length_interpolate_min_n_points=5,
    gb_normal_axis="c",
    vacuum_length=20,
    min_inplane_cell_lengths=[6, 6, None],
    featuriser=voronoiSiteFeaturiser,
    approx_frac=0.5,
    tolerance=5.0,
    bulk_offset=10.0,
    slab_thickness=2.0,
    featuriser_kwargs=None,
    n_bulk=10,
    threshold_frac=0.3,
    CleaveGBStructure_Input=None,
    PlotCleave_Input=None,
):
    from pyiron_workflow_atomistics.calculator import validate_calculation_inputs

    wf.validate = validate_calculation_inputs(
        calculation_engine=calculation_engine,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )
    wf.validate_static = validate_calculation_inputs(
        calculation_engine=static_calculation_engine,
        calc_structure_fn=static_calc_structure_fn,
        calc_structure_fn_kwargs=static_calc_structure_fn_kwargs,
    )
    from pyiron_workflow_atomistics.utils import (
        get_calc_fn_calc_fn_kwargs_from_calculation_engine,
    )

    wf.calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(
        calculation_engine=calculation_engine,
        structure=gb_structure,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )
    wf.static_calc_fn_calc_fn_kwargs = (
        get_calc_fn_calc_fn_kwargs_from_calculation_engine(
            calculation_engine=static_calculation_engine,
            structure=gb_structure,
            calc_structure_fn=static_calc_structure_fn,
            calc_structure_fn_kwargs=static_calc_structure_fn_kwargs,
        )
    )
    from pyiron_workflow_atomistics.utils import get_working_subdir_kwargs

    wf.calc_structure_fn_kwargs_gb_length_optimiser = get_working_subdir_kwargs(
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        base_working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        new_working_directory="gb_length_optimiser",
    )
    from pyiron_workflow_atomistics.gb.optimiser import full_gb_length_optimization

    wf.gb_length_optimiser = full_gb_length_optimization(
        gb_structure=gb_structure,
        calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        calc_structure_fn_kwargs=wf.calc_structure_fn_kwargs_gb_length_optimiser,
        equil_bulk_volume=equil_bulk_volume,
        equil_bulk_energy=equil_bulk_energy,
        extensions_stage1=extensions_stage1,
        extensions_stage2=extensions_stage2,
        interpolate_min_n_points=length_interpolate_min_n_points,
        gb_normal_axis=gb_normal_axis,
    )
    from pyiron_workflow_atomistics.structure_manipulator.tools import add_vacuum

    wf.gb_with_vacuum = add_vacuum(
        wf.gb_length_optimiser.outputs.gb_structure_final,
        vacuum_length=vacuum_length,
        axis=gb_normal_axis,
    )
    wf.calc_structure_fn_kwargs_gb_with_vacuum_rel = get_working_subdir_kwargs(
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        base_working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        new_working_directory="gb_with_vacuum_rel",
    )
    from pyiron_workflow_atomistics.calculator import calculate_structure_node

    wf.gb_with_vacuum_rel = calculate_structure_node(
        structure=wf.gb_with_vacuum,
        _calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        _calc_structure_fn_kwargs=wf.calc_structure_fn_kwargs_gb_with_vacuum_rel,
    )
    from pyiron_workflow_atomistics.structure_manipulator.tools import (
        create_supercell_with_min_dimensions,
    )

    wf.gb_seg_supercell = create_supercell_with_min_dimensions(
        wf.gb_with_vacuum_rel.outputs.calc_output.final_structure,
        min_dimensions=min_inplane_cell_lengths,
    )
    wf.calc_structure_fn_kwargs_gb_seg_supercell = get_working_subdir_kwargs(
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        base_working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        new_working_directory="gb_seg_supercell",
    )
    wf.gb_seg_supercell_rel = calculate_structure_node(
        structure=wf.gb_seg_supercell,
        _calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        _calc_structure_fn_kwargs=wf.calc_structure_fn_kwargs_gb_seg_supercell,
    )
    wf.area = _get_area(
        wf.gb_with_vacuum_rel.outputs.calc_output.final_structure, gb_normal_axis
    )
    wf.surface_energy = _get_surface_energy(
        wf.gb_with_vacuum_rel.outputs.calc_output.final_energy,
        wf.gb_length_optimiser.outputs.gb_structure_final_energy,
        wf.area,
    )
    from pyiron_workflow_atomistics.gb.analysis import find_GB_plane, plot_GB_plane

    wf.gb_plane_extractor = find_GB_plane(
        atoms=wf.gb_with_vacuum_rel.outputs.calc_output.final_structure,
        featuriser=featuriser,
        axis=gb_normal_axis,
        approx_frac=approx_frac,
        tolerance=tolerance,
        bulk_offset=bulk_offset,
        slab_thickness=slab_thickness,
        featuriser_kwargs=featuriser_kwargs,
        n_bulk=n_bulk,
        threshold_frac=threshold_frac,
    )
    wf.gb_plane_extractor_plot = plot_GB_plane(
        atoms=wf.gb_with_vacuum_rel.outputs.calc_output.final_structure,
        res=wf.gb_plane_extractor.outputs.gb_plane_analysis_dict,
        projection=(0, 2),
        reps=(5, 1),
        figsize=(10, 6),
        bulk_color="C0",
        window_cmap="viridis",
        plane_linestyles=("--", "-"),
        axis=2,
        dpi=300,
        working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        save_filename=f"pureGB_plane_identifier.jpg",
    )
    wf.calc_structure_fn_kwargs_cleavage_study = get_working_subdir_kwargs(
        calc_structure_fn_kwargs=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        base_working_directory=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        new_working_directory="cleavage_study",
    )

    wf.calc_structure_fn_kwargs_cleavage_study_static = get_working_subdir_kwargs(
        calc_structure_fn_kwargs=wf.static_calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs,
        base_working_directory=wf.static_calc_fn_calc_fn_kwargs.outputs.calc_fn_kwargs[
            "working_directory"
        ],
        new_working_directory="cleavage_study",
    )
    from pyiron_workflow_atomistics.utils import modify_dataclass

    wf.CleaveGBStructureInput = modify_dataclass(
        CleaveGBStructure_Input,
        "cleavage_target_coord",
        wf.gb_plane_extractor.outputs.gb_plane_analysis_dict["gb_cart"],
    )
    from pyiron_workflow_atomistics.gb.cleavage import rigid_and_relaxed_cleavage_study

    wf.cleavage_study = rigid_and_relaxed_cleavage_study(
        gb_structure=wf.gb_with_vacuum_rel.outputs.calc_output.final_structure,
        gb_structure_energy=wf.gb_with_vacuum_rel.outputs.calc_output.final_energy,
        gb_plane_cart_loc=wf.gb_plane_extractor.outputs.gb_plane_analysis_dict[
            "gb_cart"
        ],
        calc_structure_fn=wf.calc_fn_calc_fn_kwargs.outputs.calc_fn,
        calc_structure_fn_kwargs=wf.calc_structure_fn_kwargs_cleavage_study,
        static_calc_structure_fn=wf.static_calc_fn_calc_fn_kwargs.outputs.calc_fn,
        static_calc_structure_fn_kwargs=wf.calc_structure_fn_kwargs_cleavage_study_static,
        CleaveGBStructure_Input=wf.CleaveGBStructureInput,
        PlotCleave_Input=PlotCleave_Input,
    )
    wf.min_rigid_cleavage_energy = get_min_energy_from_cleavage_study(
        wf.cleavage_study.outputs.cleavage_results_rigid
    )
    wf.min_relaxed_cleavage_energy = get_min_energy_from_cleavage_study(
        wf.cleavage_study.outputs.cleavage_results_relax
    )
    return (
        wf.gb_length_optimiser.outputs.gb_structure_final,
        wf.gb_length_optimiser.outputs.gb_structure_final_energy,
        wf.gb_length_optimiser.outputs.results_df,
        wf.gb_length_optimiser.outputs.stage2_opt_GBEnergy,
        wf.gb_length_optimiser.outputs.stage2_opt_excvol,
        wf.surface_energy,
        wf.gb_with_vacuum_rel.outputs.calc_output.final_structure,
        wf.gb_with_vacuum_rel.outputs.calc_output.final_energy,
        wf.gb_plane_extractor.outputs.gb_plane_analysis_dict,
        wf.min_rigid_cleavage_energy,
        wf.cleavage_study.outputs.cleavage_results_rigid,
        wf.min_relaxed_cleavage_energy,
        wf.cleavage_study.outputs.cleavage_results_relax,
    )


@pwf.api.as_function_node("min_energy")
def get_min_energy_from_cleavage_study(cleavage_study_df):
    min_energy = cleavage_study_df.cleavage_energy.min()
    return min_energy
