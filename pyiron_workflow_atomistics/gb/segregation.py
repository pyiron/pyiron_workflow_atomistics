import pyiron_workflow as pwf
import os
from ase import Atoms
from pyiron_workflow.api import for_node
from pyiron_workflow_atomistics.calculator import (
    fillin_default_calckwargs,
    generate_kwargs_variants,
    ase_calculate_structure_node_interface,
)
from typing import Any, Callable
from pyiron_workflow_atomistics.calculator import calculate_structure_node
from os import getcwd


@pwf.as_function_node("structure", "output_dir")
def create_seg_structure_and_output_dir(
    structure: Atoms,
    defect_site: int,
    element: str,
    structure_basename: str,
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
):
    seg_structure = structure.copy()
    seg_structure[defect_site].symbol = element
    structure_name = f"{structure_basename}_{element}_{defect_site}"
    output_dir = os.path.join(parent_dir, structure_name)
    print(os.getcwd())
    print(output_dir)
    return seg_structure, output_dir


@pwf.as_function_node
def get_df_col_as_list(df, col):
    output_list = df[col].to_list()
    return output_list


@pwf.as_macro_node("output_df")
def calculate_segregation_GB(
    wf,
    structure: Atoms,
    defect_sites: list[int],
    element: str,
    structure_basename: str,
    calc_structure_fn: Callable[..., Any],
    calc_kwargs: dict[str, Any],
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
):
    wf.gb_seg_structure_generator = for_node(
        create_seg_structure_and_output_dir,
        structure=structure,
        iter_on=("defect_site"),
        defect_site=defect_sites,
        structure_basename=structure_basename,
        element=element,
        parent_dir=parent_dir,
    )
    wf.gb_seg_structure_list = get_df_col_as_list(
        wf.gb_seg_structure_generator.outputs.df, "structure"
    )
    wf.gb_seg_structure_dirs = get_df_col_as_list(
        wf.gb_seg_structure_generator.outputs.df, "output_dir"
    )
    wf.kwargs_removed_working_directory = fillin_default_calckwargs(
        calc_kwargs=calc_kwargs, default_values=None, remove_keys=["working_directory"]
    )
    wf.gb_seg_calcs_kwargs = generate_kwargs_variants(
        base_kwargs=wf.kwargs_removed_working_directory,
        key="working_directory",
        values=wf.gb_seg_structure_dirs,
    )
    wf.gb_seg_calcs = for_node(
        calculate_structure_node,
        zip_on=("structure", "calc_structure_fn_kwargs"),
        structure=wf.gb_seg_structure_list,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=wf.gb_seg_calcs_kwargs,
    )
    return wf.gb_seg_calcs
