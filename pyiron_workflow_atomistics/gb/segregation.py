import pyiron_workflow as pwf
import os
from ase import Atoms
from pyiron_workflow.api import for_node
from pyiron_workflow_atomistics.calculator import (
    fillin_default_calckwargs,
    generate_kwargs_variants,
)
from typing import Any, Callable
from pyiron_workflow_atomistics.calculator import calculate_structure_node
from os import getcwd
from pyiron_workflow_atomistics.dataclass_storage import Engine

@pwf.as_function_node("structure", "output_dir")
def create_seg_structure_and_output_dir(
    structure: Atoms,
    defect_site: int,
    element: str,
    structure_basename: str,
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
):
    # print("In create_seg_structure_and_output_dir")
    seg_structure = structure.copy()
    seg_structure[defect_site].symbol = element
    structure_name = f"{structure_basename}_{element}_{defect_site}"
    output_dir = os.path.join(parent_dir, structure_name)
    # print("Exiting create_seg_structure_and_output_dir")
    return seg_structure, output_dir


@pwf.as_function_node
def get_df_col_as_list(df, col):
    # print("In get_df_col_as_list")
    output_list = df[col].to_list()
    return output_list

@pwf.as_function_node
def get_calc_fn_calc_fn_kwargs_list_from_calculation_engine(calculation_engine, structure_list, calc_structure_fn, calc_structure_fn_kwargs):
    calc_fn_list = []
    calc_fn_kwargs_list = []
    if calculation_engine:
        # print("In get_calc_fn_calc_fn_kwargs_list_from_calculation_engine")
        for structure in structure_list:
            # print(structure)
            calculation_engine.calc_fn = calc_structure_fn
            calc_fn, calc_fn_kwargs = calculation_engine.calculate_fn(structure)
            # print(calc_fn_kwargs["potential_elements"])
            calc_fn_list.append(calc_fn)
            calc_fn_kwargs_list.append(calc_fn_kwargs)
    else:
        import warnings
        warnings.warn("No calculation engine provided, using calc_structure_fn and calc_structure_fn_kwargs...\n This is VERY DANGEROUS and I hope you know what you are doing")
        if isinstance(calc_structure_fn, list) and isinstance(calc_structure_fn_kwargs, list):
            calc_fn_list = calc_structure_fn
            calc_fn_kwargs_list = calc_structure_fn_kwargs
        elif isinstance(calc_structure_fn, Callable) and isinstance(calc_structure_fn_kwargs, dict):
            for structure in structure_list:
                calc_fn_list.append(calc_structure_fn)
                calc_fn_kwargs_list.append(calc_structure_fn_kwargs)
        else:
            raise ValueError("calc_structure_fn and calc_structure_fn_kwargs must be either a list of Callables and dicts or a single Callable and dict")
    return calc_fn_list, calc_fn_kwargs_list

# @pwf.as_macro_node("gb_seg_calcs_kwargs", "gb_seg_calcs")
# def featurise_sites(wf, structure_list, defect_sites, calc_fn_kwargs_list):
#     from pyiron_workflow_atomistics.featurisers import distanceMatrixSiteFeaturiser, voronoiSiteFeaturiser
#     wf.featurised_sites = for_node(
#         distanceMatrixSiteFeaturiser,
#         zip_on=("structure", "defect_sites"),
#         structure=structure_list,
#         defect_sites=defect_sites,
#     )
#     return wf.featurised_sites

@pwf.as_function_node("df")
def write_df(df, unique_sites_df, file_name, parent_dir):
    df = df.drop(columns=["_calc_structure_fn", "_calc_structure_fn_kwargs"])
    df_out = pd.concat([unique_sites_df, df], axis=1)
    df_out.to_pickle(os.path.join(parent_dir, file_name))
    return df_out

@pwf.as_function_node("unique_sites_list", "df")
def get_unique_sites_SOAP(structure: Atoms,
                          defect_sites: list[int],
                          r_cut: float = 6.0,
                          n_max: int = 10,
                          l_max: int = 10,
                          n_jobs: int = -1,
                          periodic: bool = True,
                          pca_zca_model: dict | None = None,
                          pca_variance_threshold: float = 0.999,
                          similarity_threshold: float = 0.99999):
  from pyiron_workflow_atomistics.featurisers import soapSiteFeaturiser, pca_whiten, summarize_cosine_groups
  a = soapSiteFeaturiser(atoms = structure,
                       site_indices= defect_sites,
                       r_cut=r_cut,
                       n_max=n_max,
                       l_max=l_max,
                       n_jobs=n_jobs,
                       periodic=periodic)
  Z, model = pca_whiten(X=a,
           n_components=pca_variance_threshold,
           method="pca",
           model=pca_zca_model)
  df = summarize_cosine_groups(Z, threshold=similarity_threshold, ids=defect_sites, include_singletons=True)
  return df.rep.tolist(), df

import pandas as pd
@pwf.as_macro_node("gb_seg_calcs_df")
def calculate_substitutional_segregation_GB(
    wf,
    structure: Atoms,
    defect_sites: list[int],
    element: str,
    structure_basename: str,
    calculation_engine: Engine | None = None,
    calc_structure_fn: Callable[..., Any] | None | list[Callable[..., Any]] = None,
    calc_structure_fn_kwargs: dict[str, Any] | None | list[dict[str, Any]] = None,
    unique_sites_df: pd.DataFrame | None = None,
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
    df_filename: str = "seg_calcs_df.pkl",
):
    from pyiron_workflow_atomistics.calculator import validate_calculation_inputs   
    wf.validate = validate_calculation_inputs(calculation_engine = calculation_engine,
                                              calc_structure_fn = calc_structure_fn,
                                              calc_structure_fn_kwargs = calc_structure_fn_kwargs)

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
    # from pyiron_workflow_atomistics.gb.segregation import get_calc_fn_calc_fn_kwargs_list_from_calculation_engine
    wf.calc_fn_calc_fn_kwargs_list = get_calc_fn_calc_fn_kwargs_list_from_calculation_engine(
        calculation_engine = calculation_engine,
        structure_list = wf.gb_seg_structure_list,
        calc_structure_fn = calc_structure_fn,
        calc_structure_fn_kwargs = calc_structure_fn_kwargs
    )
    from pyiron_workflow_atomistics.calculator import add_arg_to_kwargs_list
    wf.gb_seg_calcs_kwargs = add_arg_to_kwargs_list(
        kwargs_list=wf.calc_fn_calc_fn_kwargs_list.outputs.calc_fn_kwargs_list,
        key="working_directory",
        value=wf.gb_seg_structure_dirs,
        remove_if_exists=True,
    )
    wf.gb_seg_calcs = for_node(
        calculate_structure_node,
        zip_on=("structure", "_calc_structure_fn_kwargs", "_calc_structure_fn"),
        structure=wf.gb_seg_structure_list,
        _calc_structure_fn=wf.calc_fn_calc_fn_kwargs_list.outputs.calc_fn_list,
        _calc_structure_fn_kwargs=wf.gb_seg_calcs_kwargs,
    )
    wf.gb_seg_calcs_df = write_df(
        df=wf.gb_seg_calcs.outputs.df,
        unique_sites_df=unique_sites_df,
        file_name=df_filename,
        parent_dir=parent_dir,
    )
    return wf.gb_seg_calcs_df.outputs.df
