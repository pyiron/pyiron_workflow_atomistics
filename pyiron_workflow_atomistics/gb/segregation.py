import os
from typing import Any

import pyiron_workflow as pwf
from ase import Atoms
from pyiron_workflow.api import for_node

from pyiron_workflow_atomistics.engine import Engine, run


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


@pwf.as_function_node("engines")
def _make_engines_from_dirs(engine: Engine, output_dirs: list) -> list:
    """Return a list of engines, one per output directory path.

    On POSIX, os.path.join(wd, absolute_path) == absolute_path, so passing
    absolute paths to with_working_directory is safe and sets the directory
    correctly.
    """
    return [engine.with_working_directory(d) for d in output_dirs]


import pandas as pd


@pwf.as_function_node("df")
def write_df(df, unique_sites_df, file_name, parent_dir):
    df_out = pd.concat([unique_sites_df, df], axis=1)
    df_out.to_pickle(os.path.join(parent_dir, file_name))
    return df_out


@pwf.as_function_node("unique_sites_list", "df")
def get_unique_sites_SOAP(
    structure: Atoms,
    defect_sites: list[int],
    r_cut: float = 6.0,
    n_max: int = 10,
    l_max: int = 10,
    n_jobs: int = -1,
    periodic: bool = True,
    pca_zca_model: dict | None = None,
    pca_variance_threshold: float = 0.999,
    similarity_threshold: float = 0.99999,
):
    from pyiron_workflow_atomistics.featurisers import (
        pca_whiten,
        soapSiteFeaturiser,
        summarize_cosine_groups,
    )

    a = soapSiteFeaturiser(
        atoms=structure,
        site_indices=defect_sites,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        n_jobs=n_jobs,
        periodic=periodic,
    )
    Z, model = pca_whiten(
        X=a, n_components=pca_variance_threshold, method="pca", model=pca_zca_model
    )
    df = summarize_cosine_groups(
        Z, threshold=similarity_threshold, ids=defect_sites, include_singletons=True
    )
    return df.rep.tolist(), df


@pwf.as_macro_node("gb_seg_calcs_df")
def calculate_substitutional_segregation_GB(
    wf,
    structure: Atoms,
    defect_sites: list[int],
    element: str,
    structure_basename: str,
    engine: Engine,
    unique_sites_df: pd.DataFrame | None = None,
    parent_dir: str = os.path.join(os.getcwd(), "segregation_structures"),
    df_filename: str = "seg_calcs_df.pkl",
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
    wf.gb_seg_engines = _make_engines_from_dirs(
        engine=engine,
        output_dirs=wf.gb_seg_structure_dirs,
    )
    wf.gb_seg_calcs = for_node(
        run,
        zip_on=("structure", "engine"),
        structure=wf.gb_seg_structure_list,
        engine=wf.gb_seg_engines,
    )
    wf.gb_seg_calcs_df = write_df(
        df=wf.gb_seg_calcs.outputs.df,
        unique_sites_df=unique_sites_df,
        file_name=df_filename,
        parent_dir=parent_dir,
    )
    return wf.gb_seg_calcs_df.outputs.df
