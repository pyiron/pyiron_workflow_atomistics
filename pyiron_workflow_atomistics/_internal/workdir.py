"""Internal helpers for working-directory composition (fan-out)."""

from __future__ import annotations

import os

import pyiron_workflow as pwf

from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dict


@pwf.as_function_node("output_dirs")
def get_subdirpaths(parent_dir: str, output_subdirs: list[str]) -> list[str]:
    output_dirs = [os.path.join(parent_dir, sub) for sub in output_subdirs]
    return output_dirs


@pwf.api.as_function_node("dict_with_adjusted_working_directory")
def get_working_subdir_kwargs(
    calc_structure_fn_kwargs: dict,
    base_working_directory: str,
    new_working_directory: str,
):
    dict_with_adjusted_working_directory = modify_dict.node_function(
        calc_structure_fn_kwargs,
        {
            "working_directory": os.path.join(
                base_working_directory, new_working_directory
            )
        },
    )
    return dict_with_adjusted_working_directory
