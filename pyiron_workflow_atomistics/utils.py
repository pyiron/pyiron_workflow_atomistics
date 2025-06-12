import pyiron_workflow as pwf
from typing import List
import os

@pwf.as_function_node
def update_dataclass(dataclass_instance, key, value):
    from dataclasses import replace, asdict

    if key not in asdict(dataclass_instance):
        raise KeyError(
            f"Field '{key}' not in dataclass {type(dataclass_instance).__name__}"
        )

    updated_dataclass = replace(dataclass_instance, **{key: value})
    return updated_dataclass


@pwf.as_function_node("output_dirs")
def get_subdirpaths(parent_dir: str, output_subdirs: List[str]):
    """
    Generate a list of working directory paths for each calculation.

    Parameters
    ----------
    parent_dir : str
        Base working directory path.
    output_subdirs : list of str
        List of subdirectory names to append to the base working_directory.

    Returns
    -------
    dirpaths : list of str
        List of full paths for each subdirectory.
    """
    dirpaths = []
    for sub in output_subdirs:
        output_subdir = os.path.join(parent_dir, sub)
        dirpaths.append(output_subdir)
    return dirpaths