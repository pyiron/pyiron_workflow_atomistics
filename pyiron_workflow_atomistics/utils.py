import pyiron_workflow as pwf
from typing import List
import os


@pwf.as_function_node
def modify_dataclass(dataclass_instance, **kwargs):
    from dataclasses import asdict
    from copy import deepcopy

    data = deepcopy(asdict(dataclass_instance))   # deep-copies nested containers
    bad  = set(kwargs) - data.keys()
    if bad:
        raise KeyError(f"Unknown field(s): {sorted(bad)}")

    data.update(kwargs)
    # re-construct a brand-new instance from the dict
    return type(dataclass_instance)(**data)




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

@pwf.as_function_node("per_atom_quantity")
def get_per_atom_quantity(quantity, structure):
    per_atom_quantity = quantity/len(structure)
    return per_atom_quantity