import pyiron_workflow as pwf
from typing import List
import os

@pwf.api.as_function_node("dict_with_adjusted_working_directory")
def get_working_subdir(calc_structure_fn_kwargs: dict, base_working_directory: str, new_working_directory: str):
    return modify_dict.node_function(calc_structure_fn_kwargs, {"working_directory": os.path.join(base_working_directory, new_working_directory)})

@pwf.as_function_node("new_string")
def add_string(base_string: str, new_string: str):
    return base_string + new_string
from typing import Any

@pwf.as_function_node("modded_dataclass")
def modify_dataclass(dataclass_instance, entry_name: str, entry_value: Any):
    from dataclasses import asdict
    from copy import deepcopy
    kwarg_dict = {entry_name: entry_value}
    data = deepcopy(asdict(dataclass_instance))   # deep-copies nested containers
    bad  = set(kwarg_dict) - data.keys()
    if bad:
        raise KeyError(f"Unknown field(s): {sorted(bad)}")

    data.update(**kwarg_dict)
    dataclass_instance = type(dataclass_instance)(**data)
    # re-construct a brand-new instance from the dict
    return dataclass_instance

@pwf.as_function_node("modded_dict")
def modify_dict(dict_instance: dict, updates: dict):
    from copy import deepcopy

    # 1) Clone the whole dict (including nested structures)
    new_dict = deepcopy(dict_instance)

    # 2) Check that every key in updates actually exists in the original
    invalid = set(updates) - set(new_dict)
    if invalid:
        raise KeyError(f"Unknown key(s): {sorted(invalid)}")

    # 3) Apply the updates on the copy
    new_dict.update(updates)

    # 4) Return the modified clone, leaving the original untouched
    return new_dict


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