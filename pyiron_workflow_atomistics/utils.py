import pyiron_workflow as pwf
from typing import List, Callable
import os
from pyiron_workflow_atomistics.dataclass_storage import Engine
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor


@pwf.as_function_node
def convert_structure(structure, target="ase"):
    if target == "ase":
        converted_structure = AseAtomsAdaptor.get_atoms(structure)
    elif target in ("pmg", "pymatgen"):
        converted_structure = AseAtomsAdaptor.get_structure(structure)
    else:
        raise ValueError(f"Unknown target: {target}")
    return converted_structure


def extract_outputs_from_EngineOutputs(
    engine_outputs: list, keys: list[str], only_converged=True
):
    """
    Extract specified keys from a list of EngineOutput objects.

    Parameters
    ----------
    engine_outputs : list of EngineOutput
        The list of outputs from which to extract values.
    keys : list of str
        The attribute names to extract from each EngineOutput.
    only_converged : bool
        Whether to only include outputs that are converged.

    Returns
    -------
    extracted : dict[str, list]
        Dictionary where each key maps to a list of values extracted from EngineOutput.
    """
    extracted = {key: [] for key in keys}

    for output in engine_outputs:
        if only_converged and not getattr(output, "convergence", False):
            continue
        for key in keys:
            value = getattr(output, key, None)
            extracted[key].append(value)

    return extracted


@pwf.api.as_function_node("dict_with_adjusted_working_directory")
def get_working_subdir_kwargs(
    calc_structure_fn_kwargs: dict,
    base_working_directory: str,
    new_working_directory: str,
):
    return modify_dict.node_function(
        calc_structure_fn_kwargs,
        {
            "working_directory": os.path.join(
                base_working_directory, new_working_directory
            )
        },
    )


@pwf.api.as_function_node("calc_fn", "calc_fn_kwargs")
def get_calc_fn_calc_fn_kwargs_from_calculation_engine(
    calculation_engine, structure, calc_structure_fn, calc_structure_fn_kwargs
):
    if calculation_engine:
        calc_fn, calc_fn_kwargs = calculation_engine.get_calculate_fn(structure)
    else:
        calc_fn = calc_structure_fn
        calc_fn_kwargs = calc_structure_fn_kwargs
    # print(calc_fn_kwargs["working_directory"])
    return calc_fn, calc_fn_kwargs


@pwf.as_function_node("new_string")
def add_string(base_string: str, new_string: str):
    return base_string + new_string


from typing import Any


@pwf.as_function_node("modded_dataclass")
def modify_dataclass(dataclass_instance, entry_name: str, entry_value: Any):
    from dataclasses import asdict
    from copy import deepcopy

    kwarg_dict = {entry_name: entry_value}
    data = deepcopy(asdict(dataclass_instance))  # deep-copies nested containers
    bad = set(kwarg_dict) - data.keys()
    if bad:
        raise KeyError(f"Unknown field(s): {sorted(bad)}")

    data.update(**kwarg_dict)
    dataclass_instance = type(dataclass_instance)(**data)
    # re-construct a brand-new instance from the dict
    return dataclass_instance


@pwf.as_function_node("modded_dataclass_multi")
def modify_dataclass_multi(dataclass_instance, entry_names, entry_values):
    """
    Wraps your single-entry node so you can pass lists of names & values.
    Usage:
      new = modify_dataclass_multi(old, ["a","b"], [1,2])
    """
    if len(entry_names) != len(entry_values):
        raise ValueError("entry_names and entry_values must have the same length")

    ds = dataclass_instance
    for name, val in zip(entry_names, entry_values):
        ds = modify_dataclass.node_function(ds, name, val)
    return ds


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
    per_atom_quantity = quantity / len(structure)
    return per_atom_quantity
