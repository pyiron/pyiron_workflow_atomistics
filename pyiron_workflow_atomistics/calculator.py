from typing import Any

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from pyiron_snippets.logger import logger


@pwf.as_function_node("calc_output")
def calculate_structure_node(
    structure: Atoms,
    calculation_engine=None,  # Optional[Engine] = None,
    _calc_structure_fn=None,
    _calc_structure_fn_kwargs=None,
) -> Any:
    if calculation_engine is not None:
        calc_structure_fn, calc_structure_fn_kwargs = (
            calculation_engine.get_calculate_fn(structure=structure)
        )
    else:
        calc_structure_fn = _calc_structure_fn
        calc_structure_fn_kwargs = _calc_structure_fn_kwargs
    
    # Safety check: remove 'structure' from kwargs if present to avoid duplicate argument error
    # 'structure' is always passed explicitly, so it should not be in kwargs
    if calc_structure_fn_kwargs is not None:
        calc_structure_fn_kwargs = calc_structure_fn_kwargs.copy()
        calc_structure_fn_kwargs.pop("structure", None)
    
    output = calc_structure_fn(structure=structure, **calc_structure_fn_kwargs)
    return output


@pwf.as_function_node("valid")
def validate_calculation_inputs(
    calc_structure_fn=None,
    calc_structure_fn_kwargs=None,
    calculation_engine=None,
):
    """
    Validates that either:
    - `calculation_engine` is provided, and the others are None, OR
    - both `calc_structure_fn` and `calc_structure_fn_kwargs` are provided, and `calculation_engine` is None.

    Raises
    ------
    ValueError if the inputs do not satisfy the logic.
    """
    valid = False
    using_engine = calculation_engine is not None
    using_fn_and_kwargs = (
        calc_structure_fn is not None and calc_structure_fn_kwargs is not None
    )

    if using_engine and (
        calc_structure_fn is not None or calc_structure_fn_kwargs is not None
    ):
        raise ValueError(
            "If 'calculation_engine' is provided, both 'calc_structure_fn' and 'calc_structure_fn_kwargs' must be None."
        )
    elif not using_engine and not using_fn_and_kwargs:
        raise ValueError(
            "If 'calculation_engine' is not provided, both 'calc_structure_fn' and 'calc_structure_fn_kwargs' must be provided."
        )
    else:
        valid = True
    return valid


@pwf.as_function_node("output_dict")
def convert_EngineOutput_to_output_dict(EngineOutput: Any):
    return EngineOutput.to_dict()


@pwf.as_function_node("output_values")
def extract_output_values_from_EngineOutput(EngineOutput: Any, key: str):
    if isinstance(EngineOutput, list):
        output_dict = [item.to_dict()[key] for item in EngineOutput]
    else:
        output_dict = EngineOutput.to_dict()[key]
    return output_dict


@pwf.as_function_node("output")
def extract_values_from_dict(output_dict: list[dict[str, Any]], key: str):
    """
    Extract a list of values for a specified key from a list of result dictionaries.

    Parameters
    ----------
    results_list : list of dict
        Each dict should contain the specified key.
    key : str
        The dictionary key to extract values for (e.g., 'energy', 'volume').

    Returns
    -------
    values : list
        List of values corresponding to key from each dict.

    Raises
    ------
    KeyError
        If any entry in results_list is missing the specified key.
    """
    try:
        extracted_values = [entry[key] for entry in output_dict]
    except Exception as e:
        logger.error(f"Error {e} when trying to parse output")
        extracted_values = np.nan
    return extracted_values


@pwf.as_function_node("full_calc_kwargs2")
def fillin_default_calckwargs(
    calc_kwargs: dict[str, Any],
    default_values: dict[str, Any] | None | str = None,
    remove_keys: list[str] | None = None,
) -> dict[str, Any]:
    # 1) overlay any user-supplied default overrides
    built_in = {}
    if isinstance(default_values, dict):
        built_in.update(default_values)

    # 2) start with everything user passed in
    full: dict[str, Any] = dict(calc_kwargs)

    # 3) fill in missing built-ins
    for key, default in built_in.items():
        full.setdefault(key, default)

    # 4) ensure properties is a tuple
    if "properties" in full:
        full["properties"] = tuple(full["properties"])

    # 5) remove any keys requested
    if remove_keys:
        for key in remove_keys:
            full.pop(key, None)

    return full


@pwf.as_function_node("kwargs_variant")
def generate_kwargs_variant(
    base_kwargs: dict[str, Any],
    key: str,
    value: Any,
):
    from copy import deepcopy

    kwargs_variant = deepcopy(base_kwargs)
    kwargs_variant[key] = value
    return kwargs_variant


@pwf.as_function_node("kwargs_variants")
def generate_kwargs_variants(
    base_kwargs: dict[str, Any],
    key: str,
    values: list[Any],
):
    """
    Given a base kwargs dict, produce one dict per value in `values`,
    each with `key` set to that value (overriding any existing entry).

    Parameters
    ----------
    base_kwargs
        The original kwargs to copy.
    key
        The dict key whose value you want to vary.
    values
        A list of values to assign to `key`.

    Returns
    -------
    List of dicts
        Each is a shallow copy of base_kwargs with base_kwargs[key] = value.
    """
    return_kwargs = [{**base_kwargs, key: v} for v in values]
    # print(return_kwargs)
    return return_kwargs


@pwf.as_function_node("kwargs_variants_with_remove")
def add_arg_to_kwargs_list(
    kwargs_list: list[dict[str, Any]],
    key: str,
    value: Any | list[Any],
    remove_if_exists: bool = False,
):
    """
    Given a list of kwargs dicts, add a key-value pair to each dict.
    If `value` is a list, the value of the key is the i-th element of the list.
    If `remove_if_exists` is True, the key is removed from the dict if it exists.

    Parameters
    ----------
    kwargs_list: list[dict[str, Any]]
    """
    from copy import deepcopy

    return_kwargs = [deepcopy(kwargs) for kwargs in kwargs_list]
    for i, kwargs in enumerate(return_kwargs):
        if remove_if_exists:
            kwargs.pop(key, None)
        if key in kwargs.keys():
            raise ValueError(f"Key {key} already exists in kwargs dict")
        if isinstance(value, list):
            kwargs[key] = value[i]
        else:
            kwargs[key] = value
    return return_kwargs
