"""Internal helpers for mid-graph mutation of dataclass / dict objects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import pyiron_workflow as pwf


@pwf.as_function_node("modded_dataclass")
def modify_dataclass(dataclass_instance, entry_name: str, entry_value: Any):
    data = deepcopy(asdict(dataclass_instance))
    if entry_name not in data:
        raise KeyError(f"Unknown field: {entry_name!r}")
    data[entry_name] = entry_value
    modded_dataclass = type(dataclass_instance)(**data)
    return modded_dataclass


@pwf.as_function_node("modded_dataclass_multi")
def modify_dataclass_multi(dataclass_instance, entry_names, entry_values):
    if len(entry_names) != len(entry_values):
        raise ValueError("entry_names and entry_values must have the same length")
    ds = dataclass_instance
    for name, val in zip(entry_names, entry_values):
        ds = modify_dataclass.node_function(ds, name, val)
    return ds


@pwf.as_function_node("modded_dict")
def modify_dict(dict_instance: dict, updates: dict) -> dict:
    new = deepcopy(dict_instance)
    invalid = set(updates) - set(new)
    if invalid:
        raise KeyError(f"Unknown key(s): {sorted(invalid)}")
    new.update(updates)
    return new
