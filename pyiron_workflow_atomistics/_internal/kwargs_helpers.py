"""Internal helpers for fan-out parameter sweeps. NOT part of the public API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pyiron_workflow as pwf


@pwf.as_function_node("full_calc_kwargs2")
def fillin_default_calckwargs(
    calc_kwargs: dict[str, Any],
    default_values: dict[str, Any] | None = None,
    remove_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Merge user kwargs with defaults; coerce ``properties`` to tuple; drop keys."""
    built_in: dict[str, Any] = (
        dict(default_values) if isinstance(default_values, dict) else {}
    )
    full: dict[str, Any] = dict(calc_kwargs)
    for key, default in built_in.items():
        full.setdefault(key, default)
    if "properties" in full:
        full["properties"] = tuple(full["properties"])
    if remove_keys:
        for key in remove_keys:
            full.pop(key, None)
    return full


@pwf.as_function_node("kwargs_variant")
def generate_kwargs_variant(
    base_kwargs: dict[str, Any], key: str, value: Any
) -> dict[str, Any]:
    """Return a deepcopy of ``base_kwargs`` with ``key`` set to ``value``."""
    out = deepcopy(base_kwargs)
    out[key] = value
    return out


@pwf.as_function_node("kwargs_variants")
def generate_kwargs_variants(
    base_kwargs: dict[str, Any], key: str, values: list[Any]
) -> list[dict[str, Any]]:
    """Return one variant per element of ``values`` with ``key`` overridden."""
    variants = [{**base_kwargs, key: v} for v in values]
    return variants
