"""Derived scalar quantities (per-atom values, etc.)."""

from __future__ import annotations

import pyiron_workflow as pwf


@pwf.as_function_node("per_atom_quantity")
def get_per_atom_quantity(quantity: float, structure) -> float:
    """Divide a total-cell quantity by the number of atoms."""
    per_atom_quantity = quantity / len(structure)
    return per_atom_quantity
