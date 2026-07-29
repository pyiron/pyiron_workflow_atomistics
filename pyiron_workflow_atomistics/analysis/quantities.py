"""Derived scalar quantities (per-atom values, etc.)."""

from __future__ import annotations

import flowrep as fr


@fr.atomic("per_atom_quantity")
def get_per_atom_quantity(quantity: float, structure) -> float:
    """Divide a total-cell quantity by the number of atoms."""
    per_atom_quantity = quantity / len(structure)
    return per_atom_quantity
