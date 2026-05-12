"""Structure manipulation — engine-agnostic builders, transformations, defects."""

from .build import create_surface_slab, get_bulk
from .defects import create_vacancy, substitutional_swap
from .transform import (
    add_vacuum,
    create_supercell,
    create_supercell_with_min_dimensions,
    rattle,
)

__all__ = [
    "get_bulk",
    "create_surface_slab",
    "add_vacuum",
    "create_supercell",
    "create_supercell_with_min_dimensions",
    "rattle",
    "create_vacancy",
    "substitutional_swap",
]
