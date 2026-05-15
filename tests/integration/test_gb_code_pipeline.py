"""Integration tests for the gb_code → grain boundary pipeline.

Exercises the public macros end-to-end with `gb_code` and the constructor
helpers. Marked slow because each call runs a small workflow graph.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms


@pytest.mark.slow
def test_construct_GB_from_GBCode_macro_builds_an_atoms_structure():
    """End-to-end: feed a Σ5 [100] BCC GB through the public macro and
    verify it yields a valid ASE Atoms object with the BCC element symbol."""
    pytest.importorskip("gb_code")
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        construct_GB_from_GBCode,
    )

    fn = construct_GB_from_GBCode(
        axis=[1, 0, 0],
        basis="bcc",
        lattice_param=2.828,
        m=3,
        n=1,
        GB1=(0, 1, -2),
        element="Fe",
        req_length_grain=10.0,
        grain_length_axis=0,
        equil_volume=11.3,
    )()  # invoke the macro

    final = fn["final_structure"]
    original = fn["original_GBcode_structure"]
    assert isinstance(final, Atoms)
    assert isinstance(original, Atoms)
    assert all(s == "Fe" for s in final.get_chemical_symbols())
    assert len(final) > 0
    # ``wrap_and_sort_structure`` produces z-sorted output for the final cell.
    z = final.get_positions()[:, 2]
    assert np.all(np.diff(z) >= -1e-6)


@pytest.mark.slow
def test_construct_GB_from_GBCode_macro_grain_length_extension_in_original():
    """The pre-realignment ``original_GBcode_structure`` is extended along
    ``grain_length_axis`` to at least ``2 · req_length_grain`` (the rearrange
    + align steps may permute axes after that)."""
    pytest.importorskip("gb_code")
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        construct_GB_from_GBCode,
    )

    req = 8.0
    fn = construct_GB_from_GBCode(
        axis=[1, 0, 0],
        basis="bcc",
        lattice_param=2.828,
        m=3,
        n=1,
        GB1=(0, 1, -2),
        element="Fe",
        req_length_grain=req,
        grain_length_axis=0,
        equil_volume=11.3,
    )()
    original = fn["original_GBcode_structure"]
    cell_lengths = np.linalg.norm(original.cell.array, axis=1)
    # At least one cell vector is long enough to host the requested grain.
    assert cell_lengths.max() >= 2 * req - 1e-3
