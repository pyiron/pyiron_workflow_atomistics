"""Tests for ``structure/transform.py``.

Existing tests in ``test_bulk.py`` and ``test_gb_cleavage.py`` already cover
the happy paths through ``rattle`` and ``add_vacuum(axis='c')``; this file
fills in the remaining branches:

* ``add_vacuum`` with int axis, invalid string axis, invalid non-{str,int} axis
* ``create_supercell`` (never exercised previously)
* ``create_supercell_with_min_dimensions`` default ``min_dimensions``
* ``forloop_function`` (the dataframe-bypass utility)
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

# --- add_vacuum -------------------------------------------------------------


def test_add_vacuum_int_axis_extends_cell():
    from pyiron_workflow_atomistics.structure.transform import add_vacuum

    struct = bulk("Cu", "fcc", a=3.6, cubic=True)
    original_c = struct.cell[2, 2]
    out = add_vacuum.node_function(struct, vacuum_length=10.0, axis=2)
    # ASE's atoms.center(vacuum=5.0) on axis 2 leaves 5 Å on each side of the
    # atomic extent; the resulting cell is longer than the original.
    assert out.cell[2, 2] > original_c


def test_add_vacuum_rejects_invalid_string_axis():
    from pyiron_workflow_atomistics.structure.transform import add_vacuum

    with pytest.raises(ValueError, match="Invalid axis"):
        add_vacuum.node_function(
            bulk("Cu", "fcc", a=3.6, cubic=True), vacuum_length=5.0, axis="z"
        )


def test_add_vacuum_rejects_invalid_axis_type():
    from pyiron_workflow_atomistics.structure.transform import add_vacuum

    with pytest.raises(ValueError, match="Invalid axis"):
        add_vacuum.node_function(
            bulk("Cu", "fcc", a=3.6, cubic=True), vacuum_length=5.0, axis=3.14
        )


def test_add_vacuum_rejects_out_of_range_int_axis():
    from pyiron_workflow_atomistics.structure.transform import add_vacuum

    with pytest.raises(ValueError, match="Invalid axis"):
        add_vacuum.node_function(
            bulk("Cu", "fcc", a=3.6, cubic=True), vacuum_length=5.0, axis=5
        )


# --- create_supercell -------------------------------------------------------


def test_create_supercell_repeats_along_each_axis():
    from pyiron_workflow_atomistics.structure.transform import create_supercell

    base = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = create_supercell.node_function(base, supercell_repeats=(2, 3, 1))
    assert len(out) == len(base) * 6
    # Cell vectors scale by the repeat counts.
    np.testing.assert_allclose(out.cell[0], base.cell[0] * 2)
    np.testing.assert_allclose(out.cell[1], base.cell[1] * 3)
    np.testing.assert_allclose(out.cell[2], base.cell[2])


# --- create_supercell_with_min_dimensions ----------------------------------


def test_supercell_with_min_dimensions_default_min_is_6_6_None():
    """Default min_dimensions=[6, 6, None] means a, b expand to >=6Å, c untouched."""
    from pyiron_workflow_atomistics.structure.transform import (
        create_supercell_with_min_dimensions,
    )

    base = bulk("Cu", "fcc", a=3.6, cubic=True)  # 3.6 Å per axis
    out = create_supercell_with_min_dimensions.node_function(
        base_structure=base, min_dimensions=None
    )
    # 3.6 -> need ceil(6/3.6) = 2 repeats on a, b; c untouched (factor=1).
    assert out.cell[0, 0] == pytest.approx(7.2)
    assert out.cell[1, 1] == pytest.approx(7.2)
    assert out.cell[2, 2] == pytest.approx(3.6)


def test_supercell_with_min_dimensions_keeps_dim_when_already_large_enough():
    from pyiron_workflow_atomistics.structure.transform import (
        create_supercell_with_min_dimensions,
    )

    base = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 1, 1))
    # base now 10.8 Å along a, 3.6 along b/c. Asking for [6,...] on a should keep it.
    out = create_supercell_with_min_dimensions.node_function(
        base_structure=base, min_dimensions=[6.0, 6.0, None]
    )
    assert out.cell[0, 0] == pytest.approx(10.8)
    assert out.cell[1, 1] == pytest.approx(7.2)


# --- forloop_function -------------------------------------------------------


def test_forloop_function_iterates_one_kwarg():
    from pyiron_workflow_atomistics.structure.transform import forloop_function

    def compute(a, b, scale=1):
        return scale * (a + b)

    results = forloop_function.node_function(
        function=compute,
        kwarg_to_iterate="a",
        kwarg_values=[1, 2, 3],
        other_kwargs={"b": 10, "scale": 0.5},
    )
    assert results == [5.5, 6.0, 6.5]


def test_forloop_function_defaults_other_kwargs_to_empty_dict():
    from pyiron_workflow_atomistics.structure.transform import forloop_function

    def square(x):
        return x * x

    results = forloop_function.node_function(
        function=square, kwarg_to_iterate="x", kwarg_values=[2, 3, 4]
    )
    assert results == [4, 9, 16]


def test_forloop_function_handles_empty_kwarg_values():
    from pyiron_workflow_atomistics.structure.transform import forloop_function

    results = forloop_function.node_function(
        function=lambda x: x, kwarg_to_iterate="x", kwarg_values=[]
    )
    assert results == []
