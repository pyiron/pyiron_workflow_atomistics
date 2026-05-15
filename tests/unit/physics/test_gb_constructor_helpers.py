"""Tier 1 helpers for `physics._grain_boundary_code.constructor`.

The constructor's small pure helpers (axis-label, structure conversion,
rearrange-by-axes) are tested directly; the macro `construct_GB_from_GBCode`
is exercised once end-to-end as an integration test in
``tests/integration/test_gb_code_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from pymatgen.core import Lattice, Structure

# ---------------------------------------------------------------------------
# _axis_index_to_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis, expected", [(0, "x"), (1, "y"), (2, "z")])
def test_axis_index_to_label_valid(axis, expected):
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        _axis_index_to_label,
    )

    assert _axis_index_to_label(axis) == expected


@pytest.mark.parametrize("bad", [3, -1, "x", None, 1.5, True, False])
def test_axis_index_to_label_rejects_bad_input(bad):
    """Floats (even integer-valued ones), bools, and non-integers must all
    raise rather than silently truncate to a valid axis index."""
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        _axis_index_to_label,
    )

    with pytest.raises(ValueError, match="0, 1, or 2"):
        _axis_index_to_label(bad)


def test_axis_index_to_label_accepts_numpy_integers():
    """The guard must still accept numpy integer scalars, which are used
    throughout the codebase as axis indices."""
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        _axis_index_to_label,
    )

    assert _axis_index_to_label(np.int64(0)) == "x"
    assert _axis_index_to_label(np.int32(2)) == "z"


# ---------------------------------------------------------------------------
# wrap_and_sort_structure
# ---------------------------------------------------------------------------


def test_wrap_and_sort_structure_orders_by_axis():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        wrap_and_sort_structure,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    out = wrap_and_sort_structure.node_function(cu, axis=2)
    z = out.get_positions()[:, 2]
    assert np.all(np.diff(z) >= -1e-9)


def test_wrap_and_sort_structure_with_axis_x():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        wrap_and_sort_structure,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    out = wrap_and_sort_structure.node_function(cu, axis=0)
    x = out.get_positions()[:, 0]
    assert np.all(np.diff(x) >= -1e-9)


# ---------------------------------------------------------------------------
# get_multiplier_to_extend_gb_to_min_length
# ---------------------------------------------------------------------------


def test_multiplier_extends_axis_to_at_least_2x_req_length():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        get_multiplier_to_extend_gb_to_min_length,
    )

    struct = Structure(
        Lattice.cubic(3.5),
        ["Fe"],
        [[0, 0, 0]],
    )
    factors = get_multiplier_to_extend_gb_to_min_length(
        struct, axis=0, req_length_grain=15
    )
    assert factors[0] * 3.5 >= 2 * 15
    assert factors[1] == 1
    assert factors[2] == 1


def test_multiplier_returns_1_when_already_long_enough():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        get_multiplier_to_extend_gb_to_min_length,
    )

    struct = Structure(Lattice.cubic(40.0), ["Fe"], [[0, 0, 0]])
    factors = get_multiplier_to_extend_gb_to_min_length(
        struct, axis=2, req_length_grain=15
    )
    assert factors == [1, 1, 1]


# ---------------------------------------------------------------------------
# rearrange_structure_lattice_vectors
# ---------------------------------------------------------------------------


def test_rearrange_structure_lattice_vectors_swap_a_and_b():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        rearrange_structure_lattice_vectors,
    )

    s = Structure(
        Lattice.from_parameters(2.0, 4.0, 6.0, 90, 90, 90),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    out = rearrange_structure_lattice_vectors(s, order=("b", "a", "c"))
    # The new 'a' length matches the original 'b' length and vice versa.
    assert out.lattice.a == pytest.approx(4.0)
    assert out.lattice.b == pytest.approx(2.0)
    assert out.lattice.c == pytest.approx(6.0)


def test_rearrange_structure_lattice_vectors_invalid_order_raises():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        rearrange_structure_lattice_vectors,
    )

    s = Structure(Lattice.cubic(3.5), ["Fe"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="permutation"):
        rearrange_structure_lattice_vectors(s, order=("a", "a", "c"))


# ---------------------------------------------------------------------------
# align_lattice_to_axes
# ---------------------------------------------------------------------------


def test_align_lattice_to_axes_produces_diagonal_lattice():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        align_lattice_to_axes,
    )

    s = Structure(
        Lattice.from_parameters(2.0, 4.0, 6.0, 90, 90, 90),
        ["Fe", "Cu"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    out = align_lattice_to_axes(s)
    # Off-diagonal entries of the new lattice matrix must be zero.
    M = np.asarray(out.lattice.matrix)
    off_diag = M - np.diag(np.diag(M))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-9)
    # Diagonal entries match the original abc.
    np.testing.assert_allclose(np.diag(M), [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# convert_structure
# ---------------------------------------------------------------------------


def test_convert_structure_ase_to_pymatgen_and_back():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        convert_structure,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    pmg = convert_structure.node_function(cu, target="pmg")
    assert isinstance(pmg, Structure)
    ase_again = convert_structure.node_function(pmg, target="ase")
    np.testing.assert_allclose(ase_again.get_positions(), cu.get_positions(), atol=1e-9)


def test_convert_structure_pymatgen_alias_works():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        convert_structure,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    pmg = convert_structure.node_function(cu, target="pymatgen")
    assert isinstance(pmg, Structure)


def test_convert_structure_invalid_target_raises():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        convert_structure,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    with pytest.raises(ValueError, match="Not a valid conversion target"):
        convert_structure.node_function(cu, target="nope")


# ---------------------------------------------------------------------------
# get_expected_equilibrium_c_struct
# ---------------------------------------------------------------------------


def test_get_expected_equilibrium_c_struct_scales_chosen_axis():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        get_expected_equilibrium_c_struct,
    )

    s = Structure(Lattice.cubic(3.5), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    v0 = 11.5  # target volume per atom in Å³
    new_struct, new_axis_length = get_expected_equilibrium_c_struct.node_function(
        s, v0_per_atom=v0, axis=2
    )
    # V = V0 * N_atoms; orthogonal lattice with a*b=12.25 → c = N*v0 / (a*b)
    expected = (v0 * 2) / (3.5 * 3.5)
    assert new_axis_length == pytest.approx(expected)
    assert new_struct.lattice.c == pytest.approx(expected)
    # The two orthogonal axes stay unchanged.
    assert new_struct.lattice.a == pytest.approx(3.5)
    assert new_struct.lattice.b == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# merge_structure_sites
# ---------------------------------------------------------------------------


def test_merge_structure_sites_collapses_near_duplicates():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        merge_structure_sites,
    )

    s = Structure(
        Lattice.cubic(10.0),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]],
    )
    merged = merge_structure_sites.node_function(
        s, merge_dist_tolerance=0.5, merge_mode="average"
    )
    assert len(merged) == 1


def test_merge_structure_sites_keeps_distinct_sites():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        merge_structure_sites,
    )

    s = Structure(
        Lattice.cubic(10.0),
        ["Fe", "Cu"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    merged = merge_structure_sites.node_function(
        s, merge_dist_tolerance=0.5, merge_mode="average"
    )
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# get_realigned_structure
# ---------------------------------------------------------------------------


def test_get_realigned_structure_runs_with_default_options():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        get_realigned_structure,
    )

    s = Structure(
        Lattice.from_parameters(3.5, 4.0, 5.0, 90, 90, 90),
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    out = get_realigned_structure.node_function(s)
    assert isinstance(out, Structure)
    M = np.asarray(out.lattice.matrix)
    off_diag = M - np.diag(np.diag(M))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-9)


def test_get_realigned_structure_with_equivalence_check():
    """The ``perform_equiv_check`` branch logs whether the structures match;
    just verify it runs without raising."""
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        get_realigned_structure,
    )

    s = Structure(
        Lattice.from_parameters(3.5, 4.0, 5.0, 90, 90, 90),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    out = get_realigned_structure.node_function(s, perform_equiv_check=True)
    assert isinstance(out, Structure)


def test_get_realigned_structure_no_ab_reorder():
    from pyiron_workflow_atomistics.physics._grain_boundary_code.constructor import (
        get_realigned_structure,
    )

    s = Structure(Lattice.cubic(3.5), ["Fe"], [[0.0, 0.0, 0.0]])
    out = get_realigned_structure.node_function(s, arrange_ab_by_length=False)
    assert isinstance(out, Structure)
