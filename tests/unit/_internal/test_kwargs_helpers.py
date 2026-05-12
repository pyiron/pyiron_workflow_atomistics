"""Tests for the private kwargs-plumbing helpers."""

from __future__ import annotations


def test_fillin_default_calckwargs_merges_defaults():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import (
        fillin_default_calckwargs,
    )

    out = fillin_default_calckwargs.node_function(
        calc_kwargs={"a": 1},
        default_values={"a": 0, "b": 2},
    )
    assert out == {"a": 1, "b": 2}


def test_fillin_default_calckwargs_drops_keys():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import (
        fillin_default_calckwargs,
    )

    out = fillin_default_calckwargs.node_function(
        calc_kwargs={"a": 1, "secret": 2},
        remove_keys=["secret"],
    )
    assert "secret" not in out


def test_fillin_default_calckwargs_coerces_properties_to_tuple():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import (
        fillin_default_calckwargs,
    )

    out = fillin_default_calckwargs.node_function(
        calc_kwargs={"properties": ["energy", "forces"]},
    )
    assert out["properties"] == ("energy", "forces")


def test_generate_kwargs_variant_is_a_deepcopy():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import (
        generate_kwargs_variant,
    )

    base = {"x": 0, "nested": {"a": 1}}
    out = generate_kwargs_variant.node_function(base, "x", 5)
    assert out == {"x": 5, "nested": {"a": 1}}
    out["nested"]["a"] = 99
    assert base["nested"]["a"] == 1


def test_generate_kwargs_variants_lists_them():
    from pyiron_workflow_atomistics._internal.kwargs_helpers import (
        generate_kwargs_variants,
    )

    out = generate_kwargs_variants.node_function(
        base_kwargs={"x": 0},
        key="x",
        values=[1, 2, 3],
    )
    assert out == [{"x": 1}, {"x": 2}, {"x": 3}]
