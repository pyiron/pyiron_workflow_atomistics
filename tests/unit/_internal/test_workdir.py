"""Tests for the workdir fan-out helpers."""

from __future__ import annotations

import os


def test_get_subdirpaths_joins_parent_and_subdirs():
    from pyiron_workflow_atomistics._internal.workdir import get_subdirpaths

    out = get_subdirpaths.node_function(
        parent_dir="/tmp/parent",
        output_subdirs=["a", "b", "c"],
    )
    assert out == [
        os.path.join("/tmp/parent", "a"),
        os.path.join("/tmp/parent", "b"),
        os.path.join("/tmp/parent", "c"),
    ]


def test_get_subdirpaths_empty_subdirs():
    from pyiron_workflow_atomistics._internal.workdir import get_subdirpaths

    assert get_subdirpaths.node_function(parent_dir="/tmp", output_subdirs=[]) == []


def test_get_working_subdir_kwargs_overrides_working_directory():
    from pyiron_workflow_atomistics._internal.workdir import get_working_subdir_kwargs

    original = {"working_directory": "/old", "force_tol": 0.01}
    out = get_working_subdir_kwargs.node_function(
        calc_structure_fn_kwargs=original,
        base_working_directory="/base",
        new_working_directory="sub",
    )
    assert out["working_directory"] == os.path.join("/base", "sub")
    assert out["force_tol"] == 0.01
    # Original dict is not mutated.
    assert original["working_directory"] == "/old"


def test_get_working_subdir_kwargs_requires_working_directory_present():
    """`modify_dict` rejects keys that are not already present, so the
    caller must seed `working_directory` even if it will be overridden."""
    import pytest

    from pyiron_workflow_atomistics._internal.workdir import get_working_subdir_kwargs

    with pytest.raises(KeyError, match="working_directory"):
        get_working_subdir_kwargs.node_function(
            calc_structure_fn_kwargs={},
            base_working_directory="/base",
            new_working_directory="run1",
        )
