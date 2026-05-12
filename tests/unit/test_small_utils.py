"""Tests for the small utility modules.

Covers:
* ``analysis.quantities.get_per_atom_quantity``
* ``physics._grain_boundary_helpers.geometry.axis_to_index``
* ``_internal.dataclass_helpers`` (modify_dataclass + variants)
* ``_internal.engine_output.extract_outputs_from_EngineOutputs``
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from ase.build import bulk

# --- analysis/quantities ----------------------------------------------------


def test_get_per_atom_quantity_divides_by_atom_count():
    from pyiron_workflow_atomistics.analysis.quantities import get_per_atom_quantity

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)  # 4 atoms
    assert len(structure) == 4
    per_atom = get_per_atom_quantity.node_function(quantity=-16.0, structure=structure)
    assert per_atom == -4.0


# --- _grain_boundary_helpers/geometry ---------------------------------------


@pytest.mark.parametrize(
    "axis, expected", [("a", 0), ("b", 1), ("c", 2), (0, 0), (5, 5)]
)
def test_axis_to_index_accepts_str_and_int(axis, expected):
    from pyiron_workflow_atomistics.physics._grain_boundary_helpers.geometry import (
        axis_to_index,
    )

    assert axis_to_index(axis) == expected


def test_axis_to_index_rejects_unknown_string():
    from pyiron_workflow_atomistics.physics._grain_boundary_helpers.geometry import (
        axis_to_index,
    )

    with pytest.raises(ValueError, match="Invalid axis string"):
        axis_to_index("z")


def test_axis_to_index_rejects_non_str_non_int():
    from pyiron_workflow_atomistics.physics._grain_boundary_helpers.geometry import (
        axis_to_index,
    )

    with pytest.raises(TypeError, match="Axis must be either"):
        axis_to_index(1.5)


# --- _internal/dataclass_helpers --------------------------------------------


@dataclass
class _Thing:
    a: int = 1
    b: str = "x"
    c: float = 0.0


def test_modify_dataclass_returns_a_new_instance_with_the_field_replaced():
    from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dataclass

    src = _Thing(a=1, b="x", c=0.0)
    out = modify_dataclass.node_function(src, "b", "y")
    assert isinstance(out, _Thing)
    assert out.b == "y"
    # Other fields untouched
    assert out.a == 1
    assert out.c == 0.0
    # Original instance not mutated
    assert src.b == "x"


def test_modify_dataclass_raises_on_unknown_field():
    from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dataclass

    with pytest.raises(KeyError, match="Unknown field"):
        modify_dataclass.node_function(_Thing(), "does_not_exist", 0)


def test_modify_dataclass_multi_replaces_each_field_in_turn():
    from pyiron_workflow_atomistics._internal.dataclass_helpers import (
        modify_dataclass_multi,
    )

    src = _Thing(a=1, b="x", c=0.0)
    out = modify_dataclass_multi.node_function(src, ["a", "c"], [42, 3.14])
    assert out.a == 42
    assert out.c == 3.14
    assert out.b == "x"  # untouched
    assert src.a == 1 and src.c == 0.0  # original unchanged


def test_modify_dataclass_multi_rejects_length_mismatch():
    from pyiron_workflow_atomistics._internal.dataclass_helpers import (
        modify_dataclass_multi,
    )

    with pytest.raises(ValueError, match="same length"):
        modify_dataclass_multi.node_function(_Thing(), ["a", "b"], [1])


def test_modify_dict_applies_updates_without_mutating_source():
    from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dict

    src = {"x": 1, "y": 2, "z": 3}
    out = modify_dict.node_function(src, {"x": 10, "z": 30})
    assert out == {"x": 10, "y": 2, "z": 30}
    # Source is untouched.
    assert src == {"x": 1, "y": 2, "z": 3}


def test_modify_dict_rejects_unknown_keys():
    from pyiron_workflow_atomistics._internal.dataclass_helpers import modify_dict

    with pytest.raises(KeyError, match="Unknown key"):
        modify_dict.node_function({"a": 1}, {"a": 2, "missing": 3})


# --- _internal/engine_output ------------------------------------------------


class _MockOutput:
    def __init__(self, energy, volume, converged=True):
        self.final_energy = energy
        self.final_volume = volume
        self.converged = converged


def test_extract_outputs_pulls_keys_from_each_output():
    from pyiron_workflow_atomistics._internal.engine_output import (
        extract_outputs_from_EngineOutputs,
    )

    outs = [_MockOutput(1.0, 10.0), _MockOutput(2.0, 20.0), _MockOutput(3.0, 30.0)]
    extracted = extract_outputs_from_EngineOutputs(
        engine_outputs=outs, keys=["final_energy", "final_volume"]
    )
    assert extracted == {
        "final_energy": [1.0, 2.0, 3.0],
        "final_volume": [10.0, 20.0, 30.0],
    }


def test_extract_outputs_filters_unconverged_by_default():
    from pyiron_workflow_atomistics._internal.engine_output import (
        extract_outputs_from_EngineOutputs,
    )

    outs = [
        _MockOutput(1.0, 10.0, converged=True),
        _MockOutput(2.0, 20.0, converged=False),  # filtered
        _MockOutput(3.0, 30.0, converged=True),
    ]
    extracted = extract_outputs_from_EngineOutputs(
        engine_outputs=outs, keys=["final_energy"]
    )
    assert extracted == {"final_energy": [1.0, 3.0]}


def test_extract_outputs_includes_unconverged_when_disabled():
    from pyiron_workflow_atomistics._internal.engine_output import (
        extract_outputs_from_EngineOutputs,
    )

    outs = [
        _MockOutput(1.0, 10.0, converged=True),
        _MockOutput(2.0, 20.0, converged=False),
    ]
    extracted = extract_outputs_from_EngineOutputs(
        engine_outputs=outs, keys=["final_energy"], only_converged=False
    )
    assert extracted == {"final_energy": [1.0, 2.0]}


def test_extract_outputs_missing_key_returns_none():
    from pyiron_workflow_atomistics._internal.engine_output import (
        extract_outputs_from_EngineOutputs,
    )

    outs = [_MockOutput(1.0, 10.0)]
    extracted = extract_outputs_from_EngineOutputs(
        engine_outputs=outs, keys=["final_energy", "field_that_does_not_exist"]
    )
    assert extracted == {
        "final_energy": [1.0],
        "field_that_does_not_exist": [None],
    }
