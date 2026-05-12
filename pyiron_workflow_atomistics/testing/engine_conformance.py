"""Pytest mixin verifying Engine Protocol conformance.

Subclass with::

    class TestMyEngineConformance(EngineConformanceTests):
        @staticmethod
        def engine_factory(tmp_path):
            return MyEngine(EngineInput=CalcInputStatic(),
                            working_directory=str(tmp_path))

The class needs no `__init__`; pytest discovers methods directly.
"""

from __future__ import annotations

from dataclasses import is_dataclass

from pyiron_workflow_atomistics.engine import Engine


class EngineConformanceTests:
    """Subclass and override ``engine_factory`` (required).

    Optionally override ``test_structure_factory`` to swap the default
    4-atom Cu FCC bulk used by the run() smoke test.
    """

    def test_satisfies_engine_protocol(self, tmp_path):
        """Engine instances must satisfy the runtime_checkable Protocol
        and be @dataclass-decorated so dataclasses.replace works in
        with_working_directory."""
        eng = type(self).engine_factory(tmp_path)
        assert isinstance(eng, Engine), (
            f"{type(eng).__name__} does not satisfy the Engine Protocol "
            "(missing working_directory, get_calculate_fn, or "
            "with_working_directory)"
        )
        assert is_dataclass(eng), (
            f"{type(eng).__name__} must be a @dataclass for the "
            "dataclasses.replace()-based with_working_directory pattern"
        )
