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


class EngineConformanceTests:
    """Subclass and override ``engine_factory`` (required).

    Optionally override ``test_structure_factory`` to swap the default
    4-atom Cu FCC bulk used by the run() smoke test.
    """
