"""Public test helpers for downstream Engine implementations.

The :class:`EngineConformanceTests` mixin is the single source of truth
for the :class:`pyiron_workflow_atomistics.engine.Engine` Protocol
contract. Downstream packages subclass it with their own engine factory
and run pytest; every contract clause is exercised by a named method.
"""

from .engine_conformance import EngineConformanceTests

__all__ = ["EngineConformanceTests"]
