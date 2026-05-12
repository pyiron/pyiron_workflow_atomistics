"""pyiron_workflow_atomistics — atomistic-simulation workflows for pyiron.

Subpackages:
    engine    — the Engine Protocol, EngineOutput, run, and ASEEngine.
    structure — generic structure builders / transforms / defects.
    physics   — physics workflows (bulk, surface, point_defect, grain_boundary).
    analysis  — featurisation, post-processing, derived quantities.

Internal-only:
    _internal — kwargs plumbing, dataclass/dict mutators, workdir helpers.
"""

from . import _version

__version__ = _version.get_versions()["version"]
__all__ = ["__version__"]

from . import testing  # noqa: F401  -- exposes pyiron_workflow_atomistics.testing
