"""pyiron_workflow_atomistics — atomistic-simulation workflows for pyiron.

Subpackages:
    engine    — the Engine Protocol, EngineOutput, calculate, and ASEEngine.
    structure — generic structure builders / transforms / defects.
    physics   — physics workflows (bulk, surface, point_defect, grain_boundary, phonons).
    analysis  — featurisation, post-processing, derived quantities.
    testing   — pytest mixins for engine-conformance verification.

Internal-only:
    _internal — kwargs plumbing, dataclass/dict mutators, workdir helpers.
"""

from . import _version

__version__ = _version.get_versions()["version"]
__all__ = ["__version__", "testing"]


def __getattr__(name):
    """Lazy submodule loading (PEP 562).

    `testing` is exposed as ``pyiron_workflow_atomistics.testing`` but is
    only loaded on first access, not at package import time. Without this,
    setuptools' ``[tool.setuptools.dynamic.version] attr = "...__version__"``
    triggers a full package import at build time and the cascading
    ``from . import testing`` → ``from ase import Atoms`` chain fails in
    build-isolation envs that don't yet have ase. PEP 562 lazy loading
    keeps the public attribute working while making ``import
    pyiron_workflow_atomistics`` cheap.
    """
    if name == "testing":
        import importlib

        module = importlib.import_module(".testing", __name__)
        globals()["testing"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
