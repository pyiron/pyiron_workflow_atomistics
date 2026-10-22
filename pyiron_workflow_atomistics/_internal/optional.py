"""Guarded access to the optional ``[analysis]`` dependencies.

``structuretoolkit``, ``pyscal3`` and ``scikit-learn`` are *optional*
dependencies, installed with the ``analysis`` extra. They are needed only by
the CNA / Voronoi structure descriptors (``analysis.structure_descriptors``),
the melting-point workflows (``physics.melting``) and the void analysis in
``structure.defects``. Keeping them out of the mandatory requirements means
packages that only use, e.g., ``pyiron_workflow_atomistics.engine`` do not pull
them in.

Every module that needs one of them imports it through :func:`require` at the
point of use, so the package itself always imports and a missing install
produces an actionable message instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

ANALYSIS_EXTRA = "analysis"

# import name -> distribution name on PyPI / conda-forge
_DISTRIBUTIONS = {
    "pyscal3": "pyscal3",
    "sklearn": "scikit-learn",
    "structuretoolkit": "structuretoolkit",
}


def missing_message(module: str, feature: str, extra: str = ANALYSIS_EXTRA) -> str:
    """The error text for a missing optional dependency."""
    top_level = module.partition(".")[0]
    distribution = _DISTRIBUTIONS.get(top_level, top_level)
    return (
        f"{feature} needs the optional `{distribution}` package, which is not "
        f"installed.\n\n"
        f"    pip install 'pyiron_workflow_atomistics[{extra}]'\n\n"
        f"The `{extra}` extra installs structuretoolkit, pyscal3 and "
        f"scikit-learn; installing `{distribution}` directly (pip or "
        f"conda-forge) works too."
    )


def require(module: str, feature: str, extra: str = ANALYSIS_EXTRA) -> ModuleType:
    """Import ``module``, or raise an actionable error naming the extra.

    Only the absence of ``module``'s own top-level package is translated; a
    ``ModuleNotFoundError`` raised from *inside* an installed package (a
    broken install) propagates unchanged so it stays debuggable.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        top_level = module.partition(".")[0]
        if exc.name is not None and exc.name.partition(".")[0] != top_level:
            raise
        raise ModuleNotFoundError(
            missing_message(module, feature, extra), name=exc.name
        ) from exc
