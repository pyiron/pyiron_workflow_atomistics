"""Engine Protocol, EngineOutput dataclass, and the single run() entry point.

The Engine Protocol defines the contract every compute engine (ASE, VASP,
LAMMPS, ...) must satisfy so physics workflows can use them interchangeably.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms


@runtime_checkable
class Engine(Protocol):
    """An engine computes properties of a structure.

    Implementations live alongside their backends:
    :class:`pyiron_workflow_atomistics.engine.ase.ASEEngine` here, future
    ``VaspEngine`` / ``LammpsEngine`` in their own packages.

    Contract
    --------
    Engines MUST be pickleable so workflows can be checkpointed or submitted
    to SLURM. Engines MUST implement ``with_working_directory`` purely (no
    mutation of self). These properties are documented but not enforced via
    ``__reduce__`` — relying on duck typing keeps the contract simple.

    Attributes
    ----------
    working_directory
        Root directory the engine writes calc artefacts into. Sub-workflows
        compose paths by calling :meth:`with_working_directory`.
    """

    working_directory: str

    def get_calculate_fn(
        self, structure: Atoms
    ) -> tuple[Callable[..., "EngineOutput"], dict[str, Any]]:
        """Return ``(callable, kwargs)``. The callable will be invoked as
        ``callable(structure=structure, **kwargs)`` and must return an
        :class:`EngineOutput`."""

    def with_working_directory(self, subdir: str) -> "Engine":
        """Return a *copy* of this engine whose ``working_directory`` is
        ``os.path.join(self.working_directory, subdir)``.

        Pure — never mutates ``self``. Replaces the historical
        ``duplicate_engine`` helper.
        """


@dataclass
class EngineOutput:
    """Structured result of a single engine evaluation.

    Required
    --------
    final_structure
        The final atomic structure (post-relaxation or last MD step).
    final_energy
        Total potential energy in eV.
    converged
        True if the engine reports the calculation converged.

    Optional per-property
    ---------------------
    Single trailing values; engines fill what they compute.

    Trajectory (relax / MD only; ``None`` for static)
    -------------------------------------------------
    Lists indexed by ionic step.

    Examples
    --------
    >>> from ase.build import bulk
    >>> out = EngineOutput(
    ...     final_structure=bulk("Cu", "fcc", a=3.6, cubic=True),
    ...     final_energy=-3.5,
    ...     converged=True,
    ... )
    >>> out.to_dict()["final_energy"]
    -3.5
    """

    final_structure: Atoms
    final_energy: float
    converged: bool

    final_forces: np.ndarray | None = None
    final_stress: np.ndarray | None = None              # (3, 3)
    final_stress_voigt: np.ndarray | None = None        # (6,)
    final_volume: float | None = None
    final_magmoms: np.ndarray | None = None

    energies: list[float] | None = None
    forces: list[np.ndarray] | None = None
    stresses: list[np.ndarray] | None = None
    structures: list[Atoms] | None = None
    n_ionic_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of the dataclass fields (ASE objects preserved by reference)."""
        return asdict(self)


@pwf.as_function_node("subengine")
def subengine(engine: Engine, subdir: str) -> Engine:
    """Function-node wrapper around :meth:`Engine.with_working_directory`.

    Use this inside ``@pwf.as_macro_node`` graphs where the engine arrives
    as an input channel. Calling ``engine.with_working_directory(...)``
    directly in a macro body dispatches the channel's ``__call__`` and
    crashes with ``ReadinessError``; routing the same call through this
    node delays the resolution to graph-execution time.
    """
    subengine = engine.with_working_directory(subdir)
    return subengine


@pwf.as_function_node("path")
def subdir_path(engine: Engine, subdir: str) -> str:
    """Function-node returning ``os.path.join(engine.working_directory, subdir)``.

    Companion to :func:`subengine` for sites that need the path string
    rather than a new engine inside a macro graph.
    """
    import os as _os

    path = _os.path.join(engine.working_directory, subdir)
    return path


@pwf.as_function_node("engine_output")
def run(structure: Atoms, engine: Engine) -> EngineOutput:
    """Execute ``engine`` on ``structure``.

    The one node every physics workflow uses to compute things.

    Examples
    --------
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize, run
    >>> engine = ASEEngine(
    ...     EngineInput=CalcInputMinimize(force_convergence_tolerance=0.05),
    ...     calculator=EMT(),
    ...     working_directory="./_demo",
    ... )
    >>> out = run.node_function(bulk("Cu", "fcc", a=3.6, cubic=True), engine)  # doctest: +SKIP
    """
    fn, kwargs = engine.get_calculate_fn(structure)
    return fn(structure=structure, **kwargs)
