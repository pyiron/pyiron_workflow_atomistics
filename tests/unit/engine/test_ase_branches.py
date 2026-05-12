"""Targeted branch tests for ``engine/ase.py``.

The end-to-end happy paths are already covered by ``test_ase.py``,
``test_ase_md.py``, ``test_pure_gb_study.py``, and ``test_point_defect.py``.
This file fills in:

* unsupported EngineInput dispatch (TypeError fallback)
* ``optimizer_class=None`` static path inside ``ase_calc_structure``
* ``relax_cell=True`` variable-cell relaxation path
* ``write_to_disk=True`` artifact emission for minimization
* unsupported-property request inside ``_gather``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


# --- dispatch fallback ------------------------------------------------------


@dataclass
class _BogusInput:
    """Not a CalcInputStatic / Minimize / MD — must trigger the dispatch fallback."""
    something: int = 1


def test_get_calculate_fn_raises_for_unknown_engine_input(tmp_path):
    from pyiron_workflow_atomistics.engine import ASEEngine

    engine = ASEEngine(
        EngineInput=_BogusInput(),  # type: ignore[arg-type]
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    with pytest.raises(TypeError, match="Unsupported EngineInput type"):
        engine.get_calculate_fn(structure=bulk("Cu", "fcc", a=3.6, cubic=True))


# --- ase_calc_structure static path (optimizer_class=None) ------------------


def test_ase_calc_structure_static_path_records_one_frame(tmp_path):
    """optimizer_class=None bypasses the BFGS path and emits a single
    structure+results pair (the static evaluation)."""
    from pyiron_workflow_atomistics.engine.ase import ase_calc_structure

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = ase_calc_structure(
        structure=structure,
        calc=EMT(),
        optimizer_class=None,
        optimizer_kwargs={},
        record_interval=1,
        fmax=0.0,
        max_steps=0,
        relax_cell=False,
        properties=("energy", "forces", "volume"),
        write_to_disk=False,
        working_directory=str(tmp_path),
    )
    assert out.converged is True
    # Single-frame trajectory: energies list has exactly one entry.
    assert out.energies is not None and len(out.energies) == 1


# --- relax_cell=True path ---------------------------------------------------


def test_ase_engine_relax_cell_uses_ExpCellFilter(tmp_path):
    """CalcInputMinimize(relax_cell=True) exercises the ExpCellFilter branch."""
    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputMinimize,
        run,
    )

    structure = bulk("Cu", "fcc", a=3.5, cubic=True)  # off-equilibrium
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1,
            max_iterations=20,
            relax_cell=True,
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = run.node_function(structure=structure, engine=engine)
    assert out.converged is True
    # Cell should have moved toward EMT equilibrium (~3.6 Å for Cu).
    new_a = out.final_structure.cell[0, 0]
    assert new_a > 3.5, f"expected cell expansion; got {new_a}"


# --- write_to_disk emission paths ------------------------------------------


def test_ase_calc_structure_minimize_with_write_to_disk_emits_artifacts(tmp_path):
    """write_to_disk=True must produce initial/final/trajectory files."""
    from pyiron_workflow_atomistics.engine import (
        ASEEngine,
        CalcInputMinimize,
        run,
    )

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    structure.rattle(0.05, seed=0)
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(
            force_convergence_tolerance=0.1, max_iterations=20
        ),
        calculator=EMT(),
        working_directory=str(tmp_path),
        write_to_disk=True,
    )
    out = run.node_function(structure=structure, engine=engine)
    assert out.converged is True
    for name in (
        "initial_structure.xyz",
        "initial_results.json",
        "trajectory.xyz",
        "trajectory_results.json",
        "final_structure.xyz",
        "final_results.json",
    ):
        assert (tmp_path / name).exists(), f"missing artifact: {name}"


# --- _gather unsupported property ------------------------------------------


def test__gather_raises_on_unsupported_property(tmp_path):
    """Asking for a property the calculator cannot deliver must KeyError."""
    from pyiron_workflow_atomistics.engine.ase import _gather

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    structure.calc = EMT()
    # EMT doesn't compute "dipole"; _gather must raise rather than silently drop.
    with pytest.raises(KeyError, match="Requested properties not available"):
        _gather(structure, ("energy", "dipole"))
