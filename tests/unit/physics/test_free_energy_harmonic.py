"""Tests for pyiron_workflow_atomistics.physics.free_energy.harmonic."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_harmonic_free_energy_emt_al_2x2x2(tmp_path):
    pytest.importorskip("phonopy", reason="phonopy not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
        harmonic_free_energy,
    )

    structure = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )

    out = harmonic_free_energy(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=(0.0, 300.0),
        working_directory=str(tmp_path),
        subdir="harmonic",
    )

    assert out.mode == "harmonic"
    assert out.reference_phase == "solid"
    # ZPE > 0 at T=0
    assert out.free_energy_array[0] > 0.0
    # F decreases with T (entropy dominates)
    assert out.free_energy_array[1] < out.free_energy_array[0]
    # Entropy at T=0 is zero
    assert out.entropy_array[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "expected_n_displacements"),
    [
        pytest.param(
            {"crystalstructure": "fcc", "a": 4.05, "cubic": True}, 1, id="fcc"
        ),
        pytest.param(
            {"crystalstructure": "hcp", "a": 2.86, "orthorhombic": True}, 4, id="hcp"
        ),
    ],
)
def test_fc2_generator_and_synthesiser_agree_on_displacements(
    kwargs, expected_n_displacements
):
    """The phono3py generator and the phonopy synthesiser must build the same cells.

    `_generate_fc2_supercells` displaces via `phono3py.generate_fc2_displacements`
    (``is_diagonal`` defaults to False); `_produce_fc2_view` re-derives the same
    dataset via `phonopy.generate_displacements` (``is_diagonal`` defaults to
    **True**). Left implicit, the two disagree whenever site symmetry is low
    enough for diagonal displacement directions to save work: an hcp
    orthorhombic cell generates 4 supercells and phonopy expects 2, so
    `_produce_fc2_view` raises "FC2 force/supercell mismatch".

    fcc is parametrised alongside because it gives 1 either way — it passes with
    or without the fix, which is exactly why the bug survived unnoticed.

    Asserting positions, not just counts: equal counts alone could be
    coincidental, and mismatched displacement vectors would silently fit wrong
    force constants rather than raising.
    """
    pytest.importorskip("phonopy", reason="phonopy not installed")
    pytest.importorskip("phono3py", reason="phono3py not installed")

    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import EngineOutput
    from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
        _produce_fc2_view,
    )
    from pyiron_workflow_atomistics.physics.phonons.harmonic import (
        _generate_fc2_supercells,
    )

    structure = bulk("Al", **kwargs)
    supercell_matrix = 2 * np.eye(3, dtype=int)

    generated = _generate_fc2_supercells(
        structure=structure,
        fc2_supercell_matrix=supercell_matrix,
        displacement_distance=0.03,
        is_plusminus="auto",
    )
    assert len(generated) == expected_n_displacements

    engine_outputs = []
    for cell in generated:
        evaluated = cell.copy()
        evaluated.calc = EMT()
        engine_outputs.append(
            EngineOutput(
                final_structure=cell,
                final_energy=float(evaluated.get_potential_energy()),
                converged=True,
                final_forces=evaluated.get_forces(),
            )
        )

    # Pre-fix this raises RuntimeError for the hcp case.
    view = _produce_fc2_view(
        structure=structure,
        fc2_supercell_matrix=supercell_matrix,
        fc2_engine_outputs=engine_outputs,
        displacement_distance=0.03,
        is_plusminus="auto",
    )

    expected_cells = view.supercells_with_displacements
    assert len(expected_cells) == len(generated)
    for generated_cell, expected_cell in zip(generated, expected_cells, strict=True):
        assert np.allclose(
            generated_cell.get_positions(),
            np.asarray(expected_cell.positions),
            atol=1e-10,
        )
        assert np.allclose(
            np.asarray(generated_cell.get_cell()),
            np.asarray(expected_cell.cell),
            atol=1e-10,
        )
