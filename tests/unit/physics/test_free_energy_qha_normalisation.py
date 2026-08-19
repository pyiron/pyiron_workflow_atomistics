"""Regression tests for the per-atom normalisation of the QHA inputs.

Everything handed to ``phonopy.qha.QHA`` must share one basis. Three arrays
arrive from three places and all three used to disagree:

* ``volumes``              — ``_static_energies_per_volume``, per atom
* ``electronic_energies``  — ``_static_energies_per_volume``, was per CELL
* ``fe_phonon``/``cv``/``entropy`` — ``_harmonic_grid_over_volumes``, was per
  PRIMITIVE CELL

The static-energy mismatch inflates the fitted bulk modulus by the unit-cell
atom count (B = V d²E/dV²). The phonon mismatch mis-weights the vibrational
term against the static term by ``n_atoms_primitive`` — invisible for elemental
cubic cells, a factor of 2 for hcp, and *different between phases*.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.bulk import generate_structures
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
    _harmonic_grid_over_volumes,
    _static_energies_per_volume,
)

EV_PER_ANG3_TO_GPA = 160.21766208
EV_TO_KJ_MOL = 96.48533212331002  # c.eV * c.Avogadro / 1000


def _emt_engine(tmp_path):
    return ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )


def _strained(base, num_points=5):
    return generate_structures(
        base_structure=base,
        axes=["iso"],
        strain_range=(-0.03, 0.03),
        num_points=num_points,
    )


def _bulk_modulus_GPa(energies, volumes):
    _, _, B = EquationOfState(
        np.asarray(volumes).tolist(),
        np.asarray(energies).tolist(),
        eos="birchmurnaghan",
    ).fit()
    return B * EV_PER_ANG3_TO_GPA


def test_static_energies_are_per_atom(tmp_path):
    """Returned energies must be per atom, matching the per-atom volumes.

    Pre-patch this fails by a factor of 4 for the 4-atom conventional cell.
    """
    base = bulk("Al", "fcc", a=4.05, cubic=True)
    structures = _strained(base)

    energies, volumes = _static_energies_per_volume(
        strained_structures=structures, engine=_emt_engine(tmp_path)
    )

    expected_energies = []
    expected_volumes = []
    for s in structures:
        ref = s.copy()
        ref.calc = EMT()
        expected_energies.append(ref.get_potential_energy() / len(s))
        expected_volumes.append(s.get_volume() / len(s))

    assert np.allclose(energies, expected_energies, rtol=1e-10)
    assert np.allclose(volumes, expected_volumes, rtol=1e-10)


def test_bulk_modulus_is_independent_of_unit_cell_size(tmp_path):
    """The same crystal in a bigger unit cell must give the same bulk modulus.

    This is the physically meaningful statement of the bug: nothing observable
    may depend on how many atoms the caller chose to put in the cell. With the
    per-cell/per-atom mismatch the two cells below disagree by exactly 2x
    (B scales as the atom count: 4 vs 8).
    """
    engine = _emt_engine(tmp_path)

    base_4 = bulk("Al", "fcc", a=4.05, cubic=True)
    base_8 = bulk("Al", "fcc", a=4.05, cubic=True).repeat((1, 1, 2))
    assert len(base_4) == 4 and len(base_8) == 8

    B = []
    for base in (base_4, base_8):
        energies, volumes = _static_energies_per_volume(
            strained_structures=_strained(base), engine=engine
        )
        B.append(_bulk_modulus_GPa(energies, volumes))

    assert B[0] == pytest.approx(B[1], rel=1e-6)
    # Sanity: EMT Al is a ~40 GPa material, not a ~160 GPa one. Guards against
    # both cells being wrong by the same factor, which the ratio check alone
    # would not catch.
    assert 25.0 < B[0] < 60.0


class _StubEngine:
    """Minimal Engine stand-in: only ``with_working_directory`` is exercised."""

    def with_working_directory(self, subdir):
        return self


def _stub_harmonic_output(n_atoms_primitive: int, temperatures) -> FreeEnergyOutput:
    """A FreeEnergyOutput carrying known per-atom arrays, per the harmonic spec."""
    T = np.asarray(temperatures, dtype=float)
    return FreeEnergyOutput(
        mode="harmonic",
        reference_phase="solid",
        free_energy=0.01,
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=0.0,
        n_atoms=4,
        elements=["Al"],
        simfolder="",
        report={"n_atoms_primitive": int(n_atoms_primitive)},
        temperature_array=T,
        free_energy_array=np.linspace(0.04, -0.09, T.size),
        entropy_array=np.linspace(0.0, 3.0e-4, T.size),
        heat_capacity_array=np.linspace(0.0, 2.5e-4, T.size),
    )


@pytest.mark.parametrize("n_atoms_primitive", [1, 2, 4])
def test_harmonic_grid_keeps_the_per_atom_basis(monkeypatch, n_atoms_primitive):
    """F/S/Cv must be eV→kJ/mol only, never rescaled by ``n_atoms_primitive``.

    ``harmonic_free_energy`` reports per primitive-cell ATOM. The static
    energies and volumes QHA is fed alongside are per atom too, so the phonon
    arrays must not be lifted to a per-primitive-CELL basis. Pre-patch this
    fails for every ``n_atoms_primitive > 1`` — i.e. silently for hcp, which is
    exactly where a cross-phase comparison goes wrong.
    """
    import pyiron_workflow_atomistics.physics.free_energy.quasiharmonic as qha_mod

    temperatures = (0.0, 300.0, 600.0)
    stub = _stub_harmonic_output(n_atoms_primitive, temperatures)
    monkeypatch.setattr(qha_mod, "harmonic_free_energy", lambda **kwargs: stub)

    structures = [bulk("Al", "fcc", a=a, cubic=True) for a in (4.00, 4.05, 4.10)]
    F_TV, S_TV, Cv_TV = _harmonic_grid_over_volumes(
        strained_structures=structures,
        engine=_StubEngine(),
        fc2_supercell_matrix=2 * np.eye(3, dtype=int),
        temperatures=temperatures,
        displacement_distance=0.03,
        is_plusminus="auto",
        working_directory=".",
    )

    n_V = len(structures)
    for got, per_atom, unit in (
        (F_TV, stub.free_energy_array, EV_TO_KJ_MOL),
        (S_TV, stub.entropy_array, EV_TO_KJ_MOL * 1000.0),
        (Cv_TV, stub.heat_capacity_array, EV_TO_KJ_MOL * 1000.0),
    ):
        expected = np.repeat((np.asarray(per_atom) * unit)[:, None], n_V, axis=1)
        assert got.shape == expected.shape
        assert np.allclose(got, expected, rtol=1e-12), (
            "phonon arrays were rescaled off the per-atom basis "
            f"(n_atoms_primitive={n_atoms_primitive})"
        )


def test_harmonic_grid_is_invariant_to_primitive_cell_detection():
    """Two cells differing only in ``n_atoms_primitive`` must give the same F/S/Cv.

    ``n_atoms_primitive`` is decided at run time by spglib, and it varies with
    cell choice and with numerical noise in the positions. Nothing QHA consumes
    may depend on it, or the same crystal yields different thermodynamics
    depending on how it happened to be described.
    """
    import pyiron_workflow_atomistics.physics.free_energy.quasiharmonic as qha_mod

    temperatures = (0.0, 300.0, 600.0)
    structures = [bulk("Al", "fcc", a=4.05, cubic=True)]
    grids = []
    original = qha_mod.harmonic_free_energy
    try:
        for n_prim in (1, 2):
            stub = _stub_harmonic_output(n_prim, temperatures)

            def _stubbed(_stub=stub, **kwargs):
                return _stub

            qha_mod.harmonic_free_energy = _stubbed
            grids.append(
                _harmonic_grid_over_volumes(
                    strained_structures=structures,
                    engine=_StubEngine(),
                    fc2_supercell_matrix=2 * np.eye(3, dtype=int),
                    temperatures=temperatures,
                    displacement_distance=0.03,
                    is_plusminus="auto",
                    working_directory=".",
                )
            )
    finally:
        qha_mod.harmonic_free_energy = original

    for arr_1, arr_2 in zip(*grids, strict=True):
        assert np.allclose(arr_1, arr_2, rtol=1e-12)
