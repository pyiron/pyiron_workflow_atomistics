"""Tests for `structure.defects`.

Tier 1 — pure helpers + the two as_function_node builders. Always runs.
Tier 2 — gated on ``pyscal3``; covers the void analysis pipeline.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk


# ---------------------------------------------------------------------------
# Tier 1 — create_vacancy
# ---------------------------------------------------------------------------


def test_create_vacancy_removes_one_atom():
    from pyiron_workflow_atomistics.structure.defects import create_vacancy

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    n0 = len(cu)
    vac = create_vacancy.node_function(cu, remove_atom_index=0)
    assert len(vac) == n0 - 1
    # input not mutated
    assert len(cu) == n0


def test_create_vacancy_removes_specified_index():
    from pyiron_workflow_atomistics.structure.defects import create_vacancy

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    target = cu.positions[5].copy()
    vac = create_vacancy.node_function(cu, remove_atom_index=5)
    # Position 5 from the original must no longer appear in the vacancy cell.
    assert not np.any(np.all(np.isclose(vac.positions, target), axis=1))


# ---------------------------------------------------------------------------
# Tier 1 — substitutional_swap
# ---------------------------------------------------------------------------


def test_substitutional_swap_changes_one_symbol():
    from pyiron_workflow_atomistics.structure.defects import substitutional_swap

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    out = substitutional_swap.node_function(cu, defect_site=0, new_symbol="Ni")
    syms = out.get_chemical_symbols()
    assert syms[0] == "Ni"
    assert all(s == "Cu" for s in syms[1:])
    # original unmutated
    assert cu.get_chemical_symbols()[0] == "Cu"


def test_substitutional_swap_with_explicit_index():
    from pyiron_workflow_atomistics.structure.defects import substitutional_swap

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    out = substitutional_swap.node_function(cu, defect_site=7, new_symbol="Fe")
    assert out.get_chemical_symbols()[7] == "Fe"
    assert out.get_chemical_symbols()[0] == "Cu"


# ---------------------------------------------------------------------------
# Tier 1 — filter_condition pure-Python branches
# ---------------------------------------------------------------------------


def _mock_sys(boxdims):
    return SimpleNamespace(boxdims=np.asarray(boxdims, dtype=float))


def test_filter_condition_inside_band_and_rvv():
    from pyiron_workflow_atomistics.structure.defects import filter_condition

    sys = _mock_sys([10.0, 10.0, 10.0])
    # pos lies inside [d_min, d_max] along axis=2, rvv inside (rvv_min, rvv_max).
    assert filter_condition(
        sys, pos=[0, 0, 4.0], rvv=0.5,
        distance_min=2.0, distance_max=6.0,
        axis=2, rvv_min=0.0, rvv_max=1.0,
    ) is True


def test_filter_condition_rvv_outside_returns_false():
    from pyiron_workflow_atomistics.structure.defects import filter_condition

    sys = _mock_sys([10.0, 10.0, 10.0])
    assert filter_condition(
        sys, pos=[0, 0, 4.0], rvv=5.0,
        distance_min=2.0, distance_max=6.0,
        axis=2, rvv_min=0.0, rvv_max=1.0,
    ) is False


def test_filter_condition_wrap_near_zero():
    """When pos[axis] is outside [d_min, d_max] but lies in [0, width] (the
    near-zero band wrap), the function should accept it."""
    from pyiron_workflow_atomistics.structure.defects import filter_condition

    sys = _mock_sys([10.0, 10.0, 10.0])
    # band centred at (2+6)/2 = 4 with width 2; wrap window = [0, 2].
    assert filter_condition(
        sys, pos=[0, 0, 1.0], rvv=0.5,
        distance_min=2.0, distance_max=6.0,
        axis=2, rvv_min=0.0, rvv_max=1.0,
    ) is True


def test_filter_condition_wrap_near_boxdim():
    """Pos in [boxdim - width, boxdim) is also accepted (top of the wrap)."""
    from pyiron_workflow_atomistics.structure.defects import filter_condition

    sys = _mock_sys([10.0, 10.0, 10.0])
    # band centred at 4 with width 2 → top wrap window = [8, 10).
    assert filter_condition(
        sys, pos=[0, 0, 9.0], rvv=0.5,
        distance_min=2.0, distance_max=6.0,
        axis=2, rvv_min=0.0, rvv_max=1.0,
    ) is True


def test_filter_condition_outside_all_bands_returns_false():
    from pyiron_workflow_atomistics.structure.defects import filter_condition

    sys = _mock_sys([10.0, 10.0, 10.0])
    # mid = 4, width = 2 → bands are [0,2], [2,6], [8,10). pos=7 lies outside.
    assert filter_condition(
        sys, pos=[0, 0, 7.0], rvv=0.5,
        distance_min=2.0, distance_max=6.0,
        axis=2, rvv_min=0.0, rvv_max=1.0,
    ) is False


# ---------------------------------------------------------------------------
# Tier 1 — tabulate_voids (matplotlib smoke; backend forced to Agg in conftest)
# ---------------------------------------------------------------------------


def test_tabulate_voids_runs_and_closes_figure():
    import matplotlib.pyplot as plt

    from pyiron_workflow_atomistics.structure.defects import tabulate_voids

    void_ratios = [0.2, 0.5, 1.0]
    void_count = [10, 20, 5]
    n_before = len(plt.get_fignums())
    tabulate_voids(void_ratios, void_count)
    # A new figure was created.
    assert len(plt.get_fignums()) == n_before + 1
    plt.close("all")


# ---------------------------------------------------------------------------
# Tier 2 — pyscal3-gated void analysis pipeline
# ---------------------------------------------------------------------------


def _fcc_cu_pyscal_system():
    """Build a fresh pyscal3 System for an 8x cubic-fcc Cu supercell."""
    pyscal3 = pytest.importorskip("pyscal3")
    System = pyscal3.System

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    sys = System()
    sys.read.file(cu, format="ase")
    return sys, cu


@pytest.mark.slow
def test_get_ra_matches_packing_factor_formula():
    """``get_ra`` returns (pf · V_atom / (4π/3))^(1/3); for fcc Cu with pf=0.74
    and 32 atoms in a 7.2³ box, the closed form is ≈ 1.272 Å."""
    from pyiron_workflow_atomistics.structure.defects import get_ra

    sys, _ = _fcc_cu_pyscal_system()
    ra = get_ra(sys, sys.natoms, pf=0.74)
    # Closed-form: volume = 7.2^3 = 373.248; v_atom = 11.664; ra = (0.74*11.664*3/(4π))**(1/3)
    expected = ((0.74 * (7.2**3 / 32)) / ((4 / 3) * np.pi)) ** (1 / 3)
    assert ra == pytest.approx(expected, rel=1e-9)


@pytest.mark.slow
def test_get_octahedral_positions_returns_finite_list():
    """For an 8x fcc Cu supercell the octahedral-site finder should produce
    a non-trivial number of candidate positions inside the box."""
    pytest.importorskip("pyscal3")
    from pyiron_workflow_atomistics.structure.defects import get_octahedral_positions

    sys, _ = _fcc_cu_pyscal_system()
    sys.find.neighbors(method="voronoi", cutoff=0.1)
    positions = get_octahedral_positions(sys, alat=3.6)
    assert len(positions) > 0
    box = sys.box
    for p in positions:
        assert 0 <= p[0] <= box[0][0]
        assert 0 <= p[1] <= box[1][1]
        assert 0 <= p[2] <= box[2][2]


@pytest.mark.slow
def test_calculate_voids_returns_sys_and_ratios(tmp_path, monkeypatch):
    """End-to-end: write a small Cu cell to a LAMMPS data file, run
    ``calculate_voids``, and confirm it returns a populated sys + ratios."""
    pytest.importorskip("pyscal3")
    from ase.io.lammpsdata import write_lammps_data

    from pyiron_workflow_atomistics.structure.defects import calculate_voids

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    monkeypatch.chdir(tmp_path)
    write_lammps_data("cu.data", cu)

    sys, void_ratios, void_count = calculate_voids(
        "cu.data",
        format="lammps-data",
        alat=3.6,
        pf=0.74,
        tabulate=False,
        write=False,
    )
    assert sys.natoms > 32  # original 32 + void sites
    assert len(void_ratios) > 0
    assert len(void_count) == len(void_ratios)
    # Sanity: tabulate=True path is covered separately to keep this test fast.


@pytest.mark.slow
def test_calculate_voids_with_tabulate_runs(tmp_path, monkeypatch):
    """Covers the ``tabulate=True`` branch (calls ``tabulate_voids`` which
    needs the matplotlib backend already forced to Agg in conftest)."""
    pytest.importorskip("pyscal3")
    import matplotlib.pyplot as plt
    from ase.io.lammpsdata import write_lammps_data

    from pyiron_workflow_atomistics.structure.defects import calculate_voids

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    monkeypatch.chdir(tmp_path)
    write_lammps_data("cu.data", cu)
    plt.close("all")
    calculate_voids(
        "cu.data",
        format="lammps-data",
        alat=3.6,
        pf=0.74,
        tabulate=True,
        write=False,
    )
    plt.close("all")


@pytest.mark.slow
def test_filter_and_cluster_atoms_runs_after_calculate_voids(tmp_path, monkeypatch):
    pytest.importorskip("pyscal3")
    from ase.io.lammpsdata import write_lammps_data

    from pyiron_workflow_atomistics.structure.defects import (
        calculate_voids,
        filter_and_cluster_atoms,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    monkeypatch.chdir(tmp_path)
    write_lammps_data("cu.data", cu)

    sys, _vr, _vc = calculate_voids(
        "cu.data",
        format="lammps-data",
        alat=3.6,
        pf=0.74,
        tabulate=False,
        write=False,
    )
    out = filter_and_cluster_atoms(
        sys, distance=[0.0, 14.4], axis=2, rvv=[0.0, 1.5], write=False
    )
    # Output is a fresh System; it should still hold a positive atom count.
    assert out.natoms > 0


@pytest.mark.slow
def test_filter_and_cluster_atoms_writes_output_when_requested(tmp_path, monkeypatch):
    pytest.importorskip("pyscal3")
    from ase.io.lammpsdata import write_lammps_data

    from pyiron_workflow_atomistics.structure.defects import (
        calculate_voids,
        filter_and_cluster_atoms,
    )

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    monkeypatch.chdir(tmp_path)
    write_lammps_data("cu.data", cu)

    sys, _, _ = calculate_voids(
        "cu.data",
        format="lammps-data",
        alat=3.6,
        pf=0.74,
        tabulate=False,
        write=False,
    )
    filter_and_cluster_atoms(
        sys, distance=[0.0, 14.4], axis=2, rvv=[0.0, 1.5], write=True
    )
    assert os.path.exists(tmp_path / "output.data")
