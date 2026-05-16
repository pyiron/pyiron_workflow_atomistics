"""Unit tests for the dedup helper."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from pyiron_workflow_atomistics.physics._grand_canonical_gb_code.store import dedup


def _row(Egb, n, dx=0.0, dy=0.0, rx=1, ry=1):
    return {
        "Egb": Egb,
        "n": n,
        "dx": dx,
        "dy": dy,
        "rx": rx,
        "ry": ry,
        "T": 0,
        "n_md_steps": 0,
        "iter": 0,
        "converged": True,
    }


def _atoms(symbol="H"):
    return Atoms(symbol, positions=[[0, 0, 0]], cell=[1, 1, 1])


def test_no_duplicates_returns_input_unchanged():
    rows = [_row(0.5, 0.1), _row(0.4, 0.2)]
    atoms = [_atoms(), _atoms()]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 2
    assert len(out_atoms) == 2


def test_same_Egb_and_n_smaller_rep_wins():
    rows = [
        _row(0.5, 0.1, rx=2, ry=3),  # rx*ry=6
        _row(0.5, 0.1, rx=1, ry=2),  # rx*ry=2 — should win
    ]
    atoms = [_atoms("H"), _atoms("He")]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 1
    assert out_rows[0]["rx"] * out_rows[0]["ry"] == 2
    assert out_atoms[0].get_chemical_symbols() == ["He"]


def test_same_rep_smaller_shift_wins():
    rows = [
        _row(0.5, 0.1, rx=1, ry=1, dx=3.0, dy=4.0),  # |d|² = 25
        _row(0.5, 0.1, rx=1, ry=1, dx=0.5, dy=0.5),  # |d|² = 0.5 — should win
    ]
    atoms = [_atoms("H"), _atoms("Li")]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 1
    assert out_atoms[0].get_chemical_symbols() == ["Li"]


def test_different_n_kept_separately():
    rows = [_row(0.5, 0.1, rx=1, ry=1), _row(0.5, 0.2, rx=1, ry=1)]
    atoms = [_atoms(), _atoms()]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 2


def test_different_Egb_kept_separately():
    rows = [_row(0.5, 0.1), _row(0.6, 0.1)]
    atoms = [_atoms(), _atoms()]
    out_rows, out_atoms = dedup(rows, atoms)
    assert len(out_rows) == 2


def test_empty_input_returns_empty():
    out_rows, out_atoms = dedup([], [])
    assert out_rows == []
    assert out_atoms == []
