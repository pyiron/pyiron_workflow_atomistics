"""Dedup helper for the GCO loop.

Port of the non-aggressive path of GRIP ``utils/unique.py:clear_best``,
operating on in-memory parallel lists of (row, atoms) instead of disk
filenames. The ``extra=True`` aggressive prune is intentionally not
ported in v1.
"""

from __future__ import annotations

from ase import Atoms


def dedup(
    rows: list[dict],
    atoms: list[Atoms],
) -> tuple[list[dict], list[Atoms]]:
    """Remove near-duplicate kept structures.

    Key
    ---
    ``(round(Egb, 3), round(n, 3))`` — two entries with the same rounded
    energy AND vacancy fraction are duplicates.

    Tie-break (lowest wins)
    -----------------------
    1. ``rx * ry`` (fewer atoms ⇒ less ambiguous reference)
    2. ``dx² + dy²`` (smaller in-plane shift)
    """
    if not rows:
        return [], []

    assert len(rows) == len(
        atoms
    ), f"rows ({len(rows)}) and atoms ({len(atoms)}) must align"

    # winner[key] = (rep_product, shift_sq, row_idx)
    winner: dict[tuple[float, float], tuple[int, float, int]] = {}

    for i, row in enumerate(rows):
        key = (round(row["Egb"], 3), round(row["n"], 3))
        rep_prod = row["rx"] * row["ry"]
        shift_sq = row["dx"] ** 2 + row["dy"] ** 2

        cur = winner.get(key)
        if cur is None:
            winner[key] = (rep_prod, shift_sq, i)
            continue
        cur_rep, cur_shift, _ = cur
        if rep_prod < cur_rep or (rep_prod == cur_rep and shift_sq < cur_shift):
            winner[key] = (rep_prod, shift_sq, i)

    kept_indices = sorted(v[2] for v in winner.values())
    return [rows[i] for i in kept_indices], [atoms[i] for i in kept_indices]
