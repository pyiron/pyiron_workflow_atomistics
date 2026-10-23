"""Physics-level input dataclasses for calphy free-energy workflows.

Only the potential lives here. The structure is `ase.Atoms` (no dataclass
needed), and the LAMMPS launcher comes from ``LammpsEngine.command``,
parsed by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LammpsPotential:
    """LAMMPS interatomic potential, passed verbatim to calphy.

    Attributes
    ----------
    pair_style
        LAMMPS ``pair_style`` line, e.g. ``"eam/alloy"``, ``"pace"``,
        ``"grace"``.
    pair_coeff
        LAMMPS ``pair_coeff`` line, e.g.
        ``"* * /path/to/Cu01.eam.alloy Cu"``. Element ordering must
        match the structure's chemical-symbol first-occurrence order
        (for ``composition_scaling``, followed by any target elements
        the structure lacks, in the order of
        ``output_chemical_composition``). A relative potential file is
        resolved against the current directory.
    potential_file
        Not supported: calphy>=2 deprecates and ignores
        ``Calculation.potential_file``, so the calphy nodes raise a
        ``ValueError`` when it is set. Put everything the potential
        needs into ``pair_style`` / ``pair_coeff``.
    """

    pair_style: str
    pair_coeff: str
    potential_file: str | None = None
