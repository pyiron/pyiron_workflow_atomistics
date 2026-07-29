import flowrep as fr
import numpy as np
from ase import Atoms


@fr.atomic
def add_vacuum(atoms, vacuum_length=20, axis="c", center_atoms=True):
    """
    Add vacuum padding to an ASE Atoms object along a specified axis.

    Parameters:
    atoms : ase.Atoms
        The ASE Atoms object to which vacuum will be added.
    vacuum_length : float, optional
        Thickness of vacuum to add (in Angstroms). Default is 20.
    axis : {'a', 'b', 'c'} or int, optional
        Axis along which to add vacuum. Can specify as 'a', 'b', 'c' or 0,1,2. Default is 'c'.
    center_atoms : bool, optional
        Whether to center the atoms in the simulation cell after adding vacuum. Default is True.

    Returns:
    ase.Atoms
        A new ASE Atoms object with added vacuum along the specified axis.
    """
    # Copy atoms to avoid modifying original
    new_atoms = atoms.copy()

    # Map axis letter to index
    axis_map = {"a": 0, "b": 1, "c": 2}
    if isinstance(axis, str):
        axis_lower = axis.lower()
        if axis_lower not in axis_map:
            raise ValueError(f"Invalid axis '{axis}'. Choose from 'a', 'b', 'c'.")
        axis_idx = axis_map[axis_lower]
    elif isinstance(axis, int) and axis in (0, 1, 2):
        axis_idx = axis
    else:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'a', 'b', 'c' or 0,1,2.")

    # Use ASE's add_vacuum
    # ase_add_vacuum(new_atoms, vacuum_length, axis=axis_idx)
    new_atoms.center(vacuum=vacuum_length / 2, axis=axis_idx)
    return new_atoms


@fr.atomic("supercell")
def create_supercell(base_structure: Atoms, supercell_repeats: tuple) -> Atoms:
    # Create the supercell
    supercell = base_structure.repeat(supercell_repeats)
    return supercell


@fr.atomic("supercell")
def create_supercell_with_min_dimensions(
    base_structure: Atoms, min_dimensions=None
) -> Atoms:
    """
    Expand a base ASE structure into a supercell so that each cell vector
    length meets or exceeds the specified minimum dimensions.

    Parameters
    ----------
    base_structure : ase.Atoms
        The starting unit or supercell.
    min_dimensions : list of length 3 (floats or None)
        Minimum lengths along the [a, b, c] cell vectors in Å.
        Use None to disable a dimension constraint.

    Returns
    -------
    ase.Atoms
        A new Atoms object repeated along each lattice vector
        so that its cell lengths are >= the given minima.
    """
    # Get current cell vectors and their lengths
    if min_dimensions is None:
        min_dimensions = [6, 6, None]
    cell = base_structure.get_cell()
    lengths = np.linalg.norm(cell, axis=1)

    # Determine repeat factors for each axis
    repeats = []
    for length, min_len in zip(lengths, min_dimensions, strict=False):
        if min_len is None:
            repeats.append(1)
        else:
            # At least one repetition
            factor = int(np.ceil(min_len / length))
            repeats.append(max(factor, 1))

    # Create the supercell
    supercell = base_structure.repeat(tuple(repeats))
    return supercell


@fr.atomic("rattled_structure")
def rattle(structure: Atoms, rattle: float | None = None) -> Atoms:
    """Return a copy of ``structure`` with atomic positions perturbed.

    Parameters
    ----------
    structure : ase.Atoms
        Input structure.
    rattle : float, optional
        Standard deviation (Å) of the random displacement applied via
        :meth:`ase.Atoms.rattle`. If ``None`` or ``0``, no perturbation
        is applied (a plain copy is returned).
    """
    rattled_structure = structure.copy()
    if rattle:
        rattled_structure.rattle(rattle)
    return rattled_structure
