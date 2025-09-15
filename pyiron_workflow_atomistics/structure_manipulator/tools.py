import numpy as np
import pyiron_workflow as pwf
from ase import Atoms


@pwf.as_function_node
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


@pwf.as_function_node("supercell")
def create_supercell_with_min_dimensions(
    base_structure: Atoms, min_dimensions=[6, 6, None]
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
    cell = base_structure.get_cell()
    lengths = np.linalg.norm(cell, axis=1)

    # Determine repeat factors for each axis
    repeats = []
    for length, min_len in zip(lengths, min_dimensions):
        if min_len is None:
            repeats.append(1)
        else:
            # At least one repetition
            factor = int(np.ceil(min_len / length))
            repeats.append(max(factor, 1))

    # Create the supercell
    supercell = base_structure.repeat(tuple(repeats))
    return supercell


@pwf.as_function_node("structure")
def substitutional_swap_one_site(
    base_structure: Atoms, defect_site: int = 0, new_symbol: str = "Si"
) -> Atoms:
    # build the supercell
    structure = base_structure.copy()
    # swap only the one atom in the original (0,0,0) block
    structure[defect_site].symbol = new_symbol
    return structure


# Because it is really fucking annoying to have to access the data from the dataframe when all I want is a list.
@pwf.as_function_node
def forloop_function(function, kwarg_to_iterate, kwarg_values, other_kwargs=None):
    """
    Applies `function` repeatedly changing a single keyword argument over given values,
    merged with any fixed `other_kwargs`.

    :param function:          callable to invoke
    :param kwarg_to_iterate:  str               # name of the keyword argument to iterate
    :param kwarg_values:      Iterable         # values to assign to that keyword
    :param other_kwargs:      dict, optional   # any additional fixed keywords
    :return:                  list             # outputs from each call
    # Example usage:
    # def compute(a, b, scale=1):
    #     return scale * (a + b)
    #
    # results = forloop_function(
    #     function=compute,
    #     kwarg_to_iterate='a',
    #     kwarg_values=[1,2,3],
    #     other_kwargs={'b':10, 'scale':0.5}
    # )
    # print(results)  # [5.5, 6.0, 6.5]
    """
    output_lst = []
    other_kwargs = other_kwargs or {}

    # Iterate over all provided values for the single kwarg
    for val in kwarg_values:
        # Build parameters for this call
        params = {kwarg_to_iterate: val}
        # Merge in fixed kwargs
        params.update(other_kwargs)
        # Invoke the function
        result = function(**params)
        output_lst.append(result)

    return output_lst
