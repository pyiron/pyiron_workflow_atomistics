import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import pyiron_workflow_atomistics.gb_generator as gbc
import tempfile
import sys
import glob

def get_pmg_struct_from_gbcode(axis, basis, lattice_param, m, n, GB1, element, req_length_grain=15, grain_length_axis=0):
    """
    Generates a grain boundary (GB) structure and extends it to the specified minimum length.
    Args:
        axis, basis, lattice_param, m, n, GB1, element: GB parameters.
        req_length_grain: Minimum required grain length.
        grain_length_axis: Axis along which to extend the GB.
    Returns:
        A pymatgen Structure object with the specified GB.
    """
    my_gb = gbc.GB_character()
    my_gb.ParseGB(axis, basis, lattice_param, m, n, GB1)
    my_gb.CSL_Bicrystal_Atom_generator()

    # Generate initial structure and extend to minimum length
    structure = _write_and_load_structure(my_gb)
    extend_factors = get_multiplier_to_extend_gb_to_min_length(structure, axis=grain_length_axis, req_length_grain=req_length_grain)
    
    # Extend GB structure
    structure = _write_and_load_structure(my_gb, extend_by=extend_factors[grain_length_axis])

    # Map all atoms to the specified element
    element_mapping = {el: element for el in structure.species}
    structure.replace_species(element_mapping)
    return structure

def _write_and_load_structure(my_gb, extend_by=1):
    """
    Writes the GB to a temporary file, loads it as a pymatgen Structure, and cleans up the file.
    """
    with tempfile.NamedTemporaryFile(suffix='.vasp', delete=False) as tmpfile:
        filename = my_gb.WriteGB(filename=tmpfile.name, overlap=0.0, whichG='g1', dim1=extend_by, dim2=1, dim3=1, file='VASP')
        structure = Structure.from_file(filename)
        os.remove(filename)
    return structure

def get_multiplier_to_extend_gb_to_min_length(structure, axis=0, req_length_grain=15):
    """
    Calculates the factor to extend the structure along a specific axis to meet the minimum grain length.
    Args:
        structure (pymatgen Structure): Structure to extend.
        axis: Axis along which to extend.
        req_length_grain: Minimum required grain length.
    Returns:
        List of factors to extend the structure along each axis.
    """
    lattice_length = structure.lattice.abc[axis]
    factor = int(np.ceil(req_length_grain * 2 / lattice_length))
    return [factor if i == axis else 1 for i in range(3)]

def rearrange_structure_lattice_vectors(structure, order=('a', 'b', 'c'), ensure_positive=True):
    """
    Reorders the lattice vectors of a pymatgen structure based on the specified order,
    adjusts fractional coordinates accordingly, and optionally ensures all lattice values
    are positive for consistency.
    
    Args:
        structure (pymatgen Structure): The structure to reorder.
        order (tuple): Desired order of lattice vectors, containing 'a', 'b', 'c' in any sequence.
        ensure_positive (bool): If True, makes all lattice vector values positive.

    Returns:
        pymatgen Structure: A new structure with reordered lattice vectors and coordinates.
    """
    # Validate input order
    if sorted(order) != ['a', 'b', 'c']:
        raise ValueError("Order must be a permutation of ('a', 'b', 'c').")
    
    # Map lattice vectors to 'a', 'b', 'c' labels for easy reordering
    lattice_vectors = {'a': structure.lattice.matrix[0], 
                       'b': structure.lattice.matrix[1], 
                       'c': structure.lattice.matrix[2]}
    ordered_lattice = [lattice_vectors[axis] for axis in order]
    
    # Make lattice vector values positive if ensure_positive is True
    if ensure_positive:
        ordered_lattice = [np.abs(vec) for vec in ordered_lattice]

    # Adjust fractional coordinates to match new lattice vector order
    coord_arrays = [[site.frac_coords[i] for site in structure] for i in range(3)]
    order_indices = [list('abc').index(axis) for axis in order]
    coords = [[coord_arrays[order_indices[j]][i] for j in range(3)] for i in range(len(structure))]
    
    # Re-create pymatgen Structure with reordered lattice and wrapped coordinates
    species = [site.specie for site in structure.sites]
    reordered_structure = Structure(ordered_lattice, species, coords, coords_are_cartesian=False)
    
    # Sort sites by fractional coordinate in the new third direction of the specified order
    reordered_structure.sort(lambda x: x.frac_coords[order_indices[2]])

    return reordered_structure

def align_lattice_to_axes(structure):
    """
    Aligns the structure's lattice vectors along the Cartesian axes.
    Returns:
        Aligned pymatgen Structure.
    """
    target_lattice_matrix = np.array([
        [structure.lattice.a, 0, 0], 
        [0, structure.lattice.b, 0], 
        [0, 0, structure.lattice.c]
    ])
    species = [site.species for site in structure]
    fractional_coords = [site.frac_coords for site in structure]
    return Structure(target_lattice_matrix, species, fractional_coords, coords_are_cartesian=False)

def get_realigned_structure(struct, arrange_ab_by_length=True, perform_equiv_check=False):
    """
    Reorders and aligns a structure to Cartesian axes, then checks for equivalence with the original.
    Args:
        struct: The pymatgen Structure to reorder and compare.
    Returns:
        bool indicating structural equivalence.
    """
    ## DEV NOTE: I KNOW IT LOOKS WEIRD THAT I DO THE LATTICE VECTOR REARRANGEMENT TWICE, BUT IT IS NECESSARY
    # IM TOO LAZY TO FIGURE OUT WHY THIS IS NECESSARY. (probably has something to do with aligning the cartesian axis w/lat. vectors)
    reordered_struct = struct.copy()
    # Apply the order to reorder the structure
    reordered_struct = rearrange_structure_lattice_vectors(reordered_struct, ("c", "b", "a"))
    #print(reordered_struct.lattice)
    #print()
    if arrange_ab_by_length:
        # Determine lengths of b and c
        b_length = struct.lattice.b
        a_length = struct.lattice.a
        # Set order with 'a' as the first, and the longer of 'b' and 'c' as the second
        order = ('a', 'b', 'c') if b_length >= a_length else ('b', 'a', 'c')
        reordered_struct = rearrange_structure_lattice_vectors(reordered_struct,
                                                               order = order)
        #print(reordered_struct.lattice)
        #print()
    reordered_struct = align_lattice_to_axes(reordered_struct)
    #print(reordered_struct.lattice)
    #print()
    if perform_equiv_check:
        matcher = StructureMatcher()
        is_equal = matcher.fit(struct, reordered_struct)
        #print("Reordered and aligned lattice:\n", reordered.lattice)
        print("Are structures equivalent?", is_equal)

    return reordered_struct

# Load the grain boundary data
# lattice_param = wf.a0.outputs.a0.value
# structure_lst = [
#     get_pmg_struct_from_gbcode(row.Axis, "bcc", lattice_param, row.m, row.n, row.GB1, "Fe")
#     for _, row in tqdm(df_gbcode.iterrows(), total=len(df_gbcode), desc="Processing rows")
# ]

