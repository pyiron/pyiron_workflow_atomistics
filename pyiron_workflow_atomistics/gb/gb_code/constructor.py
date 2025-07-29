import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from . import gb_generator as gbc
import pyiron_workflow as pwf
import tempfile
import sys
import glob
from pyiron_workflow import Workflow
from pymatgen.io.ase import AseAtomsAdaptor

from pyiron_snippets.logger import logger

@pwf.as_function_node("wrapped_sorted_structure")
def wrap_and_sort_structure(structure, axis=2):
    treated_struct = structure.copy()
    treated_struct.wrap()
    order = np.argsort(treated_struct.get_positions()[:, axis])

    # either reassign your variable to the sorted Atoms…
    treated_struct = treated_struct[order]

    return treated_struct


def get_pmg_struct_from_gbcode(
    axis,
    basis,
    lattice_param,
    m,
    n,
    GB1,
    element,
    req_length_grain=15,
    grain_length_axis=0,
):
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
    extend_factors = get_multiplier_to_extend_gb_to_min_length(
        structure, axis=grain_length_axis, req_length_grain=req_length_grain
    )

    # Extend GB structure
    structure = _write_and_load_structure(
        my_gb, extend_by=extend_factors[grain_length_axis]
    )

    # Map all atoms to the specified element
    element_mapping = {el: element for el in structure.species}
    structure.replace_species(element_mapping)
    return structure


def _write_and_load_structure(my_gb, extend_by=1):
    """
    Writes the GB to a temporary file, loads it as a pymatgen Structure, and cleans up the file.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    with tempfile.NamedTemporaryFile(suffix=".vasp", delete=False) as tmpfile:
        filename = my_gb.WriteGB(
            filename=tmpfile.name,
            overlap=0.0,
            whichG="g1",
            dim1=extend_by,
            dim2=1,
            dim3=1,
            file="VASP",
        )
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


def rearrange_structure_lattice_vectors(
    structure, order=("a", "b", "c"), ensure_positive=True
):
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
    if sorted(order) != ["a", "b", "c"]:
        raise ValueError("Order must be a permutation of ('a', 'b', 'c').")

    # Map lattice vectors to 'a', 'b', 'c' labels for easy reordering
    lattice_vectors = {
        "a": structure.lattice.matrix[0],
        "b": structure.lattice.matrix[1],
        "c": structure.lattice.matrix[2],
    }
    ordered_lattice = [lattice_vectors[axis] for axis in order]

    # Make lattice vector values positive if ensure_positive is True
    if ensure_positive:
        ordered_lattice = [np.abs(vec) for vec in ordered_lattice]

    # Adjust fractional coordinates to match new lattice vector order
    coord_arrays = [[site.frac_coords[i] for site in structure] for i in range(3)]
    order_indices = [list("abc").index(axis) for axis in order]
    coords = [
        [coord_arrays[order_indices[j]][i] for j in range(3)]
        for i in range(len(structure))
    ]

    # Re-create pymatgen Structure with reordered lattice and wrapped coordinates
    species = [site.specie for site in structure.sites]
    reordered_structure = Structure(
        ordered_lattice, species, coords, coords_are_cartesian=False
    )

    # Sort sites by fractional coordinate in the new third direction of the specified order
    reordered_structure.sort(lambda x: x.frac_coords[order_indices[2]])

    return reordered_structure


def align_lattice_to_axes(structure):
    """
    Aligns the structure's lattice vectors along the Cartesian axes.
    Returns:
        Aligned pymatgen Structure.
    """
    target_lattice_matrix = np.array(
        [
            [structure.lattice.a, 0, 0],
            [0, structure.lattice.b, 0],
            [0, 0, structure.lattice.c],
        ]
    )
    species = [site.species for site in structure]
    fractional_coords = [site.frac_coords for site in structure]
    return Structure(
        target_lattice_matrix, species, fractional_coords, coords_are_cartesian=False
    )


@pwf.as_function_node
def get_realigned_structure(
    struct, arrange_ab_by_length=True, perform_equiv_check=False
):
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
    reordered_struct = rearrange_structure_lattice_vectors(
        reordered_struct, ("c", "b", "a")
    )
    # print(reordered_struct.lattice)
    # print()
    if arrange_ab_by_length:
        # Determine lengths of b and c
        b_length = struct.lattice.b
        a_length = struct.lattice.a
        # Set order with 'a' as the first, and the longer of 'b' and 'c' as the second
        order = ("a", "b", "c") if b_length >= a_length else ("b", "a", "c")
        reordered_struct = rearrange_structure_lattice_vectors(
            reordered_struct, order=order
        )
        # print(reordered_struct.lattice)
        # print()
    reordered_struct = align_lattice_to_axes(reordered_struct)
    # print(reordered_struct.lattice)
    # print()
    if perform_equiv_check:
        matcher = StructureMatcher()
        is_equal = matcher.fit(struct, reordered_struct)
        # print("Reordered and aligned lattice:\n", reordered.lattice)
        logger.info(f"Are structures equivalent? {is_equal}")

    return reordered_struct


@pwf.as_function_node
def get_gbstruct_from_gbcode(
    axis=[1, 1, 0],
    basis="bcc",
    lattice_param=2.828,
    m=2,
    n=1,
    GB1=(-4, -1, -3),
    element="Fe",
    req_length_grain=15,
    grain_length_axis=0,
):
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
    extend_factors = get_multiplier_to_extend_gb_to_min_length(
        structure, axis=grain_length_axis, req_length_grain=req_length_grain
    )

    # Extend GB structure
    structure = _write_and_load_structure(
        my_gb, extend_by=extend_factors[grain_length_axis]
    )

    # Map all atoms to the specified element
    element_mapping = {el: element for el in structure.species}
    structure.replace_species(element_mapping)
    return structure


@pwf.as_function_node
def merge_structure_sites(structure, merge_dist_tolerance=1.3, merge_mode="average"):
    structure_merged = structure.copy()
    structure_merged.merge_sites(tol=merge_dist_tolerance, mode=merge_mode)
    logger.info(f"Merged {len(structure) - len(structure_merged)} sites")
    return structure_merged


@pwf.as_function_node
def get_expected_equilibrium_c_struct(struct, v0_per_atom, axis=2):
    from pymatgen.core.lattice import Lattice

    # Calculate the new lattice parameter along the specified axis to achieve the target volume per atom
    adj_equilibrium_vol = v0_per_atom * len(struct)
    # Get the current lattice parameters
    abc = list(struct.lattice.abc)
    # Indices of the two axes orthogonal to the specified axis
    axes_orth = [i for i in range(3) if i != axis]
    # Product of the two orthogonal lattice parameters
    orth_product = abc[axes_orth[0]] * abc[axes_orth[1]]
    # Compute the new lattice parameter along the specified axis
    adj_axis_length = adj_equilibrium_vol / orth_product
    # Prepare new lattice parameters
    new_abc = abc.copy()
    new_abc[axis] = adj_axis_length
    # Assume orthogonal cell (all angles 90)
    struct_eq = struct.copy()
    struct_eq.lattice = Lattice.from_parameters(*new_abc, 90, 90, 90)
    return struct_eq, adj_axis_length


@pwf.as_function_node
def convert_structure(struct, target="ase"):
    """
    Convert between ASE Atoms and Pymatgen Structure.

    Parameters:
    -----------
    struct : ase.Atoms or pymatgen.core.structure.Structure
        Input structure to convert.
    target : {'ase', 'pmg'}
        Conversion target: 'ase' for ASE Atoms, 'pmg' for Pymatgen Structure.

    Returns:
    --------
    Converted structure in the desired format.
    """
    # Determine conversion
    if target == "ase":
        converted_struct = AseAtomsAdaptor().get_atoms(struct)
    elif target in ("pmg", "pymatgen"):
        converted_struct = AseAtomsAdaptor().get_structure(struct)
    else:
        raise ValueError(f"Not a valid conversion target: {target}")
    return converted_struct


@Workflow.wrap.as_macro_node("original_GBcode_structure", "final_structure")
def construct_GB_from_GBCode(
    wf,
    axis,
    basis,
    lattice_param,
    m,
    n,
    GB1,
    element,
    req_length_grain,
    grain_length_axis,
    arrange_ab_by_length=True,
    perform_equiv_check=False,
    merge_dist_tolerance=1.3,
    merge_mode="average",
    equil_volume=None,
):
    """
    Macro node to build a grain boundary structure pipeline from a GB code.

    Parameters:
    -----------
    wf : pwf.Workflow
        The Workflow to which nodes are added.
    axis : list or array-like
        Miller direction vector for the GB code (e.g., [1,1,1]).
    basis : str
        Crystal basis, e.g., "bcc" or "fcc".
    lattice_param : float
        Lattice parameter for bulk and GB construction.
    m : int
        GB periodicity parameter m.
    n : int
        GB periodicity parameter n.
    GB1 : tuple of ints
        First GB plane vector, e.g., (-1, -1, 2).
    element : str
        Chemical symbol of the element, e.g., "Fe".
    req_length_grain : float
        Required grain length in Ångstroms.
    grain_length_axis : int
        Index of axis for grain length definition (0,1,2 for a,b,c).
    arrange_ab_by_length : bool, optional
        Whether to sort a/b lattice vectors by length (default True).
    perform_equiv_check : bool, optional
        Whether to check for equivalent sites during realign (default False).
    merge_dist_tolerance : float, optional
        Distance tolerance for merging sites in Å (default 1.3).
    merge_mode : str, optional
        Merge strategy, e.g., "average" or "first" (default "average").
    equil_volume : float, optional
        Equilibrium volume per atom for bulk structure (v0_per_atom).

    Returns:
    --------
        wf.structure : ase.Atoms
            The final GB structure.
        wf.sorted_structure : ase.Atoms
            The final GB structure sorted by the specified axis.
    """
    wf.gbcode_GBstruct = get_gbstruct_from_gbcode(
        axis=axis,
        basis=basis,
        lattice_param=lattice_param,
        m=m,
        n=n,
        GB1=GB1,
        element=element,
        req_length_grain=req_length_grain,
        grain_length_axis=grain_length_axis,
    )
    wf.gbplane_normal_aligned_c_struct = get_realigned_structure(
        wf.gbcode_GBstruct,
        arrange_ab_by_length=arrange_ab_by_length,
        perform_equiv_check=perform_equiv_check,
    )
    wf.merged_gbcode_GBstruct = merge_structure_sites(
        wf.gbplane_normal_aligned_c_struct,
        merge_dist_tolerance=merge_dist_tolerance,
        merge_mode=merge_mode,
    )
    wf.merged_gbcode_GBstruct_equilibrated_bulkvol = get_expected_equilibrium_c_struct(
        struct=wf.merged_gbcode_GBstruct, v0_per_atom=equil_volume
    )
    wf.structure = convert_structure(wf.gbcode_GBstruct, target="ase")
    wf.treated_struct = convert_structure(
        wf.merged_gbcode_GBstruct_equilibrated_bulkvol.outputs.struct_eq, target="ase"
    )
    wf.sorted_structure = wrap_and_sort_structure(wf.treated_struct)

    return (wf.structure, wf.sorted_structure)
