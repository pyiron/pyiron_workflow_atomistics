# Utility functions for surface analysis
import numpy as np
import pyiron_workflow as pwf
from ase import Atoms
from ase.build import surface, bulk
from ase.data import atomic_numbers
from typing import Union, List, Tuple, Optional


@pwf.as_function_node("surface_slab")
def create_surface_slab(
    bulk_structure: Atoms,
    miller_indices: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (1, 1, 1),
    layers: int = 3,
    vacuum: float = 10.0,
    periodic: bool = True
) -> Atoms:
    """
    Create a surface slab from a bulk structure using Miller indices.
    
    Parameters:
    -----------
    bulk_structure : ase.Atoms
        The bulk crystal structure
    miller_indices : tuple of ints, default (1, 1, 1)
        Miller indices (h, k, l) for the surface plane
    layers : int, default 3
        Number of layers in the slab
    vacuum : float, default 10.0
        Vacuum thickness in Angstroms
    periodic : bool, default True
        Whether to make the slab periodic in the surface plane
        
    Returns:
    --------
    ase.Atoms
        Surface slab structure
    """
    slab = surface(
        bulk_structure,
        miller_indices,
        layers=layers,
        vacuum=vacuum,
        periodic=periodic
    )
    return slab


@pwf.as_function_node("bulk_from_symbol")
def create_bulk_from_symbol(
    symbol: str,
    crystalstructure: str = "fcc",
    a: Optional[float] = None,
    cubic: bool = False,
    orthorhombic: bool = False,
    b: Optional[float] = None,
    c: Optional[float] = None,
    covera: Optional[float] = None,
    u: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    ab: Optional[float] = None,
    magmom: Optional[float] = None,
    latticeconstant: Optional[float] = None
) -> Atoms:
    """
    Create a bulk crystal structure from element symbol and crystal structure type.
    
    Parameters:
    -----------
    symbol : str
        Chemical symbol of the element
    crystalstructure : str, default "fcc"
        Crystal structure type ("fcc", "bcc", "hcp", "diamond", "zincblende", "rocksalt", "cesiumchloride", "fluorite", "wurtzite")
    a : float, optional
        Lattice parameter a
    cubic : bool, default False
        Whether to create cubic structure
    orthorhombic : bool, default False
        Whether to create orthorhombic structure
    b : float, optional
        Lattice parameter b
    c : float, optional
        Lattice parameter c
    covera : float, optional
        c/a ratio for hexagonal structures
    u : float, optional
        Internal parameter for wurtzite
    alpha, beta, gamma : float, optional
        Unit cell angles
    ab : float, optional
        a/b ratio
    magmom : float, optional
        Magnetic moment per atom
    latticeconstant : float, optional
        Lattice constant (used if a is not specified)
        
    Returns:
    --------
    ase.Atoms
        Bulk crystal structure
    """
    kwargs = {}
    if a is not None:
        kwargs['a'] = a
    if b is not None:
        kwargs['b'] = b
    if c is not None:
        kwargs['c'] = c
    if covera is not None:
        kwargs['covera'] = covera
    if u is not None:
        kwargs['u'] = u
    if alpha is not None:
        kwargs['alpha'] = alpha
    if beta is not None:
        kwargs['beta'] = beta
    if gamma is not None:
        kwargs['gamma'] = gamma
    if ab is not None:
        kwargs['ab'] = ab
    if magmom is not None:
        kwargs['magmom'] = magmom
    if latticeconstant is not None:
        kwargs['latticeconstant'] = latticeconstant
        
    return bulk(
        symbol,
        crystalstructure,
        cubic=cubic,
        orthorhombic=orthorhombic,
        **kwargs
    )

@pwf.as_function_node("surface_structure")
def create_surface(symbol: str,
    miller_indices: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (1, 1, 1),
    min_length: int | float | np.float64 = 50,
    vacuum: float | np.float64 | int = 10.0,
    crystalstructure: str = "fcc",
    a: Optional[float] = None,
    cubic: bool = False,
    periodic: bool = True,
    b: Optional[float] = None,
    c: Optional[float] = None,
    covera: Optional[float] = None,
    u: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    ab: Optional[float] = None,
    magmom: Optional[float] = None,
    latticeconstant: Optional[float] = None):
    length = 0
    layers = 0
    while length < min_length:
        layers += 1
        surface = create_surface_with_layers.node_function(symbol, miller_indices, layers, vacuum, crystalstructure, a, cubic, periodic, b, c, covera, u, alpha, beta, gamma, ab, magmom, latticeconstant)
        length = surface.cell[-1][-1]
    surface = create_surface_with_layers.node_function(symbol, miller_indices, layers, vacuum, crystalstructure, a, cubic, periodic, b, c, covera, u, alpha, beta, gamma, ab, magmom, latticeconstant)
    return surface
    
@pwf.as_function_node("surface_structure")
def create_surface_with_layers(
    symbol: str,
    miller_indices: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (1, 1, 1),
    layers: int = 3,
    vacuum: float = 10.0,
    crystalstructure: str = "fcc",
    a: Optional[float] = None,
    cubic: bool = False,
    periodic: bool = True,
    b: Optional[float] = None,
    c: Optional[float] = None,
    covera: Optional[float] = None,
    u: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    ab: Optional[float] = None,
    magmom: Optional[float] = None,
    latticeconstant: Optional[float] = None
) -> Atoms:
    """
    Create a surface slab directly from element symbol and crystal structure.
    
    Parameters:
    -----------
    symbol : str
        Chemical symbol of the element
    miller_indices : tuple of ints, default (1, 1, 1)
        Miller indices (h, k, l) for the surface plane
    layers : int, default 3
        Number of layers in the slab
    vacuum : float, default 10.0
        Vacuum thickness in Angstroms
    crystalstructure : str, default "fcc"
        Crystal structure type
    a : float, optional
        Lattice parameter a
    cubic : bool, default False
        Whether to create cubic structure
    periodic : bool, default True
        Whether to make the slab periodic in the surface plane
        
    Returns:
    --------
    ase.Atoms
        Surface slab structure
    """
    # Create bulk structure
    bulk_kwargs = {
        'crystalstructure': crystalstructure,
        'cubic': cubic
    }
    if a is not None:
        bulk_kwargs['a'] = a
    if b is not None:
        bulk_kwargs['b'] = b
    if c is not None:
        bulk_kwargs['c'] = c
    if covera is not None:
        bulk_kwargs['covera'] = covera
    if u is not None:
        bulk_kwargs['u'] = u
    if alpha is not None:
        bulk_kwargs['alpha'] = alpha
    if beta is not None:
        bulk_kwargs['beta'] = beta
    if gamma is not None:
        bulk_kwargs['gamma'] = gamma
    if ab is not None:
        bulk_kwargs['ab'] = ab
    if magmom is not None:
        bulk_kwargs['magmom'] = magmom
    if latticeconstant is not None:
        bulk_kwargs['latticeconstant'] = latticeconstant
        
    bulk_structure = bulk(symbol, **bulk_kwargs)
    
    # Create surface slab
    slab = surface(
        bulk_structure,
        miller_indices,
        layers=layers,
        vacuum=vacuum,
        periodic=periodic
    )
    return slab


@pwf.as_function_node("surface_info")
def get_surface_info(slab: Atoms) -> dict:
    """
    Get information about a surface slab.
    
    Parameters:
    -----------
    slab : ase.Atoms
        Surface slab structure
        
    Returns:
    --------
    dict
        Dictionary containing surface information
    """
    cell = slab.get_cell()
    positions = slab.get_positions()
    
    # Calculate surface area
    surface_area = np.linalg.norm(np.cross(cell[0], cell[1]))
    
    # Calculate slab thickness (excluding vacuum)
    z_positions = positions[:, 2]
    z_min, z_max = np.min(z_positions), np.max(z_positions)
    slab_thickness = z_max - z_min
    
    # Calculate vacuum thickness
    total_height = cell[2, 2]
    vacuum_thickness = total_height - slab_thickness
    
    # Count atoms
    num_atoms = len(slab)
    
    # Get unique elements
    unique_elements = list(set(slab.get_chemical_symbols()))
    
    return {
        'num_atoms': num_atoms,
        'unique_elements': unique_elements,
        'surface_area': surface_area,
        'slab_thickness': slab_thickness,
        'vacuum_thickness': vacuum_thickness,
        'total_height': total_height,
        'cell_vectors': cell,
        'chemical_formula': slab.get_chemical_formula()
    }