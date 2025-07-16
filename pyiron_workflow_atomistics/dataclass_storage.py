from dataclasses import dataclass
import ase

@dataclass
class BuildBulkStructure_Input:
    element_name: str
    crystalstructure: str = None
    a: float = None 
    b: float = None
    c: float = None
    alpha: float = None
    covera: float = None
    u: float = None
    orthorhombic: bool = False
    cubic: bool = False
    basis: list[list[float]] = None
    structure: ase.Atoms = None

    """
    Dataclass to build a bulk structure.
    Parameters
    ----------
    element_name : str
        The name of the element to build the bulk structure for.
    crystalstructure : str
        The crystal structure of the element.
    a : float
        The lattice parameter of the bulk structure.
    b : float
        The lattice parameter of the bulk structure.
    c : float
        The lattice parameter of the bulk structure.
    alpha : float
        The lattice parameter of the bulk structure.
    orthorhombic : bool
        Whether to build an orthorhombic structure.
    cubic : bool
        Whether to build a cubic structure.
    basis : list[list[float]]
        The basis of the bulk structure.
    Returns
    -------
    structure : ase.Atoms
        The bulk structure.
    """