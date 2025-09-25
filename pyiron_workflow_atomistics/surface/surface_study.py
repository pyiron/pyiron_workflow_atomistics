import numpy as np
from typing import Union, Tuple, Optional
from pyiron_workflow_atomistics.surface.builder import create_surface
from pyiron_workflow_atomistics.calculator import calculate_structure_node
import pyiron_workflow as pwf

@pwf.as_function_node("calc_output")
def _calculate_if_not_present_(input_structure, 
                                calculation_engine,
                                mu_bulk = None,
                                ):
    if mu_bulk is None:
        output = calculate_structure_node.node_function(input_structure, calculation_engine=calculation_engine)
        mu_bulk_out = output.final_energy/len(input_structure)
    else:
        mu_bulk_out = mu_bulk
    return mu_bulk_out

@pwf.as_function_node("surface_energy")
def get_surface_energy(E_slab, E_bulk_per_atom, N_slab, area_one_side):
    # gamma in eV/Å^2
    gamma_fs = (E_slab - N_slab * E_bulk_per_atom) / (2.0 * area_one_side)
    gamma_J_per_m2 = gamma_fs * 16.021766208
    return gamma_J_per_m2

@pwf.as_function_node("area_one_side")
def area_one_side(slab):
    cell = slab.cell
    return np.linalg.norm(np.cross(cell[0], cell[1]))  # Å^2

@pwf.as_function_node("n_atoms")
def get_n_atoms(atoms):
    return len(atoms)
    
@pwf.as_macro_node("unrelaxed_surface",
                   "relaxed_surface",
                   "relaxed_surface_calc_output",
                   "mu_bulk",
                   "surface_energy")
def calculate_surface_energy(
    wf,
    calculation_engine,
    symbol: str = "Fe",
    miller_indices: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (1, 1, 1),
    min_length: int | float | np.float64 = 50,
    vacuum: float | np.float64 | int = 10.0,
    crystalstructure: str = "fcc",
    calc_structure_fn = None,
    calc_structure_fn_kwargs = None,
    mu_bulk: Optional[float] = None,
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
    """
    Calculate the surface energy of a surface (J/m^2).
    calculation_engine: any atomistics Engine
    symbol: str
    miller_indices: tuple
    min_length: float
    vacuum: float
    crystalstructure: str
    mu_bulk: float, the bulk per-atom chemical potential of the crystal
    a: float
    cubic: bool = False
    periodic: bool = True
    b: float = None
    c: float = None
    covera: float = None
    u: float = None
    alpha: float = None
    """
    wf.slab_novac = create_surface(symbol=symbol,
                                    miller_indices=miller_indices,
                                    min_length=min_length,
                                    vacuum=0,
                                    crystalstructure=crystalstructure,
                                    a=a,
                                    cubic=cubic,
                                    periodic=periodic,
                                    b=b,
                                    c=c,
                                    covera=covera,
                                    u=u,
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma,
                                    ab=ab,
                                    magmom=magmom,
                                    latticeconstant=latticeconstant)
    from pyiron_workflow_atomistics.calculator import validate_calculation_inputs
    wf.validate = validate_calculation_inputs(
        calculation_engine=calculation_engine,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )
    from pyiron_workflow_atomistics.utils import (
        get_calc_fn_calc_fn_kwargs_from_calculation_engine,
    )

    wf.calc_fn_calc_fn_kwargs = get_calc_fn_calc_fn_kwargs_from_calculation_engine(
        calculation_engine=calculation_engine,
        structure=wf.slab_novac,
        calc_structure_fn=calc_structure_fn,
        calc_structure_fn_kwargs=calc_structure_fn_kwargs,
    )
    wf.slab_vac = create_surface(symbol=symbol,
                                miller_indices=miller_indices,
                                min_length=min_length,
                                vacuum=vacuum,
                                crystalstructure=crystalstructure,
                                a=a,
                                cubic=cubic,
                                periodic=periodic,
                                b=b,
                                c=c,
                                covera=covera,
                                u=u,
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                ab=ab,
                                magmom=magmom,
                                latticeconstant=latticeconstant)
    wf.calc_slab = calculate_structure_node(wf.slab_vac,
                                            calculation_engine=calculation_engine,
                                            _calc_structure_fn=calc_structure_fn,
                                            _calc_structure_fn_kwargs=calc_structure_fn_kwargs)
    wf.mu_bulk_out = _calculate_if_not_present_(wf.slab_novac,
                                                calculation_engine=calculation_engine,
                                                mu_bulk=mu_bulk)
    wf.n_atoms_slab = get_n_atoms(wf.slab_vac)
    wf.area_one_side = area_one_side(wf.slab_novac)
    wf.surface_energy = get_surface_energy(E_slab = wf.calc_slab.outputs.calc_output.final_energy, 
                                           E_bulk_per_atom = wf.mu_bulk_out, 
                                           N_slab = wf.n_atoms_slab , 
                                           area_one_side = wf. area_one_side)
    return (wf.slab_vac,
            wf.calc_slab.outputs.calc_output.final_structure,
            wf.calc_slab,
            wf.mu_bulk_out,
            wf.surface_energy)
