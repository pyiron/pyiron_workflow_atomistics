import os
import json
from ase.optimize import BFGS
from ase.io import write as ase_write
from ase import Atoms
from ase.calculators.calculator import Calculator
import pyiron_workflow as pwf
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

def calc_structure(
    structure: Atoms,
    calc: Calculator,
    fmax: float = 0.01,
    max_steps: int = 10000,
    properties: Tuple[str, ...] = ('energy', 'forces', 'stresses'),
    write_to_disk: bool = False,
    output_dir: str = "calc_output",
    initial_struct_path: Optional[str] = 'initial_structure.xyz',
    initial_results_path: Optional[str] = 'initial_results.json',
    traj_struct_path: Optional[str] = 'trajectory.xyz',
    traj_results_path: Optional[str] = 'trajectory_results.json',
    final_struct_path: Optional[str] = 'final_structure.xyz',
    final_results_path: Optional[str] = 'final_results.json'
):
    """
    Relax an ASE Atoms object and optionally write snapshots to disk.

    Parameters
    ----------
    structure : Atoms
        Initial ASE Atoms object to be relaxed.
    calc : Calculator
        ASE calculator to assign to the atoms.
    fmax : float, optional
        Force convergence criterion (default: 0.01).
    max_steps : int, optional
        Maximum number of optimizer steps (default: 10000).
    properties : tuple of str, optional
        Properties to gather, e.g., ('energy', 'forces', 'stresses').
    write_to_disk : bool, optional
        If True, write snapshots and results to disk under `output_dir`.
    output_dir : str, optional
        Directory to create for output files (default: 'calc_output').
    initial_struct_path : str or None, optional
        Filename for initial structure snapshot in extended XYZ format.
        If `None`, no initial‐structure file is written (default: 'initial_structure.xyz').
    initial_results_path : str or None, optional
        Filename for initial results JSON.
        If `None`, no initial‐results file is written (default: 'initial_results.json').
    traj_struct_path : str or None, optional
        Filename for trajectory extended XYZ file.
        If `None`, no trajectory‐structure file is written (default: 'trajectory.xyz').
    traj_results_path : str or None, optional
        Filename for trajectory results JSON.
        If `None`, no trajectory‐results file is written (default: 'trajectory_results.json').
    final_struct_path : str or None, optional
        Filename for final structure snapshot in extended XYZ format.
        If `None`, no final‐structure file is written (default: 'final_structure.xyz').
    final_results_path : str or None, optional
        Filename for final results JSON.
        If `None`, no final‐results file is written (default: 'final_results.json').

    Returns
    -------
    dict
        Dictionary with keys:
        - 'initial': {'structure': Atoms, 'results': dict}
        - 'trajectory': list of {'structure': Atoms, 'results': dict}
        - 'final': {'structure': Atoms, 'results': dict}
        - 'converged': bool

    Raises
    ------
    KeyError
        If a requested property is not available on the Atoms object.
    """
    # Clean up property names
    props = [p.strip() for p in properties]

    def gather(atoms: Atoms) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {
            'energy':    atoms.get_potential_energy(),
            'forces':    atoms.get_forces().tolist(),
            'cell':      atoms.get_cell().tolist(),
            'volume':    atoms.get_volume(),
            'positions': atoms.get_positions().tolist(),
            'numbers':   atoms.get_atomic_numbers().tolist(),
            'masses':    atoms.get_masses().tolist(),
        }
        if 'stresses' in props:
            try:
                all_results['stresses'] = atoms.get_stress().tolist()
            except Exception:
                pass
        mapping = {
            'charges': 'get_charges',
            'dipole': 'get_dipole_moment',
            'magmoms': 'get_magnetic_moments',
            'virial': 'get_virial',
            'pressure': 'get_pressure'
        }
        for key, method in mapping.items():
            if key in props:
                try:
                    val = getattr(atoms, method)()
                    all_results[key] = val.tolist() if hasattr(val, 'tolist') else val
                except Exception:
                    pass
        missing = [p for p in props if p not in all_results]
        if missing:
            raise KeyError(f"Requested properties not available: {missing}")
        return {p: all_results[p] for p in props}

    atoms = structure.copy()
    atoms.calc = calc


    os.makedirs(output_dir, exist_ok=True)

    # 1) Initial snapshot
    initial = {'structure': atoms.copy(), 'results': gather(atoms)}
    if write_to_disk:
        if initial_struct_path is not None:
            ase_write(os.path.join(output_dir, initial_struct_path), initial['structure'])
        if initial_results_path is not None:
            with open(os.path.join(output_dir, initial_results_path), 'w') as f:
                json.dump(initial['results'], f, indent=2)

    # 2) Trajectory
    trajectory: List[Dict[str, Any]] = []
    def record_step():
        snap = atoms.copy()
        snap_res = gather(atoms)
        trajectory.append({'structure': snap, 'results': snap_res})
        if write_to_disk and traj_struct_path is not None:
            ase_write(os.path.join(output_dir, traj_struct_path), snap, append=True)
    # Prepare output directory
    if write_to_disk:
        optimizer = BFGS(atoms, trajectory=f'{output_dir}/opt.asecalc.traj', logfile=f'{output_dir}/opt.asecalc.log')
    else:
        optimizer = BFGS(atoms)
    optimizer.attach(record_step, interval=1)
    converged = optimizer.run(fmax=fmax, steps=max_steps)

    if write_to_disk and traj_results_path is not None:
        traj_res_list = [step['results'] for step in trajectory]
        with open(os.path.join(output_dir, traj_results_path), 'w') as f:
            json.dump(traj_res_list, f, indent=2)

    # 3) Final snapshot
    final = {'structure': atoms.copy(), 'results': gather(atoms)}
    if write_to_disk:
        if final_struct_path is not None:
            ase_write(os.path.join(output_dir, final_struct_path), final['structure'])
        if final_results_path is not None:
            with open(os.path.join(output_dir, final_results_path), 'w') as f:
                json.dump(final['results'], f, indent=2)

    return {
        'initial':    initial,
        'trajectory': trajectory,
        'final':      final,
        'converged':  converged
    }


@pwf.as_function_node("atoms", "results", "converged")
def calculate_structure_node(
     # structure: Atoms,
     # calc: Calculator,
     # fmax: float = 0.01,
     # max_steps: int = 1000,
     # properties: Tuple[str, ...] = ('energy', 'forces', 'stresses'),
     # write_to_disk: bool = False,
     # output_dir: str = "calc_output",
     # initial_struct_path: Optional[str] = 'initial_structure.xyz',
     # initial_results_path: Optional[str] = 'initial_results.json',
     # traj_struct_path: Optional[str] = 'trajectory.xyz',
     # traj_results_path: Optional[str] = 'trajectory_results.json',
     # final_struct_path: Optional[str] = 'final_structure.xyz',
     # final_results_path: Optional[str] = 'final_results.json',
    structure,
    calc,
    fmax=0.01,
    max_steps=10000,
    properties=('energy', 'forces', 'stresses'),
    write_to_disk=False,
    output_dir="calc_output",
    initial_struct_path='initial_structure.xyz',
    initial_results_path='initial_results.json',
    traj_struct_path='trajectory.xyz',
    traj_results_path='trajectory_results.json',
    final_struct_path='final_structure.xyz',
    final_results_path='final_results.json'
 ):
     """
     ASE relaxation with full disk‐writing under output_dir of initial, trajectory, and final data;
     any path set to `None` will simply not be written.

     Returns
     -------
     atoms : ase.Atoms
         The final relaxed structure.
     final_results : dict
         The final requested properties.
     converged : bool
         Whether the relaxation converged.
     """
     out = calc_structure(
         structure=structure,
         calc=calc,
         fmax=fmax,
         max_steps=max_steps,
         properties=properties,
         write_to_disk=write_to_disk,
         output_dir=output_dir,
         initial_struct_path=initial_struct_path,
         initial_results_path=initial_results_path,
         traj_struct_path=traj_struct_path,
         traj_results_path=traj_results_path,
         final_struct_path=final_struct_path,
         final_results_path=final_results_path
     )

     atoms = out['final']['structure']
     final_results = out['final']['results']
     converged = out['converged']
     return atoms, final_results, converged

@pwf.as_function_node("output")
def extract_values(results_list, key):
    """
    Extract a list of values for a specified key from a list of result dictionaries.

    Parameters
    ----------
    results_list : list of dict
        Each dict should contain the specified key.
    key : str
        The dictionary key to extract values for (e.g., 'energy', 'volume').

    Returns
    -------
    values : list
        List of values corresponding to key from each dict.

    Raises
    ------
    KeyError
        If any entry in results_list is missing the specified key.
    """
    try:
        extracted_values = [entry[key] for entry in results_list]
    except Exception as e:
        print(f"Error {e} when trying to parse output")
        extracted_values = np.nan
    return extracted_values

@pwf.as_function_node("full_calc_kwargs2")
def fillin_default_calckwargs(calc_kwargs, default_values=None):
    """
    Take a partial calc_kwargs dict and fill in any missing entries
    with the standard defaults, but allow an optional `default_values`
    dict to override any of those built-in defaults.

    Parameters
    ----------
    calc_kwargs : dict
        User-provided kwargs for the calculation (may be partial).
    default_values : dict, optional
        If provided, these key→value pairs will override the built-in defaults.

    Returns
    -------
    full_calc_kwargs : dict
        A dict containing every argument, using user values when present,
        then `default_values`, then the built-in defaults.
    """
    # 1) define your built-in defaults
    built_in = {
        "output_dir":           "calc_dir",
        "fmax":                 0.01,
        "max_steps":            1000,
        "properties":           ('energy', 'forces', 'stresses'),
        "write_to_disk":        False,
        "initial_struct_path":  'initial_structure.xyz',
        "initial_results_path": 'initial_results.json',
        "traj_struct_path":     'trajectory.xyz',
        "traj_results_path":    'trajectory_results.json',
        "final_struct_path":    'final_structure.xyz',
        "final_results_path":   'final_results.json',
    }

    # 2) overlay any user-supplied default overrides
    if default_values:
        built_in.update(default_values)

    # 3) build the final dict: user → default_values → built-in
    full = {}
    for key, default in built_in.items():
        if key in calc_kwargs:
            full[key] = calc_kwargs[key]
        else:
            full[key] = default

    # 4) ensure tuple for properties
    full["properties"] = tuple(full["properties"])

    return full