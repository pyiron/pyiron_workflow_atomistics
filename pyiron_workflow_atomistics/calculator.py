import os
import json
from ase.optimize import BFGS
from ase.io import write as ase_write
from ase import Atoms
from ase.calculators.calculator import Calculator
import pyiron_workflow as pwf
import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional, List
import pandas as pd

def ase_calc_structure(
    structure: Atoms,
    calc: Calculator,
    optimizer_class = BFGS,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    record_interval: int = 1,
    fmax: float = 0.01,
    max_steps: int = 10000,
    properties: Tuple[str, ...] = ("energy", "forces", "stresses"),
    write_to_disk: bool = False,
    output_dir: str = "calc_output",
    initial_struct_path: Optional[str] = "initial_structure.xyz",
    initial_results_path: Optional[str] = "initial_results.json",
    traj_struct_path: Optional[str] = "trajectory.xyz",
    traj_results_path: Optional[str] = "trajectory_results.json",
    final_struct_path: Optional[str] = "final_structure.xyz",
    final_results_path: Optional[str] = "final_results.json",
    data_pickle: str = "job_data.pkl.gz",
) -> Dict[str, Any]:
    """
    Relax an ASE Atoms object with a customizable optimizer and recording interval,
    attach properties to each snapshot and write extended XYZ,
    and store trajectory data as a pickled DataFrame including structures.

    Returns
    -------
    dict with keys:
      - initial: {{structure, results}}
      - trajectory: list of {{structure, results}}
      - final: {{structure, results}}
      - converged: bool
    """
    # Setup
    props = [p.strip() for p in properties]
    os.makedirs(output_dir, exist_ok=True)
    optimizer_kwargs = optimizer_kwargs or {}

    def gather(atoms: Atoms) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces().tolist(),
            "cell": atoms.get_cell().tolist(),
            "volume": atoms.get_volume(),
            "positions": atoms.get_positions().tolist(),
            "numbers": atoms.get_atomic_numbers().tolist(),
            "masses": atoms.get_masses().tolist(),
        }
        if "stresses" in props:
            try:
                all_results["stresses"] = atoms.get_stress().tolist()
            except Exception:
                pass
        mapping = {
            "charges": "get_charges",
            "dipole": "get_dipole_moment",
            "magmoms": "get_magnetic_moments",
            "virial": "get_virial",
            "pressure": "get_pressure",
        }
        for key, method in mapping.items():
            if key in props:
                try:
                    val = getattr(atoms, method)()
                    all_results[key] = val.tolist() if hasattr(val, "tolist") else val
                except Exception:
                    pass
        missing = [p for p in props if p not in all_results]
        if missing:
            raise KeyError(f"Requested properties not available: {missing}")
        return {p: all_results[p] for p in props}

    def attach_props(atoms: Atoms, results: Dict[str, Any]):
        # Attach energy
        if "energy" in results:
            atoms.info["energy"] = results["energy"]
        # Attach forces
        if "forces" in results:
            atoms.set_array("forces", np.array(results["forces"]))
        # Attach stresses
        if "stresses" in results:
            atoms.info["stresses"] = results["stresses"]
        return atoms

    atoms = structure.copy()
    atoms.calc = calc

    # Initial snapshot
    initial_res = gather(atoms)
    initial_atoms = attach_props(atoms.copy(), initial_res)
    initial = {"structure": initial_atoms, "results": initial_res}
    if write_to_disk and initial_struct_path:
        ase_write(os.path.join(output_dir, initial_struct_path), initial_atoms)
    if write_to_disk and initial_results_path:
        with open(os.path.join(output_dir, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    # Trajectory recording
    trajectory: List[Dict[str, Any]] = []
    def record_step():
        snap = atoms.copy()
        snap_res = gather(atoms)
        snap_att = attach_props(snap, snap_res)
        trajectory.append({"structure": snap_att, "results": snap_res})
        if write_to_disk and traj_struct_path:
            ase_write(os.path.join(output_dir, traj_struct_path), snap_att, append=True)

    # Optimize
    optimizer = optimizer_class(atoms, **optimizer_kwargs)
    optimizer.attach(record_step, interval=record_interval)
    converged = optimizer.run(fmax=fmax, steps=max_steps)

    # Write trajectory results JSON
    if write_to_disk and traj_results_path:
        traj_res_list = [step["results"] for step in trajectory]
        with open(os.path.join(output_dir, traj_results_path), "w") as f:
            json.dump(traj_res_list, f, indent=2)

    # Final snapshot
    final_res = gather(atoms)
    final_atoms = attach_props(atoms.copy(), final_res)
    final = {"structure": final_atoms, "results": final_res}
    if write_to_disk and final_struct_path:
        ase_write(os.path.join(output_dir, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(output_dir, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)

    # Build DataFrame including structures
    df = pd.DataFrame([
        {"structure": step["structure"], **step["results"]}
        for step in trajectory
    ])
    df.to_pickle(os.path.join(output_dir, data_pickle), compression="gzip")

    return {"initial": initial, "trajectory": trajectory, "final": final, "converged": bool(converged)}

def ase_calculate_structure_node_interface(
    structure: Atoms,
    calc: Calculator,
    optimizer_class = BFGS,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    record_interval: int = 1,
    fmax: float = 0.01,
    max_steps: int = 10000,
    properties: Tuple[str, ...] = ("energy", "forces", "stresses"),
    write_to_disk: bool = False,
    output_dir: str = "calc_output",
    initial_struct_path: Optional[str] = "initial_structure.xyz",
    initial_results_path: Optional[str] = "initial_results.json",
    traj_struct_path: Optional[str] = "trajectory.xyz",
    traj_results_path: Optional[str] = "trajectory_results.json",
    final_struct_path: Optional[str] = "final_structure.xyz",
    final_results_path: Optional[str] = "final_results.json",
    data_pickle: str = "job_data.pkl.gz",
    calc_structure_fn: Callable[..., Any] = ase_calc_structure
) -> Tuple[Atoms, Dict[str, Any], bool]:
    """
    Node wrapper to call calc_structure_fn (default: ase_calc_structure)
    with all ASE kwargs forwarded from node inputs.
    """
    out = calc_structure_fn(
        structure=structure,
        calc=calc,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        record_interval=record_interval,
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
        final_results_path=final_results_path,
        data_pickle=data_pickle
    )
    atoms = out["final"]["structure"]
    final_results = out["final"]["results"]
    converged = out["converged"]
    converged = bool(converged)
    return atoms, final_results, converged

@pwf.as_function_node("atoms", "results", "converged")
def calculate_structure_node(
    structure: Atoms,
    calc_structure_fn: Callable[..., Any] = ase_calculate_structure_node_interface,
    calc_structure_fn_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Atoms, Dict[str, Any], bool]:
    if calc_structure_fn_kwargs is None:
        calc_structure_fn_kwargs = {}
    atoms, final_results, converged = calc_structure_fn(structure = structure,
                                                        **calc_structure_fn_kwargs)
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
def fillin_default_calckwargs(
    calc_kwargs: Dict[str, Any],
    default_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Take a partial calc_kwargs dict and fill in any missing entries
    with the ASE-relaxation defaults (including optimizer settings),
    allowing optional overrides of the built-ins.
    
    The goal is to return a **complete** dictionary of all calc_kwargs so no errors are ever returned by calc

    Parameters
    ----------
    calc_kwargs
        User-provided overrides for this run.
    default_values
        Optional overrides for the built-in default settings.

    Returns
    -------
    full_calc_kwargs
        Fully populated kwargs ready for `ase_calc_structure`.
    """
    # 1) define built-in defaults
    built_in: Dict[str, Any] = {
        "optimizer_class": BFGS,
        "optimizer_kwargs": None,
        "record_interval": 1,
        "fmax": 0.01,
        "max_steps": 10000,
        "properties": ("energy", "forces", "stresses"),
        "write_to_disk": False,
        "output_dir": "calc_output",
        "initial_struct_path": "initial_structure.xyz",
        "initial_results_path": "initial_results.json",
        "traj_struct_path": "trajectory.xyz",
        "traj_results_path": "trajectory_results.json",
        "final_struct_path": "final_structure.xyz",
        "final_results_path": "final_results.json",
        "data_pickle": "job_data.pkl.gz",
    }

    # 2) overlay any user-supplied default overrides
    if default_values:
        built_in.update(default_values)

    # 3) build the final dict: user → default_values → built-in
    full: Dict[str, Any] = {}
    for key, default in built_in.items():
        if key in calc_kwargs:
            full[key] = calc_kwargs[key]
        else:
            full[key] = default

    # 4) ensure properties is a tuple
    full["properties"] = tuple(full["properties"])

    return full