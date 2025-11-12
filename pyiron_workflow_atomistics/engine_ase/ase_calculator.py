import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write
from ase.optimize import BFGS
from pyiron_workflow_atomistics.dataclass_storage import CalcInputMD, EngineOutput


def ase_calc_structure(
    structure: Atoms,
    calc: Calculator,
    optimizer_class=BFGS,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    record_interval: int = 1,
    fmax: float = 0.01,
    max_steps: int = 10000,
    relax_cell: bool = False,
    energy_convergence_tolerance: Optional[float] = None,
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume"),
    write_to_disk: bool = False,
    working_directory: str = "calc_output",
    initial_struct_path: Optional[str] = "initial_structure.xyz",
    initial_results_path: Optional[str] = "initial_results.json",
    traj_struct_path: Optional[str] = "trajectory.xyz",
    traj_results_path: Optional[str] = "trajectory_results.json",
    final_struct_path: Optional[str] = "final_structure.xyz",
    final_results_path: Optional[str] = "final_results.json",
    data_pickle: str = "job_data.pkl.gz",
) -> EngineOutput:
    """
    Relax an ASE Atoms object with a customizable optimizer and recording interval,
    attach properties to each snapshot and write extended XYZ,
    and store trajectory data as a pickled DataFrame including structures.

    Returns
    -------
    EngineOutput object with final_structure, final_energy, final_forces, etc.
    """
    # Setup
    props = [p.strip() for p in properties]
    os.makedirs(working_directory, exist_ok=True)
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
        ase_write(os.path.join(working_directory, initial_struct_path), initial_atoms)
    if write_to_disk and initial_results_path:
        with open(os.path.join(working_directory, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    # Trajectory recording
    trajectory: List[Dict[str, Any]] = []

    def record_step():
        snap = atoms.copy()
        snap_res = gather(atoms)
        snap_att = attach_props(snap, snap_res)
        trajectory.append({"structure": snap_att, "results": snap_res})
        if write_to_disk and traj_struct_path:
            ase_write(
                os.path.join(working_directory, traj_struct_path), snap_att, append=True
            )

    # Optimize (or just calculate if optimizer_class is None)
    if optimizer_class is None:
        # Static calculation: just compute once
        converged = True
        record_step()  # Record the single calculation
    else:
        # Minimization: use optimizer
        # Handle cell relaxation if requested
        if relax_cell:
            from ase.constraints import ExpCellFilter
            # Wrap atoms with ExpCellFilter for cell relaxation
            atoms_filtered = ExpCellFilter(atoms)
            optimizer = optimizer_class(atoms_filtered, **optimizer_kwargs)
            # Need to record from the filtered atoms during optimization
            def record_step_filtered():
                # Get the actual atoms from the filter
                actual_atoms = atoms_filtered.atoms.copy()
                snap_res = gather(actual_atoms)
                snap_att = attach_props(actual_atoms, snap_res)
                trajectory.append({"structure": snap_att, "results": snap_res})
                if write_to_disk and traj_struct_path:
                    ase_write(
                        os.path.join(working_directory, traj_struct_path), snap_att, append=True
                    )
            optimizer.attach(record_step_filtered, interval=record_interval)
        else:
            optimizer = optimizer_class(atoms, **optimizer_kwargs)
            optimizer.attach(record_step, interval=record_interval)
        
        # Run optimizer with force convergence tolerance
        converged = optimizer.run(fmax=fmax, steps=max_steps)
        
        # If using ExpCellFilter, extract the updated atoms
        if relax_cell:
            atoms = atoms_filtered.atoms.copy()
        
        # Optional: check energy convergence if tolerance is specified
        # Note: ASE optimizers don't have built-in energy convergence,
        # so we check manually if energy_convergence_tolerance is provided
        if energy_convergence_tolerance is not None and energy_convergence_tolerance > 0:
            if trajectory:
                # Check if energy change between last two steps is below tolerance
                if len(trajectory) >= 2:
                    energy_diff = abs(
                        trajectory[-1]["results"].get("energy", 0) - 
                        trajectory[-2]["results"].get("energy", 0)
                    )
                    if energy_diff < energy_convergence_tolerance:
                        converged = True

    # Write trajectory results JSON
    if write_to_disk and traj_results_path:
        traj_res_list = [step["results"] for step in trajectory]
        with open(os.path.join(working_directory, traj_results_path), "w") as f:
            json.dump(traj_res_list, f, indent=2)

    # Final snapshot
    final_res = gather(atoms)
    final_atoms = attach_props(atoms.copy(), final_res)
    # print(final_res)
    final = {"structure": final_atoms, "results": final_res}
    if write_to_disk and final_struct_path:
        ase_write(os.path.join(working_directory, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(working_directory, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)

    # Build DataFrame including structures
    df = pd.DataFrame(
        [{"structure": step["structure"], **step["results"]} for step in trajectory]
    )
    df.to_pickle(os.path.join(working_directory, data_pickle), compression="gzip")

    # Convert to EngineOutput format
    engine_output = EngineOutput()
    engine_output.final_structure = final_atoms
    engine_output.final_results = final_res
    engine_output.convergence = bool(converged)
    
    # Extract final values
    if "energy" in final_res:
        engine_output.final_energy = final_res["energy"]
    if "forces" in final_res:
        engine_output.final_forces = np.array(final_res["forces"])
    if "stresses" in final_res:
        stress = np.array(final_res["stresses"])
        engine_output.final_stress = stress
        # For compatibility, also set stress tensor (voigt notation)
        if len(stress) == 6:  # Voigt notation
            engine_output.final_stress_tensor_voigt = stress
    if "volume" in final_res:
        engine_output.final_volume = final_res["volume"]
    else:
        engine_output.final_volume = final_atoms.get_volume()
    
    # Extract trajectory data
    if trajectory:
        engine_output.energies = [step["results"].get("energy") for step in trajectory]
        engine_output.forces = [np.array(step["results"].get("forces", [])) for step in trajectory if "forces" in step["results"]]
        if trajectory and "stresses" in trajectory[0]["results"]:
            engine_output.stresses = [np.array(step["results"]["stresses"]) for step in trajectory]
        engine_output.structures = [step["structure"] for step in trajectory]
        engine_output.n_ionic_steps = len(trajectory)
    
    # Extract magmoms if available
    if "magmoms" in final_res:
        engine_output.magmoms = final_res["magmoms"]
    
    return engine_output

def ase_md_calc_structure(
    structure: Atoms,
    calc: Calculator,
    md_input: CalcInputMD,
    record_interval: int = 1,
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume"),
    write_to_disk: bool = False,
    working_directory: str = "calc_output",
    initial_struct_path: Optional[str] = "initial_structure.xyz",
    initial_results_path: Optional[str] = "initial_results.json",
    traj_struct_path: Optional[str] = "trajectory.xyz",
    traj_results_path: Optional[str] = "trajectory_results.json",
    final_struct_path: Optional[str] = "final_structure.xyz",
    final_results_path: Optional[str] = "final_results.json",
    data_pickle: str = "job_data.pkl.gz",
) -> EngineOutput:
    """
    Run MD simulation with ASE using InputCalc dataclass for MD parameters.
    
    Returns
    -------
    EngineOutput object with final_structure, final_energy, final_forces, etc.
    """
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    
    # Import MD integrators and thermostats
    from ase.md import Langevin
    from ase.md.npt import NPT
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.nptberendsen import NPTBerendsen
    
    # Setup
    props = [p.strip() for p in properties]
    os.makedirs(working_directory, exist_ok=True)

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
        ase_write(os.path.join(working_directory, initial_struct_path), initial_atoms)
    if write_to_disk and initial_results_path:
        with open(os.path.join(working_directory, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    # Initialize velocities
    T0 = md_input.initial_temperature or md_input.temperature
    if T0 > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=np.random.RandomState(md_input.seed))

    # Trajectory recording
    trajectory: List[Dict[str, Any]] = []

    def record_step():
        snap = atoms.copy()
        snap_res = gather(atoms)
        snap_att = attach_props(snap, snap_res)
        trajectory.append({"structure": snap_att, "results": snap_res})
        if write_to_disk and traj_struct_path:
            ase_write(
                os.path.join(working_directory, traj_struct_path), snap_att, append=True
            )

    # Setup MD integrator based on ensemble and thermostat
    # Convert ps to fs: 1 ps = 1000 fs
    dt = md_input.time_step * 1000.0 * units.fs
    T = md_input.temperature
    
    if md_input.mode == "NVE":
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(atoms, dt)
    
    elif md_input.mode == "NVT":
        if md_input.thermostat == "nose-hoover":
            from ase.md.nvt import NVT
            dyn = NVT(atoms, dt, temperature_K=T, ttime=md_input.temperature_damping_timescale * units.fs)
        elif md_input.thermostat == "berendsen":
            dyn = NVTBerendsen(atoms, dt, temperature_K=T, taut=md_input.temperature_damping_timescale * units.fs)
        elif md_input.thermostat == "langevin":
            dyn = Langevin(atoms, dt, temperature_K=T, friction=1.0 / (md_input.temperature_damping_timescale * units.fs), rng=np.random.RandomState(md_input.seed))
        elif md_input.thermostat == "andersen":
            # Andersen thermostat via Langevin with high friction
            dyn = Langevin(atoms, dt, temperature_K=T, friction=1.0 / (md_input.temperature_damping_timescale * units.fs), rng=np.random.RandomState(md_input.seed))
        else:
            # Default to Langevin for other thermostats
            dyn = Langevin(atoms, dt, temperature_K=T, friction=1.0 / (md_input.temperature_damping_timescale * units.fs), rng=np.random.RandomState(md_input.seed))
    
    elif md_input.mode == "NPT":
        if md_input.pressure is None:
            raise ValueError("Pressure must be specified for NPT ensemble")
        P = md_input.pressure / (units.Pascal * 1e5)  # Convert Pa to bar
        if md_input.thermostat == "nose-hoover":
            dyn = NPT(atoms, dt, temperature_K=T, externalstress=P, ttime=md_input.temperature_damping_timescale * units.fs, pfactor=md_input.pressure_damping_timescale * units.fs)
        elif md_input.thermostat == "berendsen":
            dyn = NPTBerendsen(atoms, dt, temperature_K=T, pressure_au=P, taut=md_input.temperature_damping_timescale * units.fs, taup=md_input.pressure_damping_timescale * units.fs)
        else:
            raise ValueError(f"NPT mode supports only 'nose-hoover' or 'berendsen' thermostat, got '{md_input.thermostat}'")
    else:
        raise ValueError(f"Unknown MD mode: {md_input.mode}")

    # Attach logger and trajectory recorder
    dyn.attach(record_step, interval=record_interval)
    
    # Run MD
    dyn.run(md_input.n_ionic_steps)

    # Write trajectory results JSON
    if write_to_disk and traj_results_path:
        traj_res_list = [step["results"] for step in trajectory]
        with open(os.path.join(working_directory, traj_results_path), "w") as f:
            json.dump(traj_res_list, f, indent=2)

    # Final snapshot
    final_res = gather(atoms)
    final_atoms = attach_props(atoms.copy(), final_res)
    final = {"structure": final_atoms, "results": final_res}
    if write_to_disk and final_struct_path:
        ase_write(os.path.join(working_directory, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(working_directory, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)

    # Build DataFrame including structures
    df = pd.DataFrame(
        [{"structure": step["structure"], **step["results"]} for step in trajectory]
    )
    df.to_pickle(os.path.join(working_directory, data_pickle), compression="gzip")

    # Convert to EngineOutput format
    engine_output = EngineOutput()
    engine_output.final_structure = final_atoms
    engine_output.final_results = final_res
    engine_output.convergence = True  # MD always "converges" (completes)
    
    # Extract final values
    if "energy" in final_res:
        engine_output.final_energy = final_res["energy"]
    if "forces" in final_res:
        engine_output.final_forces = np.array(final_res["forces"])
    if "stresses" in final_res:
        stress = np.array(final_res["stresses"])
        engine_output.final_stress = stress
        # For compatibility, also set stress tensor (voigt notation)
        if len(stress) == 6:  # Voigt notation
            engine_output.final_stress_tensor_voigt = stress
    if "volume" in final_res:
        engine_output.final_volume = final_res["volume"]
    else:
        engine_output.final_volume = final_atoms.get_volume()
    
    # Extract trajectory data
    if trajectory:
        engine_output.energies = [step["results"].get("energy") for step in trajectory]
        engine_output.forces = [np.array(step["results"].get("forces", [])) for step in trajectory if "forces" in step["results"]]
        if trajectory and "stresses" in trajectory[0]["results"]:
            engine_output.stresses = [np.array(step["results"]["stresses"]) for step in trajectory]
        engine_output.structures = [step["structure"] for step in trajectory]
        engine_output.n_ionic_steps = len(trajectory)
    
    # Extract magmoms if available
    if "magmoms" in final_res:
        engine_output.magmoms = final_res["magmoms"]
    
    return engine_output

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
        # print(results_list, key)
        print(f"Error {e} when trying to parse output")
        extracted_values = np.nan
    return extracted_values


@pwf.as_function_node("full_calc_kwargs2")
def fillin_default_calckwargs(
    calc_kwargs: dict[str, Any],
    default_values: dict[str, Any] | None | str = None,
    remove_keys: list[str] | None = None,
) -> dict[str, Any]:
    # 1) overlay any user-supplied default overrides
    built_in = {}
    if isinstance(default_values, dict):
        built_in.update(default_values)

    # 2) start with everything user passed in
    full: dict[str, Any] = dict(calc_kwargs)

    # 3) fill in missing built-ins
    for key, default in built_in.items():
        full.setdefault(key, default)

    # 4) ensure properties is a tuple
    if "properties" in full:
        full["properties"] = tuple(full["properties"])

    # 5) remove any keys requested
    if remove_keys:
        for key in remove_keys:
            full.pop(key, None)

    return full


@pwf.as_function_node("kwargs_variant")
def generate_kwargs_variant(
    base_kwargs: dict[str, Any],
    key: str,
    value: Any,
):
    from copy import deepcopy

    kwargs_variant = deepcopy(base_kwargs)
    kwargs_variant[key] = value
    return kwargs_variant


@pwf.as_function_node("kwargs_variants")
def generate_kwargs_variants(
    base_kwargs: dict[str, Any],
    key: str,
    values: list[Any],
):
    """
    Given a base kwargs dict, produce one dict per value in `values`,
    each with `key` set to that value (overriding any existing entry).

    Parameters
    ----------
    base_kwargs
        The original kwargs to copy.
    key
        The dict key whose value you want to vary.
    values
        A list of values to assign to `key`.

    Returns
    -------
    List of dicts
        Each is a shallow copy of base_kwargs with base_kwargs[key] = value.
    """
    return_kwargs = [{**base_kwargs, key: v} for v in values]
    # print(return_kwargs)
    return return_kwargs
