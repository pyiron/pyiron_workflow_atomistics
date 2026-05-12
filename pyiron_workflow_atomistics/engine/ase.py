"""ASE-backed Engine implementation.

Consolidates and replaces engine_ase/{ase.py, ase_calculator.py, ase_engine.py}.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write
from ase.optimize import BFGS

from pyiron_workflow_atomistics.engine.inputs import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
)
from pyiron_workflow_atomistics.engine.protocol import EngineOutput


# ---------------------------------------------------------------------------
# Low-level helpers: gather() + attach_props()
# ---------------------------------------------------------------------------

def _gather(atoms: Atoms, properties: Tuple[str, ...]) -> dict[str, Any]:
    props = [p.strip() for p in properties]
    results: dict[str, Any] = {
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
            results["stresses"] = atoms.get_stress().tolist()
        except Exception:
            pass
    optional_map = {
        "charges": "get_charges",
        "dipole": "get_dipole_moment",
        "magmoms": "get_magnetic_moments",
        "virial": "get_virial",
        "pressure": "get_pressure",
    }
    for key, method in optional_map.items():
        if key in props:
            try:
                val = getattr(atoms, method)()
                results[key] = val.tolist() if hasattr(val, "tolist") else val
            except Exception:
                pass
    missing = [p for p in props if p not in results]
    if missing:
        raise KeyError(f"Requested properties not available: {missing}")
    return {p: results[p] for p in props}


def _attach_props(atoms: Atoms, results: dict[str, Any]) -> Atoms:
    if "energy" in results:
        atoms.info["energy"] = results["energy"]
    if "forces" in results:
        atoms.set_array("forces", np.array(results["forces"]))
    if "stresses" in results:
        atoms.info["stresses"] = results["stresses"]
    return atoms


def _build_engine_output(
    *,
    final_atoms: Atoms,
    final_res: dict[str, Any],
    trajectory: list[dict[str, Any]],
    converged: bool,
) -> EngineOutput:
    out = EngineOutput(
        final_structure=final_atoms,
        final_energy=float(final_res["energy"]),
        converged=bool(converged),
    )
    if "forces" in final_res:
        out.final_forces = np.array(final_res["forces"])
    if "stresses" in final_res:
        s = np.array(final_res["stresses"])
        if s.shape == (6,):
            out.final_stress_voigt = s
            # Reconstruct full 3x3 from Voigt convention (xx,yy,zz,yz,xz,xy)
            out.final_stress = np.array(
                [[s[0], s[5], s[4]],
                 [s[5], s[1], s[3]],
                 [s[4], s[3], s[2]]]
            )
        elif s.shape == (3, 3):
            out.final_stress = s
            out.final_stress_voigt = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
            )
    out.final_volume = float(final_res.get("volume", final_atoms.get_volume()))
    if "magmoms" in final_res:
        out.final_magmoms = np.array(final_res["magmoms"])
    if trajectory:
        out.energies = [step["results"].get("energy") for step in trajectory]
        out.forces = [
            np.array(step["results"]["forces"])
            for step in trajectory
            if "forces" in step["results"]
        ]
        if "stresses" in trajectory[0]["results"]:
            out.stresses = [np.array(step["results"]["stresses"]) for step in trajectory]
        out.structures = [step["structure"] for step in trajectory]
        out.n_ionic_steps = len(trajectory)
    return out


# ---------------------------------------------------------------------------
# Core calc functions (static/minimize and MD)
# ---------------------------------------------------------------------------

def ase_calc_structure(
    structure: Atoms,
    calc: Calculator,
    optimizer_class: Optional[type] = BFGS,
    optimizer_kwargs: Optional[dict[str, Any]] = None,
    record_interval: int = 1,
    fmax: float = 0.01,
    max_steps: int = 10_000,
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
    """Relax (or single-point) an ASE Atoms object and return an EngineOutput."""
    os.makedirs(working_directory, exist_ok=True)
    optimizer_kwargs = dict(optimizer_kwargs or {})

    atoms = structure.copy()
    atoms.calc = calc

    initial_res = _gather(atoms, properties)
    if write_to_disk and initial_struct_path:
        ase_write(os.path.join(working_directory, initial_struct_path), _attach_props(atoms.copy(), initial_res))
    if write_to_disk and initial_results_path:
        with open(os.path.join(working_directory, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    trajectory: list[dict[str, Any]] = []

    if optimizer_class is None:
        # Static
        snap = atoms.copy()
        snap_res = _gather(atoms, properties)
        trajectory.append({"structure": _attach_props(snap, snap_res), "results": snap_res})
        converged = True
    else:
        # Relaxation
        if relax_cell:
            from ase.constraints import ExpCellFilter
            atoms_filtered = ExpCellFilter(atoms)
            optimizer = optimizer_class(atoms_filtered, **optimizer_kwargs)

            def record_step():
                actual = atoms_filtered.atoms.copy()
                snap_res = _gather(actual, properties)
                trajectory.append({"structure": _attach_props(actual, snap_res), "results": snap_res})
                if write_to_disk and traj_struct_path:
                    ase_write(os.path.join(working_directory, traj_struct_path), _attach_props(actual.copy(), snap_res), append=True)
            optimizer.attach(record_step, interval=record_interval)
            converged = optimizer.run(fmax=fmax, steps=max_steps)
            atoms = atoms_filtered.atoms.copy()
        else:
            optimizer = optimizer_class(atoms, **optimizer_kwargs)

            def record_step():
                snap = atoms.copy()
                snap_res = _gather(atoms, properties)
                trajectory.append({"structure": _attach_props(snap, snap_res), "results": snap_res})
                if write_to_disk and traj_struct_path:
                    ase_write(os.path.join(working_directory, traj_struct_path), _attach_props(atoms.copy(), snap_res), append=True)
            optimizer.attach(record_step, interval=record_interval)
            converged = optimizer.run(fmax=fmax, steps=max_steps)

        if energy_convergence_tolerance and len(trajectory) >= 2:
            ediff = abs(trajectory[-1]["results"]["energy"] - trajectory[-2]["results"]["energy"])
            if ediff < energy_convergence_tolerance:
                converged = True

    final_res = _gather(atoms, properties)
    final_atoms = _attach_props(atoms.copy(), final_res)

    if write_to_disk and final_struct_path:
        ase_write(os.path.join(working_directory, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(working_directory, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)
    if write_to_disk and traj_results_path:
        with open(os.path.join(working_directory, traj_results_path), "w") as f:
            json.dump([step["results"] for step in trajectory], f, indent=2)

    df = pd.DataFrame([{"structure": s["structure"], **s["results"]} for s in trajectory])
    df.to_pickle(os.path.join(working_directory, data_pickle), compression="gzip")

    return _build_engine_output(
        final_atoms=final_atoms,
        final_res=final_res,
        trajectory=trajectory,
        converged=converged,
    )


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
    """Run MD with ASE using the CalcInputMD dataclass for ensemble settings."""
    from ase import units
    from ase.md import Langevin
    from ase.md.npt import NPT
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    os.makedirs(working_directory, exist_ok=True)
    atoms = structure.copy()
    atoms.calc = calc

    initial_res = _gather(atoms, properties)
    if write_to_disk and initial_struct_path:
        ase_write(os.path.join(working_directory, initial_struct_path), _attach_props(atoms.copy(), initial_res))
    if write_to_disk and initial_results_path:
        with open(os.path.join(working_directory, initial_results_path), "w") as f:
            json.dump(initial_res, f, indent=2)

    T0 = md_input.initial_temperature or md_input.temperature
    if T0 > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=np.random.RandomState(md_input.seed))

    trajectory: list[dict[str, Any]] = []

    def record_step():
        snap = atoms.copy()
        snap_res = _gather(atoms, properties)
        trajectory.append({"structure": _attach_props(snap, snap_res), "results": snap_res})
        if write_to_disk and traj_struct_path:
            ase_write(os.path.join(working_directory, traj_struct_path), _attach_props(atoms.copy(), snap_res), append=True)

    dt = md_input.time_step * units.fs   # CalcInputMD.time_step is in fs
    T = md_input.temperature
    ttime = md_input.thermostat_time_constant * units.fs

    if md_input.mode == "NVE":
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(atoms, dt)
    elif md_input.mode == "NVT":
        if md_input.thermostat == "nose-hoover":
            from ase.md.nvt import NVT
            dyn = NVT(atoms, dt, temperature_K=T, ttime=ttime)
        elif md_input.thermostat == "berendsen":
            dyn = NVTBerendsen(atoms, dt, temperature_K=T, taut=ttime)
        else:  # langevin or andersen → langevin
            dyn = Langevin(atoms, dt, temperature_K=T, friction=1.0 / ttime,
                           rng=np.random.RandomState(md_input.seed))
    elif md_input.mode == "NPT":
        if md_input.pressure is None:
            raise ValueError("Pressure must be specified for NPT ensemble")
        P_bar = md_input.pressure / 1e5
        taup = md_input.pressure_damping_timescale * units.fs
        if md_input.thermostat == "nose-hoover":
            dyn = NPT(atoms, dt, temperature_K=T, externalstress=P_bar, ttime=ttime, pfactor=taup)
        elif md_input.thermostat == "berendsen":
            dyn = NPTBerendsen(atoms, dt, temperature_K=T, pressure_au=P_bar, taut=ttime, taup=taup)
        else:
            raise ValueError(f"NPT supports only 'nose-hoover' or 'berendsen', got {md_input.thermostat!r}")
    else:
        raise ValueError(f"Unknown MD mode: {md_input.mode!r}")

    dyn.attach(record_step, interval=record_interval)
    dyn.run(md_input.n_ionic_steps)

    final_res = _gather(atoms, properties)
    final_atoms = _attach_props(atoms.copy(), final_res)

    if write_to_disk and final_struct_path:
        ase_write(os.path.join(working_directory, final_struct_path), final_atoms)
    if write_to_disk and final_results_path:
        with open(os.path.join(working_directory, final_results_path), "w") as f:
            json.dump(final_res, f, indent=2)
    if write_to_disk and traj_results_path:
        with open(os.path.join(working_directory, traj_results_path), "w") as f:
            json.dump([s["results"] for s in trajectory], f, indent=2)

    df = pd.DataFrame([{"structure": s["structure"], **s["results"]} for s in trajectory])
    df.to_pickle(os.path.join(working_directory, data_pickle), compression="gzip")

    return _build_engine_output(
        final_atoms=final_atoms,
        final_res=final_res,
        trajectory=trajectory,
        converged=True,
    )


# ---------------------------------------------------------------------------
# ASEEngine — the user-facing class
# ---------------------------------------------------------------------------

@dataclass
class ASEEngine:
    """An :class:`pyiron_workflow_atomistics.engine.protocol.Engine` backed by ASE."""

    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    calculator: Calculator
    working_directory: str = field(default_factory=os.getcwd)
    optimizer_class: type | None = BFGS
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    record_interval: int = 1
    max_steps: int = 10_000
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume")
    write_to_disk: bool = False
    initial_struct_path: Optional[str] = "initial_structure.xyz"
    initial_results_path: Optional[str] = "initial_results.json"
    traj_struct_path: Optional[str] = "trajectory.xyz"
    traj_results_path: Optional[str] = "trajectory_results.json"
    final_struct_path: Optional[str] = "final_structure.xyz"
    final_results_path: Optional[str] = "final_results.json"
    data_pickle: str = "job_data.pkl.gz"

    def get_calculate_fn(self, structure: Atoms):
        common = dict(
            calc=self.calculator,
            working_directory=self.working_directory,
            properties=self.properties,
            write_to_disk=self.write_to_disk,
            initial_struct_path=self.initial_struct_path,
            initial_results_path=self.initial_results_path,
            traj_struct_path=self.traj_struct_path,
            traj_results_path=self.traj_results_path,
            final_struct_path=self.final_struct_path,
            final_results_path=self.final_results_path,
            data_pickle=self.data_pickle,
        )
        if isinstance(self.EngineInput, CalcInputStatic):
            kwargs = {**common, "optimizer_class": None, "optimizer_kwargs": {},
                      "record_interval": 1, "fmax": 0.0, "max_steps": 0}
            return ase_calc_structure, kwargs
        if isinstance(self.EngineInput, CalcInputMinimize):
            mi = self.EngineInput
            kwargs = {**common,
                      "optimizer_class": self.optimizer_class,
                      "optimizer_kwargs": self.optimizer_kwargs,
                      "record_interval": self.record_interval,
                      "fmax": mi.force_convergence_tolerance,
                      "max_steps": self.max_steps if self.max_steps else mi.max_iterations,
                      "relax_cell": mi.relax_cell,
                      "energy_convergence_tolerance": mi.energy_convergence_tolerance}
            return ase_calc_structure, kwargs
        if isinstance(self.EngineInput, CalcInputMD):
            kwargs = {**common, "md_input": self.EngineInput, "record_interval": self.record_interval}
            return ase_md_calc_structure, kwargs
        raise TypeError(f"Unsupported EngineInput type: {type(self.EngineInput).__name__}")

    def with_working_directory(self, subdir: str) -> "ASEEngine":
        return replace(self, working_directory=os.path.join(self.working_directory, subdir))
