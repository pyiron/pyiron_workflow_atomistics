import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

from ase import Atoms
from ase.calculators.calculator import Calculator
from pyiron_workflow_atomistics.dataclass_storage import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
    Engine,
)


@dataclass
class ASEEngine(Engine):
    """
    ASE Engine using InputCalc dataclasses directly to configure calculations.
    Mode is inferred from EngineInput by checking key attributes; boilerplate defaults
    are defined on the engine via engine-specific attributes.
    Thermostat and ensemble logic is implemented directly here.
    """

    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    calculator: Calculator
    mode: Literal["static", "minimize", "md"] = field(init=False)
    working_directory: str = field(default_factory=os.getcwd)
    calc_fn: Callable = None
    calc_fn_kwargs: dict[str, Any] = None
    parse_fn: Callable = None
    parse_fn_kwargs: dict[str, Any] = None
    optimizer_class: Callable = None
    optimizer_kwargs: dict[str, Any] = None
    record_interval: int = 1
    fmax: float = 0.01
    max_steps: int = 10000
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume")
    write_to_disk: bool = False
    initial_struct_path: Optional[str] = "initial_structure.xyz"
    initial_results_path: Optional[str] = "initial_results.json"
    traj_struct_path: Optional[str] = "trajectory.xyz"
    traj_results_path: Optional[str] = "trajectory_results.json"
    final_struct_path: Optional[str] = "final_structure.xyz"
    final_results_path: Optional[str] = "final_results.json"
    data_pickle: str = "job_data.pkl.gz"

    def __post_init__(self):
        # Ensure the attribute exists and is consistent with EngineInput
        self.toggle_mode()
        # Normalize kwargs dicts
        if self.calc_fn_kwargs is None:
            self.calc_fn_kwargs = {}
        if self.parse_fn_kwargs is None:
            self.parse_fn_kwargs = {}
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}

    def toggle_mode(self):
        # Infer mode from EngineInput type
        if isinstance(self.EngineInput, CalcInputMinimize):
            inferred = "minimize"
            # Map dataclass attributes - optimizer_kwargs will be passed to optimizer constructor
            # Note: fmax and steps are passed to optimizer.run(), not constructor
            self.optimizer_kwargs = self.optimizer_kwargs or {}
        elif isinstance(self.EngineInput, CalcInputMD):
            inferred = "md"
            # MD parameters are passed via md_input object to ase_md_calc_structure
            # All MD parameters from CalcInputMD are already accessible via md_input
            pass
        elif isinstance(self.EngineInput, CalcInputStatic):
            inferred = "static"
            # Static runs: no further parameters needed
            pass
        else:
            raise TypeError(f"Unsupported EngineInput type: {type(self.EngineInput)}")

        # If mode not present (or explicitly None), assign it
        if not hasattr(self, "mode") or getattr(self, "mode", None) is None:
            self.mode = inferred
            return self.mode

        # mode already present -> warn and do NOT overwrite
        current = self.mode
        if current != inferred:
            warnings.warn(
                f"'mode' already set to '{current}', but EngineInput implies '{inferred}'. "
                "Keeping existing 'mode' (not overwriting). Ensure they agree.",
                RuntimeWarning,
            )
        else:
            warnings.warn(
                f"'mode' already set to '{current}' and also implied by EngineInput; no change.",
                RuntimeWarning,
            )
        return current

    def get_calculate_fn(self, structure: Atoms) -> tuple[Callable, dict[str, Any]]:
        if self.calc_fn is None:
            from pyiron_workflow_atomistics.engine_ase.ase_calculator import (
                ase_calc_structure,
                ase_md_calc_structure,
            )
            # Build kwargs based on mode
            # Note: 'structure' is NOT included here because calculate_structure_node
            # passes it explicitly as a keyword argument
            calc_kwargs = {
                "calc": self.calculator,
                "working_directory": self.working_directory,
                "properties": self.properties,
                "write_to_disk": self.write_to_disk,
                "initial_struct_path": self.initial_struct_path,
                "initial_results_path": self.initial_results_path,
                "traj_struct_path": self.traj_struct_path,
                "traj_results_path": self.traj_results_path,
                "final_struct_path": self.final_struct_path,
                "final_results_path": self.final_results_path,
                "data_pickle": self.data_pickle,
            }

            if self.mode == "static":
                # Static calculation: just compute energy/forces without optimization
                calc_kwargs.update(
                    {
                        "optimizer_class": None,  # No optimizer for static
                        "optimizer_kwargs": {},
                        "record_interval": 1,
                        "fmax": 0.0,  # Not used for static
                        "max_steps": 0,  # Not used for static
                    }
                )
                self.calc_fn = ase_calc_structure

            elif self.mode == "minimize":
                # Minimization: use optimizer with convergence criteria from EngineInput
                min_input = self.EngineInput
                calc_kwargs.update(
                    {
                        "optimizer_class": self.optimizer_class,
                        "optimizer_kwargs": self.optimizer_kwargs,
                        "record_interval": self.record_interval,
                        "fmax": min_input.force_convergence_tolerance,
                        "max_steps": (
                            self.max_steps
                            if self.max_steps is not None
                            else min_input.max_iterations
                        ),
                        "relax_cell": min_input.relax_cell,
                        "energy_convergence_tolerance": min_input.energy_convergence_tolerance,
                    }
                )
                self.calc_fn = ase_calc_structure

            elif self.mode == "md":
                # MD: use MD-specific function
                md_input = self.EngineInput
                calc_kwargs.update(
                    {
                        "md_input": md_input,
                        "record_interval": self.record_interval,
                    }
                )
                self.calc_fn = ase_md_calc_structure

            self.calc_fn_kwargs = calc_kwargs

        # Return a copy of kwargs to prevent modifications from affecting cached version
        # Also ensure 'structure' is not in kwargs (safety check)
        import copy
        returned_kwargs = copy.deepcopy(self.calc_fn_kwargs)
        returned_kwargs.pop("structure", None)  # Remove if somehow present
        
        return self.calc_fn, returned_kwargs

    def get_parse_fn(self) -> tuple[Callable, dict[str, Any]]:
        if self.parse_fn is None:
            # For ASE, parsing is typically done within the calculation function
            # But we can provide a simple passthrough or custom parser
            def parse_fn(**kwargs):
                return kwargs

            self.parse_fn = parse_fn
            self.parse_fn_kwargs = {}

        return self.parse_fn, self.parse_fn_kwargs
