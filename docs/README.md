# pyiron_workflow_atomistics

## Overview

This repository contains a pyiron module for atomistic simulation workflows, providing tools and utilities for working with atomic structures, calculations, and various atomistic studies. The package integrates seamlessly with the pyiron workflow system and supports multiple computational backends through a unified engine interface.

## Features

- **Calculation Engines**: Unified interface for different computational backends:
  - **ASE Engine**: Integration with ASE (Atomic Simulation Environment) for DFT, force fields, and ML potentials
  - **LAMMPS Engine**: Integration with LAMMPS for classical MD simulations
  - **Custom Engine Support**: Framework for implementing your own computational backend

- **Bulk Material Calculations**: Tools for bulk material properties:
  - Equation of state (EOS) fitting
  - Cubic lattice parameter optimization
  - Volume-energy scans
  - Bulk structure generation

- **Surface Calculations**: Surface energy and structure analysis:
  - Surface slab generation
  - Surface energy calculations
  - Surface structure optimization

- **Bulk Defects**: Point defect calculations:
  - Vacancy formation energy
  - Interstitial defect calculations
  - Defect structure generation

- **Grain Boundary Analysis**: Tools for analyzing and manipulating grain boundaries:
  - GB plane detection and analysis
  - Cleavage plane identification
  - GB segregation studies
  - GB structure manipulation

- **Structure Manipulation**: Utilities for working with atomic structures:
  - Structure featurization
  - Interstitial insertion
  - Structure transformation tools

- **Workflow Integration**: Seamless integration with pyiron workflow system:
  - Automated structure calculations
  - Standardized input/output formats (`CalcInputStatic`, `CalcInputMinimize`, `CalcInputMD`, `EngineOutput`)
  - Data processing and analysis
  - Results visualization

## Installation

The package can be installed via pip:

```bash
pip install pyiron_workflow_atomistics
```

Or via conda:

```bash
conda install -c conda-forge pyiron_workflow_atomistics
```

## Dependencies

The package requires:
- Python >= 3.9, < 3.14
- numpy == 1.26.4
- pandas == 2.3.2
- matplotlib == 3.10.6
- ase == 3.26.0
- scipy == 1.16.2
- pyiron-workflow == 0.15.2
- pymatgen == 2025.6.14
- pyiron_snippets == 0.2.0
- scikit-learn == 1.7.2

Note: For development or if you need different versions, you can install from source and adjust dependencies as needed.

## Usage

### Using Calculation Engines

The package provides engines for different computational backends. Here's how to use the ASE engine:

```python
from pyiron_workflow_atomistics.engine_ase.ase import ASEEngine
from pyiron_workflow_atomistics.dataclass_storage import CalcInputMinimize
from ase.optimize import BFGS
from ase.build import bulk
from mace.calculators import MACECalculator

# Create input parameters
inp = CalcInputMinimize()
inp.force_convergence_tolerance = 0.01
inp.relax_cell = False

# Create calculator
calculator = MACECalculator(model_path='model.model', device='cpu')

# Create engine
engine = ASEEngine(
    EngineInput=inp,
    calculator=calculator,
    optimizer_class=BFGS,
    working_directory="calculations"
)

# Use with workflow functions
from pyiron_workflow_atomistics.bulk import optimise_cubic_lattice_parameter
from pyiron_workflow import Workflow

wf = Workflow("my_workflow")
wf.opt = optimise_cubic_lattice_parameter(
    structure=bulk("Fe", a=2.88, cubic=True),
    name="Fe",
    crystalstructure="bcc",
    calculation_engine=engine,
    parent_working_directory="opt_cubic_cell",
    num_points=11
)
wf.run()
```

### Bulk Material Calculations

```python
from pyiron_workflow_atomistics.bulk import (
    optimise_cubic_lattice_parameter,
    eos_volume_scan,
    equation_of_state
)
from pyiron_workflow import Workflow

wf = Workflow("bulk_calc")

# Optimize cubic lattice parameter
wf.opt = optimise_cubic_lattice_parameter(
    structure=atoms,
    name="Fe",
    crystalstructure="bcc",
    calculation_engine=engine,
    strain_range=(-0.02, 0.02),
    num_points=11,
    eos_type="birchmurnaghan"
)

# Access results
equilibrium_lattice_param = wf.opt.outputs.a0
bulk_modulus = wf.opt.outputs.B
equilibrium_energy = wf.opt.outputs.equil_energy_per_atom
```

### Surface Calculations

```python
from pyiron_workflow_atomistics.surface.surface_study import surface_energy_study
from pyiron_workflow import Workflow

wf = Workflow("surface_calc")
wf.surface = surface_energy_study(
    structure=bulk_structure,
    miller_indices=(1, 1, 1),
    calculation_engine=engine,
    vacuum=10.0
)
wf.run()
```

### Bulk Defects

```python
from pyiron_workflow_atomistics.bulk_defect.vacancy import vacancy_formation_energy
from pyiron_workflow import Workflow

wf = Workflow("defect_calc")
wf.vacancy = vacancy_formation_energy(
    structure=bulk_structure,
    calculation_engine=engine
)
wf.run()
```

### Grain Boundary Analysis

```python
from pyiron_workflow_atomistics.gb.analysis import find_GB_plane
from pyiron_workflow_atomistics.gb.cleavage import cleave_gb_structure

# Find GB plane in a structure
gb_info = find_GB_plane(atoms, featuriser, axis="c")

# Cleave structure at GB
cleaved_structures, cleavage_planes = cleave_gb_structure(
    base_structure=atoms,
    axis_to_cleave="c",
    target_coord=target_coord
)
```

### Structure Calculations

```python
from pyiron_workflow_atomistics.calculator import calculate_structure_node

# Run structure calculations with an engine
results = calculate_structure_node(
    structure=atoms,
    calculation_engine=engine
)

# Or with explicit function and kwargs
results = calculate_structure_node(
    structure=atoms,
    _calc_structure_fn=my_calc_function,
    _calc_structure_fn_kwargs={"param": "value"}
)
```

## Implementing Custom Engines

To implement your own calculation engine, you need to create a class that inherits from `Engine` and implements the required methods. The engine system allows you to integrate any computational backend (e.g., DFT codes, force fields, ML potentials) into the pyiron workflow framework.

### Requirements

Your custom engine must:

1. **Inherit from `Engine`**: Your engine class should be a dataclass inheriting from `pyiron_workflow_atomistics.dataclass_storage.Engine`

2. **Accept `EngineInput` dataclasses**: The engine should accept one of the standardized input dataclasses:
   - `CalcInputStatic` - for single-point energy/force calculations
   - `CalcInputMinimize` - for structure optimization/minimization
   - `CalcInputMD` - for molecular dynamics simulations

3. **Implement `get_calculate_fn()` method**: This method must return a tuple `(Callable, dict[str, Any])` where:
   - The `Callable` is a function that performs the calculation
   - The `dict` contains keyword arguments to pass to the function
   - **Important**: The function should accept `structure: Atoms` as its first argument (or as a keyword argument), but `structure` should NOT be included in the returned kwargs dict

4. **Return `EngineOutput` objects**: Your calculation function must return an `EngineOutput` object (not a tuple or dict). The `EngineOutput` class has the following attributes that should be populated:

   ```python
   class EngineOutput:
       final_structure = None          # ASE Atoms object of final structure
       final_results = None            # Raw results dict (optional)
       convergence = None              # bool indicating if calculation converged
       final_energy = None             # float: final energy
       final_forces = None            # numpy array: final forces
       final_stress = None            # numpy array: final stress (pressure)
       final_volume = None            # float: final volume
       final_stress_tensor = None     # numpy array: full stress tensor (optional)
       final_stress_tensor_voigt = None  # numpy array: stress in Voigt notation (optional)
       energies = None                # list: trajectory energies
       forces = None                  # list: trajectory forces
       stresses = None                # list: trajectory stresses
       structures = None              # list: trajectory structures (ASE Atoms objects)
       magmoms = None                # numpy array: magnetic moments (optional)
       n_ionic_steps = None          # int: number of ionic steps
   ```

### Example Implementation

Here's a minimal example of a custom engine:

```python
import os
from dataclasses import dataclass, field
from typing import Any, Literal
from ase import Atoms
from pyiron_workflow_atomistics.dataclass_storage import (
    Engine,
    CalcInputStatic,
    CalcInputMinimize,
    CalcInputMD,
    EngineOutput
)

@dataclass
class MyCustomEngine(Engine):
    """
    Custom engine for your computational backend.
    """
    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    working_directory: str = field(default_factory=os.getcwd)
    mode: Literal["static", "minimize", "md"] = field(init=False)
    # Add your engine-specific parameters here
    my_backend_config: dict[str, Any] = None
    
    def __post_init__(self):
        # Infer mode from EngineInput type
        if isinstance(self.EngineInput, CalcInputMinimize):
            self.mode = "minimize"
        elif isinstance(self.EngineInput, CalcInputMD):
            self.mode = "md"
        elif isinstance(self.EngineInput, CalcInputStatic):
            self.mode = "static"
        else:
            raise TypeError(f"Unsupported EngineInput type: {type(self.EngineInput)}")
    
    def get_calculate_fn(self, structure: Atoms) -> tuple[Callable, dict[str, Any]]:
        """
        Return the calculation function and its kwargs.
        
        The function should:
        - Accept 'structure' as first argument (or keyword)
        - Return an EngineOutput object
        - NOT include 'structure' in the returned kwargs
        """
        from my_backend import my_calculation_function
        
        # Build kwargs based on mode and EngineInput
        calc_kwargs = {
            "working_directory": self.working_directory,
            # Map EngineInput parameters to your backend's parameters
            # For minimize mode:
            # "fmax": self.EngineInput.force_convergence_tolerance,
            # "max_steps": self.EngineInput.max_iterations,
            # "relax_cell": self.EngineInput.relax_cell,
            # For MD mode, pass the entire md_input object or extract specific fields
            # "md_input": self.EngineInput,  # or extract specific fields
        }
        
        return my_calculation_function, calc_kwargs
```

### Calculation Function Requirements

Your calculation function (returned by `get_calculate_fn()`) must:

1. **Accept structure**: The function signature should be:
   ```python
   def my_calculation_function(
       structure: Atoms,
       **kwargs
   ) -> EngineOutput:
   ```

2. **Return EngineOutput**: Create and populate an `EngineOutput` object:
   ```python
   def my_calculation_function(structure: Atoms, **kwargs) -> EngineOutput:
       # Perform your calculation
       final_atoms = ...  # Your final structure
       final_energy = ...  # Your final energy
       final_forces = ...  # Your final forces
       
       # Create EngineOutput
       output = EngineOutput()
       output.final_structure = final_atoms
       output.final_energy = final_energy
       output.final_forces = np.array(final_forces)
       output.final_volume = final_atoms.get_volume()
       output.convergence = True  # or False
       
       # For trajectory data (if available)
       if trajectory:
           output.energies = [step["energy"] for step in trajectory]
           output.forces = [np.array(step["forces"]) for step in trajectory]
           output.structures = [step["structure"] for step in trajectory]
           output.n_ionic_steps = len(trajectory)
       
       return output
   ```

3. **Map EngineInput parameters**: Extract relevant parameters from `EngineInput` dataclasses:
   - **CalcInputMinimize**: `force_convergence_tolerance`, `energy_convergence_tolerance`, `max_iterations`, `relax_cell`
   - **CalcInputMD**: `mode`, `thermostat`, `temperature`, `n_ionic_steps`, `time_step`, `pressure`, etc.
   - **CalcInputStatic**: No parameters (empty dataclass)

### Reference Implementations

For complete examples, see:
- **ASE Engine**: `pyiron_workflow_atomistics.engine_ase.ase.ASEEngine`
- **LAMMPS Engine**: `pyiron_workflow_lammps.engine.LammpsEngine`

These implementations demonstrate:
- Mode inference from `EngineInput` types
- Parameter mapping from dataclasses to backend-specific parameters
- Proper `EngineOutput` population
- Handling of static, minimize, and MD modes

### Integration with Workflows

Once implemented, your engine can be used with any workflow function that accepts a `calculation_engine` parameter:

```python
from pyiron_workflow_atomistics.bulk import optimise_cubic_lattice_parameter

my_engine = MyCustomEngine(
    EngineInput=CalcInputMinimize(),
    my_backend_config={"param": "value"}
)

wf.opt = optimise_cubic_lattice_parameter(
    structure=atoms,
    calculation_engine=my_engine,
    # ... other parameters
)
```

## Documentation

For detailed documentation, visit our [ReadTheDocs page](https://pyiron_workflow_atomistics.readthedocs.io).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.rst) for details.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pyiron_workflow_atomistics,
  author = {pyiron team},
  title = {pyiron_workflow_atomistics},
  year = {2024},
  url = {https://github.com/pyiron/pyiron_workflow_atomistics}
}
```
