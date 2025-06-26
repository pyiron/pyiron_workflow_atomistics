# pyiron_workflow_atomistics

## Overview

This repository contains a pyiron module for atomistic simulation workflows, providing tools and utilities for working with atomic structures, grain boundaries, and various atomistic calculations.

## Features

- **Grain Boundary Analysis**: Tools for analyzing and manipulating grain boundaries, including:
  - GB plane detection and analysis
  - Cleavage plane identification
  - Structure manipulation for GB studies

- **Structure Manipulation**: Utilities for working with atomic structures:
  - Bulk structure handling
  - Structure featurization
  - Calculator integration

- **Workflow Integration**: Seamless integration with pyiron workflow system for:
  - Automated structure calculations
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
- Python >= 3.9, < 3.13
- numpy < 2.0.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- ase >= 3.22.0
- scipy >= 1.7.0
- pyiron_workflow
- pymatgen >= 2024.8.8
- pyiron_snippets
- scikit-learn >= 1.0.0

## Usage

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

# Run structure calculations
results = calculate_structure_node(
    structure=atoms,
    calc=calculator,
    output_dir="calculations"
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
