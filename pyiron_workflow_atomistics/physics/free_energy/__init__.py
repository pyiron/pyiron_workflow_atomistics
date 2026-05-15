"""calphy-backed free-energy workflows.

Public API
----------

Dataclasses:
    LammpsPotential  - pair_style + pair_coeff + optional potential_file
    FreeEnergyOutput - typed result of every node

Function-nodes (one per calphy mode):
    free_energy                       - mode='fe'
    reversible_scaling_temperature    - mode='ts'
    reversible_scaling_pressure       - mode='pscale'
    melting_temperature               - mode='melting_temperature'
    alchemy                           - mode='alchemy'
    composition_scaling               - mode='composition_scaling'

All node-and-adapter imports defer ``calphy`` and ``pyiron_workflow_lammps``
imports to node-body call time, so importing this subpackage does not
require the ``[free-energy]`` extra.
"""

from pyiron_workflow_atomistics.physics.free_energy.calphy import (
    alchemy,
    composition_scaling,
    free_energy,
    melting_temperature,
    reversible_scaling_pressure,
    reversible_scaling_temperature,
)
from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput

__all__ = [
    "FreeEnergyOutput",
    "LammpsPotential",
    "alchemy",
    "composition_scaling",
    "free_energy",
    "melting_temperature",
    "reversible_scaling_pressure",
    "reversible_scaling_temperature",
]
