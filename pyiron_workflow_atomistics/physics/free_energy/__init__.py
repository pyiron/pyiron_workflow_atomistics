"""Free-energy workflows: calphy + phonopy harmonic / QHA + dynaphopy anharmonic.

Public API
----------
Dataclasses:
    LammpsPotential  - calphy-only: pair_style + pair_coeff + optional potential_file
    FreeEnergyOutput - typed result of every node

calphy function-nodes (one per mode):
    free_energy, reversible_scaling_temperature, reversible_scaling_pressure,
    melting_temperature, alchemy, composition_scaling

Phonon free-energy macros (NEW):
    harmonic_free_energy             - phonopy FC2 at a fixed volume
    quasiharmonic_free_energy        - phonopy.qha.QHA on top of harmonic_free_energy
    anharmonic_free_energy_dynaphopy - dynaphopy renormalised harmonic at one T
    anharmonic_free_energy_dynaphopy_tdi - dynaphopy renormalised harmonic over a T grid

All node-and-adapter imports defer ``calphy`` / ``phonopy`` / ``dynaphopy`` /
``pyiron_workflow_lammps`` imports to node-body call time, so importing this
subpackage does not require any specific optional extra.
"""

from pyiron_workflow_atomistics.physics.free_energy.anharmonic_dynaphopy import (
    anharmonic_free_energy_dynaphopy,
    anharmonic_free_energy_dynaphopy_tdi,
)
from pyiron_workflow_atomistics.physics.free_energy.calphy import (
    alchemy,
    composition_scaling,
    free_energy,
    melting_temperature,
    reversible_scaling_pressure,
    reversible_scaling_temperature,
)
from pyiron_workflow_atomistics.physics.free_energy.harmonic import (
    harmonic_free_energy,
)
from pyiron_workflow_atomistics.physics.free_energy.inputs import LammpsPotential
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.free_energy.quasiharmonic import (
    quasiharmonic_free_energy,
)

__all__ = [
    "FreeEnergyOutput",
    "LammpsPotential",
    "alchemy",
    "anharmonic_free_energy_dynaphopy",
    "anharmonic_free_energy_dynaphopy_tdi",
    "composition_scaling",
    "free_energy",
    "harmonic_free_energy",
    "melting_temperature",
    "quasiharmonic_free_energy",
    "reversible_scaling_pressure",
    "reversible_scaling_temperature",
]
