"""Elastic constants via the Materials Project stress-strain method.

Computes the full 6x6 stiffness tensor and all derived constants listed at
https://docs.materialsproject.org/methodology/materials-methodology/elasticity
using pymatgen.analysis.elasticity (the same library Materials Project uses),
driven by any ASE-backed Engine.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

EV_PER_A3_TO_GPA = 160.21766208
# ASE get_stress is tension-positive (Cauchy), same as pymatgen elasticity.
# Validated by the EMT test (Cu must give C11 > 0); flip to -1.0 if inverted.
_ASE_STRESS_SIGN = 1.0


def voigt_stress_to_gpa(stress_voigt) -> np.ndarray:
    """ASE Voigt stress (eV/A^3, order [xx,yy,zz,yz,xz,xy]) -> 3x3 tensor in GPa."""
    s = np.asarray(stress_voigt, dtype=float) * _ASE_STRESS_SIGN * EV_PER_A3_TO_GPA
    xx, yy, zz, yz, xz, xy = s
    return np.array(
        [[xx, xy, xz],
         [xy, yy, yz],
         [xz, yz, zz]]
    )


def with_calc_input(engine, calc_input):
    """Return a copy of a dataclass engine with its EngineInput replaced.

    Lets the elastic macro switch a single user-supplied engine between
    full-relax and fixed-cell-relax modes without the user wiring two engines.
    """
    return dataclasses.replace(engine, EngineInput=calc_input)


@pwf.as_function_node("deformed_structures", "strains")
def generate_mp_deformations(
    structure: Atoms,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
):
    """MP-standard deformation set (6 modes x 4 magnitudes = 24 cells).

    Returns the deformed ASE structures and their Green-Lagrange strain
    tensors (3x3), paired in order for the downstream fit.
    """
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg = AseAtomsAdaptor.get_structure(structure)
    dss = DeformedStructureSet(
        pmg,
        norm_strains=list(norm_strains),
        shear_strains=list(shear_strains),
    )
    deformed_structures = [AseAtomsAdaptor.get_atoms(s) for s in dss.deformed_structures]
    strains = [np.asarray(d.green_lagrange_strain) for d in dss.deformations]
    return deformed_structures, strains


@pwf.as_function_node("stresses")
def extract_stresses_gpa(engine_outputs):
    """3x3 stress tensors in GPa from a list of EngineOutput (input order)."""
    return [voigt_stress_to_gpa(o.final_stress_voigt) for o in engine_outputs]
