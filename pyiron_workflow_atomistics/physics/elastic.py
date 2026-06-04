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


@pwf.as_function_node("elastic_tensor")
def fit_elastic_tensor(strains, stresses, structure: Atoms, eq_stress=None):
    """Least-squares fit of the 6x6 stiffness tensor (GPa), MP convention.

    strains: list of 3x3 strain tensors (Green-Lagrange).
    stresses: list of 3x3 stress tensors in GPa (same order).
    structure: the reference (relaxed) ASE structure, used for symmetry.
    eq_stress: 3x3 reference stress in GPa (defaults to zero).
    Returns a pymatgen ElasticTensor, symmetrized.
    """
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity.strain import Strain
    from pymatgen.analysis.elasticity.stress import Stress

    pmg_strains = [Strain(s) for s in strains]
    pmg_stresses = [Stress(s) for s in stresses]
    eq = None if eq_stress is None else Stress(eq_stress)
    et = ElasticTensor.from_independent_strains(
        pmg_strains, pmg_stresses, eq_stress=eq, vasp=False
    )
    return ElasticTensor(et.voigt_symmetrized)


@pwf.as_function_node("elastic_constants")
def elastic_constants_summary(elastic_tensor, structure: Atoms) -> dict:
    """Every elastic constant in the MP elasticity methodology, as a flat dict.

    Includes full stiffness (raw + IEEE) and compliance tensors, bulk and shear
    moduli (Voigt/Reuss/Hill), Young's modulus, Poisson ratio, universal
    anisotropy, and a mechanical-stability flag (Born criteria, cubic + general).
    Moduli in GPa.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    et = elastic_tensor
    pmg = AseAtomsAdaptor.get_structure(structure)
    try:
        et_ieee = et.convert_to_ieee(pmg)
    except Exception:
        et_ieee = et

    C = np.asarray(et.voigt)
    # General Born stability: stiffness matrix positive-definite
    eigvals = np.linalg.eigvalsh(C)
    mech_stable = bool(np.all(eigvals > 0))

    d = {
        "K_Voigt": float(et.k_voigt),
        "K_Reuss": float(et.k_reuss),
        "K_VRH": float(et.k_vrh),
        "G_Voigt": float(et.g_voigt),
        "G_Reuss": float(et.g_reuss),
        "G_VRH": float(et.g_vrh),
        # pymatgen's y_mod is Pa-scale here; normalize to GPa.
        "youngs_modulus": float(et.y_mod) / 1e9 if et.y_mod > 1e6 else float(et.y_mod),
        "poisson_ratio": float(et.homogeneous_poisson),
        "universal_anisotropy": float(et.universal_anisotropy),
        "mechanically_stable": mech_stable,
        "stiffness_eigenvalues": eigvals.tolist(),
        "elastic_tensor_voigt": C.tolist(),
        "elastic_tensor_ieee": np.asarray(et_ieee.voigt).tolist(),
        "compliance_tensor_voigt": np.asarray(et.compliance_tensor.voigt).tolist(),
    }
    return d


@pwf.as_function_node("engine")
def with_calc_input_node(engine, calc_input):
    """Node wrapper around :func:`with_calc_input` for use inside the macro graph."""
    return with_calc_input(engine, calc_input)


@pwf.as_function_node("eq_stress")
def _reference_stress_gpa(engine_output):
    """Reference (relaxed) stress in GPa as a 3x3 tensor, for eq_stress."""
    return voigt_stress_to_gpa(engine_output.final_stress_voigt)


@pwf.as_macro_node(
    "relaxed_structure",
    "elastic_tensor",
    "elastic_constants",
)
def calculate_elastic_constants(
    wf,
    structure: Atoms,
    engine,
    relax_initial: bool = True,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
    fmax: float = 1e-3,
    max_iterations: int = 300,
):
    """Full MP-style elastic-constants workflow.

    relax_initial: full cell+ion relax of the input before deforming.
    Deformations are MP-standard; each deformed cell has its ions relaxed at
    fixed cell. Returns the relaxed reference structure, the fitted pymatgen
    ElasticTensor, and a flat dict of all MP elastic constants.
    """
    from pyiron_workflow_atomistics.engine import CalcInputMinimize, calculate
    from pyiron_workflow_atomistics.physics.bulk import evaluate_structures

    fixed_cell = CalcInputMinimize(
        relax_cell=False,
        force_convergence_tolerance=fmax,
        max_iterations=max_iterations,
    )

    if relax_initial:
        full_relax = CalcInputMinimize(
            relax_cell=True,
            force_convergence_tolerance=fmax,
            max_iterations=max_iterations,
        )
        wf.relax_engine = with_calc_input_node(engine, full_relax)
        wf.relax = calculate(structure=structure, engine=wf.relax_engine)
        ref_structure = wf.relax.outputs.engine_output.final_structure
        wf.eq_stress = _reference_stress_gpa(wf.relax.outputs.engine_output)
        eq_stress = wf.eq_stress
    else:
        ref_structure = structure
        eq_stress = None

    wf.deform = generate_mp_deformations(
        ref_structure, norm_strains=norm_strains, shear_strains=shear_strains
    )
    wf.deform_engine = with_calc_input_node(engine, fixed_cell)
    wf.evals = evaluate_structures(
        structures=wf.deform.outputs.deformed_structures,
        engine=wf.deform_engine,
    )
    wf.stresses = extract_stresses_gpa(wf.evals.outputs.engine_output_lst)
    wf.fit = fit_elastic_tensor(
        strains=wf.deform.outputs.strains,
        stresses=wf.stresses,
        structure=ref_structure,
        eq_stress=eq_stress,
    )
    wf.summary = elastic_constants_summary(wf.fit, ref_structure)

    return ref_structure, wf.fit, wf.summary
