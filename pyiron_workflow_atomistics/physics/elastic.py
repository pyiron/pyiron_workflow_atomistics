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

    ``engine`` must be a dataclass (e.g. ``ASEEngine``); a clear ``TypeError``
    is raised otherwise.
    """
    if not dataclasses.is_dataclass(engine):
        raise TypeError(
            f"with_calc_input requires a dataclass engine (got {type(engine).__name__}); "
            "ASEEngine is a dataclass."
        )
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
    stresses = []
    for i, o in enumerate(engine_outputs):
        if o.final_stress_voigt is None:
            raise ValueError(
                f"Engine output {i} has final_stress_voigt=None; elastic constants "
                "require computed stresses (engine must produce a stress tensor)."
            )
        stresses.append(voigt_stress_to_gpa(o.final_stress_voigt))
    return stresses


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

    Returns
    -------
    dict
        Flat dictionary with the following keys (all moduli in GPa):

        - ``K_Voigt``, ``K_Reuss``, ``K_VRH`` : bulk modulus, Voigt / Reuss /
          Voigt-Reuss-Hill average (GPa).
        - ``G_Voigt``, ``G_Reuss``, ``G_VRH`` : shear modulus, Voigt / Reuss /
          Voigt-Reuss-Hill average (GPa).
        - ``youngs_modulus`` : Young's modulus E = 9KG/(3K+G) from the VRH
          K and G (GPa).
        - ``poisson_ratio`` : homogeneous (isotropic) Poisson ratio
          (dimensionless).
        - ``universal_anisotropy`` : universal elastic anisotropy index
          (dimensionless).
        - ``mechanically_stable`` : bool; True iff all stiffness eigenvalues
          exceed a small tolerance (general Born stability criterion).
        - ``stiffness_eigenvalues`` : the 6 eigenvalues of the Voigt stiffness
          matrix (GPa).
        - ``elastic_tensor_voigt`` : 6x6 stiffness tensor, Voigt notation (GPa).
        - ``elastic_tensor_ieee`` : 6x6 stiffness tensor rotated to IEEE
          standard axes (GPa); falls back to the unrotated tensor if the IEEE
          rotation is undefined for the cell.
        - ``compliance_tensor_voigt`` : 6x6 compliance tensor, Voigt notation
          (1/GPa).
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    et = elastic_tensor
    pmg = AseAtomsAdaptor.get_structure(structure)
    try:
        et_ieee = et.convert_to_ieee(pmg)
    except (ValueError, NotImplementedError):
        # IEEE rotation can be undefined for some low-symmetry / degenerate cells;
        # fall back to the unrotated tensor rather than failing the whole workflow.
        et_ieee = et

    C = np.asarray(et.voigt)
    # General Born stability: stiffness matrix positive-definite
    eigvals = np.linalg.eigvalsh(C)
    STABILITY_TOL = 1e-8  # GPa; guards against rounding-noise eigenvalues
    mech_stable = bool(np.all(eigvals > STABILITY_TOL))

    # Young's modulus from the VRH moduli (both already in GPa), so the result
    # is unit-consistent: E = 9KG / (3K + G).
    K_vrh = float(et.k_vrh)
    G_vrh = float(et.g_vrh)
    youngs = 9.0 * K_vrh * G_vrh / (3.0 * K_vrh + G_vrh)

    d = {
        "K_Voigt": float(et.k_voigt),
        "K_Reuss": float(et.k_reuss),
        "K_VRH": K_vrh,
        "G_Voigt": float(et.g_voigt),
        "G_Reuss": float(et.g_reuss),
        "G_VRH": G_vrh,
        "youngs_modulus": youngs,
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

    Relaxes the input (optionally), applies the Materials Project standard
    deformation set, relaxes each deformed cell at fixed cell, and fits the
    full 6x6 stiffness tensor from the resulting stress-strain pairs.

    Parameters
    ----------
    structure : ase.Atoms
        Reference structure to compute elastic constants for.
    engine
        A dataclass engine that *produces stresses* (e.g. ``ASEEngine``). The
        macro internally clones it via :func:`with_calc_input` to switch
        between full-relax and fixed-cell-relax modes, so a single engine is
        supplied. The underlying calculator must return a stress tensor; an
        engine that yields ``final_stress_voigt=None`` will fail in
        :func:`extract_stresses_gpa`.
    relax_initial : bool, default True
        If True, perform a full cell+ion relaxation of ``structure`` before
        deforming, and use its residual stress as ``eq_stress`` in the fit.
        If False, deform the input as-is with no reference stress.
    norm_strains, shear_strains : tuple of float
        Normal and shear strain magnitudes for the MP deformation set
        (6 modes x len(magnitudes) deformed cells).
    fmax : float, default 1e-3
        Force-convergence tolerance (eV/A) for every relaxation.
    max_iterations : int, default 300
        Maximum optimizer steps per relaxation.

    Returns
    -------
    relaxed_structure : ase.Atoms
        The relaxed reference structure (equal to ``structure`` when
        ``relax_initial`` is False).
    elastic_tensor : pymatgen ElasticTensor
        The fitted, symmetrized 6x6 stiffness tensor (GPa).
    elastic_constants : dict
        Flat dict of all MP elastic constants (see
        :func:`elastic_constants_summary` for the full key list). Carries the
        full Voigt stiffness, the IEEE-rotated stiffness, and the compliance
        tensor, plus all derived moduli.

    Notes
    -----
    - All returned moduli are in GPa.
    - Stress sign convention is ASE/pymatgen: tension-positive Cauchy stress.
    - Deformations are parameterized by the Green-Lagrange strain tensor.
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
