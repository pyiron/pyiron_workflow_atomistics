import numpy as np
from ase import Atoms
from pyiron_workflow_atomistics.physics.elastic import voigt_stress_to_gpa, EV_PER_A3_TO_GPA


def test_ev_per_a3_to_gpa_constant():
    assert abs(EV_PER_A3_TO_GPA - 160.21766208) < 1e-6


def test_voigt_stress_to_gpa_shape_and_units():
    # 1 eV/A^3 hydrostatic in Voigt -> 3x3 GPa tensor
    voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    tensor = voigt_stress_to_gpa(voigt)
    assert tensor.shape == (3, 3)
    np.testing.assert_allclose(np.diag(tensor), [160.21766208] * 3, rtol=1e-6)
    # off-diagonal shear placement: voigt[3]=yz, [4]=xz, [5]=xy
    voigt2 = np.array([0.0, 0.0, 0.0, 2.0, 3.0, 4.0])
    t2 = voigt_stress_to_gpa(voigt2)
    np.testing.assert_allclose(t2[1, 2] / EV_PER_A3_TO_GPA, 2.0)  # yz
    np.testing.assert_allclose(t2[0, 2] / EV_PER_A3_TO_GPA, 3.0)  # xz
    np.testing.assert_allclose(t2[0, 1] / EV_PER_A3_TO_GPA, 4.0)  # xy


def test_with_calc_input_swaps_engine_mode():
    from ase.calculators.emt import EMT
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic, CalcInputMinimize
    from pyiron_workflow_atomistics.physics.elastic import with_calc_input

    base = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=".")
    relaxed = with_calc_input(base, CalcInputMinimize(relax_cell=False, force_convergence_tolerance=1e-3))
    assert isinstance(relaxed.EngineInput, CalcInputMinimize)
    assert relaxed.EngineInput.relax_cell is False
    assert relaxed.EngineInput.force_convergence_tolerance == 1e-3
    # original is untouched (immutability via dataclasses.replace)
    assert isinstance(base.EngineInput, CalcInputStatic)
    # calculator is preserved
    assert relaxed.calculator is base.calculator


def test_generate_mp_deformations_count_and_pairing():
    from ase.build import bulk
    from pyiron_workflow_atomistics.physics.elastic import generate_mp_deformations

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    # multi-output node_function returns a tuple in output order (pyiron_workflow 0.15.6)
    structs, strains = generate_mp_deformations.node_function(atoms)
    # 6 strain modes x 4 magnitudes = 24
    assert len(structs) == 24
    assert len(strains) == 24
    assert all(isinstance(s, Atoms) for s in structs)
    # strains are 3x3 arrays
    assert np.asarray(strains[0]).shape == (3, 3)
    # deformed cells actually differ from the reference cell
    ref = atoms.cell.array
    assert any(not np.allclose(s.cell.array, ref) for s in structs)


def test_extract_stresses_gpa_from_engine_outputs():
    from types import SimpleNamespace
    from pyiron_workflow_atomistics.physics.elastic import extract_stresses_gpa, EV_PER_A3_TO_GPA

    o1 = SimpleNamespace(final_stress_voigt=np.array([1.0, 0, 0, 0, 0, 0]))
    o2 = SimpleNamespace(final_stress_voigt=np.array([0, 0, 0, 0, 0, 0.5]))
    out = extract_stresses_gpa.node_function([o1, o2])
    stresses = out  # node returns single output "stresses"
    assert len(stresses) == 2
    np.testing.assert_allclose(stresses[0][0, 0], 1.0 * EV_PER_A3_TO_GPA)
    np.testing.assert_allclose(stresses[1][0, 1], 0.5 * EV_PER_A3_TO_GPA)


def test_fit_elastic_tensor_recovers_known_cubic():
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet
    from pyiron_workflow_atomistics.physics.elastic import fit_elastic_tensor

    # Known cubic stiffness (GPa)
    C11, C12, C44 = 200.0, 130.0, 100.0
    voigt = np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44],
    ])
    C_true = ElasticTensor.from_voigt(voigt)

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    pmg = AseAtomsAdaptor.get_structure(atoms)
    dss = DeformedStructureSet(pmg)
    strains = [np.asarray(d.green_lagrange_strain) for d in dss.deformations]
    # synthesize stresses (GPa) from the known tensor for each strain
    stresses = [np.asarray(C_true.calculate_stress(s)) for s in strains]

    out = fit_elastic_tensor.node_function(strains=strains, stresses=stresses, structure=atoms)
    C_fit = out  # single output "elastic_tensor"
    np.testing.assert_allclose(C_fit.voigt, voigt, atol=1.0)  # within 1 GPa


def test_elastic_constants_summary_known_cubic():
    from ase.build import bulk
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pyiron_workflow_atomistics.physics.elastic import elastic_constants_summary

    C11, C12, C44 = 200.0, 130.0, 100.0
    voigt = np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44],
    ])
    et = ElasticTensor.from_voigt(voigt)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    d = elastic_constants_summary.node_function(et, atoms)

    # Cubic Voigt bulk modulus K_V = (C11 + 2 C12)/3
    np.testing.assert_allclose(d["K_VRH"], (C11 + 2 * C12) / 3.0, rtol=1e-6)
    assert "K_Voigt" in d and "K_Reuss" in d
    assert "G_Voigt" in d and "G_Reuss" in d and "G_VRH" in d
    assert "youngs_modulus" in d and "poisson_ratio" in d
    assert "universal_anisotropy" in d
    assert d["mechanically_stable"] is True
    # full tensors present
    assert np.asarray(d["elastic_tensor_voigt"]).shape == (6, 6)
    assert np.asarray(d["compliance_tensor_voigt"]).shape == (6, 6)
    assert np.asarray(d["elastic_tensor_ieee"]).shape == (6, 6)

    K, G = d["K_VRH"], d["G_VRH"]
    expected_E = 9 * K * G / (3 * K + G)
    np.testing.assert_allclose(d["youngs_modulus"], expected_E, rtol=1e-3)


def test_public_import_surface():
    from pyiron_workflow_atomistics.physics.elastic import (
        calculate_elastic_constants,
        generate_mp_deformations,
        extract_stresses_gpa,
        fit_elastic_tensor,
        elastic_constants_summary,
        voigt_stress_to_gpa,
        with_calc_input,
    )
    assert all(callable(x) for x in (
        calculate_elastic_constants, generate_mp_deformations,
        extract_stresses_gpa, fit_elastic_tensor,
        elastic_constants_summary, voigt_stress_to_gpa, with_calc_input,
    ))
