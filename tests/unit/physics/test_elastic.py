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
