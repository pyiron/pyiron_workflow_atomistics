import numpy as np
import pytest


@pytest.mark.slow
def test_calculate_elastic_constants_emt_cu(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    wf = calculate_elastic_constants(
        structure=structure,
        engine=engine,
        relax_initial=True,
    )
    out = wf.run()
    d = out["elastic_constants"]
    C = np.asarray(d["elastic_tensor_voigt"])
    # Physically sane cubic metal: positive-definite, C11>C12, C44>0, sign correct
    assert (
        d["mechanically_stable"] is True
    ), f"C eigenvalues {d['stiffness_eigenvalues']}"
    assert C[0, 0] > 0 and C[0, 0] > C[0, 1]
    assert C[3, 3] > 0
    assert d["K_VRH"] > 0 and d["G_VRH"] > 0
    # EMT Cu bulk modulus is ~ 130-180 GPa range
    assert 80 < d["K_VRH"] < 250


@pytest.mark.slow
def test_calculate_elastic_constants_cell_relaxes_input_independent(tmp_path):
    """Regression for the macro channel-leak: relaxing the SAME material from
    two different starting lattice constants must give the SAME relaxed volume
    and elastic constants.

    With the bug, ``fmax``/``max_iterations`` leaked into CalcInputMinimize as
    pyiron_workflow channel objects, ASE "converged" the reference relaxation at
    step 0, the cell was frozen at the (arbitrary) input, and K_VRH came out
    input-dependent (e.g. a=3.9 -> K~32, a=3.615 -> K~123, both frozen at the
    input volume). After the fix both starting cells relax to EMT Cu's
    equilibrium (V ~ 11.57 A^3/atom, K_VRH ~ 134.6, G_VRH ~ 60.1 GPa).
    """
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants

    def run(a, subdir):
        structure = bulk("Cu", "fcc", a=a, cubic=True)
        input_vol = structure.get_volume() / len(structure)
        engine = ASEEngine(
            EngineInput=CalcInputStatic(),
            calculator=EMT(),
            working_directory=str(tmp_path / subdir),
        )
        wf = calculate_elastic_constants(
            structure=structure, engine=engine, relax_initial=True
        )
        out = wf.run()
        relaxed = out["relaxed_structure"]
        relaxed_vol = relaxed.get_volume() / len(relaxed)
        return input_vol, relaxed_vol, out["elastic_constants"]

    in_a, vol_a, d_a = run(3.615, "near_eq")
    in_b, vol_b, d_b = run(3.9, "far")

    # Canary: the cell must actually move (not be frozen at the input volume).
    assert abs(vol_b - in_b) > 1e-3, "cell did not relax (frozen at input volume)"

    # Input independence: both start points converge to the same equilibrium.
    assert abs(vol_a - vol_b) < 1e-2, (vol_a, vol_b)
    np.testing.assert_allclose(d_a["K_VRH"], d_b["K_VRH"], rtol=2e-2)
    np.testing.assert_allclose(d_a["G_VRH"], d_b["G_VRH"], rtol=2e-2)

    # Sanity against the known EMT Cu equilibrium.
    np.testing.assert_allclose(vol_b, 11.565, rtol=2e-2)
    np.testing.assert_allclose(d_b["K_VRH"], 134.6, rtol=5e-2)
    np.testing.assert_allclose(d_b["G_VRH"], 60.1, rtol=5e-2)
