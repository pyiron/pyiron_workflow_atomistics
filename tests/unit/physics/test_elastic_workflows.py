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
    assert d["mechanically_stable"] is True, f"C eigenvalues {d['stiffness_eigenvalues']}"
    assert C[0, 0] > 0 and C[0, 0] > C[0, 1]
    assert C[3, 3] > 0
    assert d["K_VRH"] > 0 and d["G_VRH"] > 0
    # EMT Cu bulk modulus is ~ 130-180 GPa range
    assert 80 < d["K_VRH"] < 250
