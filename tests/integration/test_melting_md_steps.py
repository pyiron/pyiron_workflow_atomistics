import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting.md_steps import (
    build_solid_liquid_interface,
    npt_relax_solid,
    strain_scan_nvt_nve,
)


def _engine(tmp_path):
    return ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path)
    )


@pytest.mark.slow
def test_npt_relax_solid_returns_structure(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    struct, out = npt_relax_solid.node_function(
        s, _engine(tmp_path), temperature=300.0, n_steps=20, timestep=2.0, seed=1
    )
    assert len(struct) == len(s)
    assert out.converged is True


@pytest.mark.slow
def test_build_interface_runs(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 6))
    iface = build_solid_liquid_interface.node_function(
        s, _engine(tmp_path), t_solid=300.0, t_liquid=2000.0, n_steps=20, timestep=2.0,
        seed=1,
    )
    assert len(iface) == len(s)
    assert len(iface.constraints) == 0


@pytest.mark.slow
def test_strain_scan_returns_records(tmp_path):
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 6))
    recs = strain_scan_nvt_nve.node_function(
        s, _engine(tmp_path), temperature=300.0, strains=[0.99, 1.01],
        crystalstructure="fcc", nvt_steps=20, nve_steps=20, timestep=2.0, seed=1,
    )
    assert len(recs) == 2
    for r in recs:
        assert set(r) >= {
            "strain", "mean_T", "mean_P", "solid_fraction", "voronoi_max", "voronoi_mean",
        }
