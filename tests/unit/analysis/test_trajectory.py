import numpy as np
from ase import Atoms, units

from pyiron_workflow_atomistics.analysis.trajectory import (
    pressures_from_trajectory,
    temperatures_from_trajectory,
)
from pyiron_workflow_atomistics.engine.protocol import EngineOutput


def _frame_at_temperature(T_target, n=64, a=10.0, seed=0):
    rng = np.random.RandomState(seed)
    atoms = Atoms(f"Ar{n}", positions=rng.rand(n, 3) * a, cell=[a, a, a], pbc=True)
    atoms.set_momenta(rng.standard_normal((n, 3)))
    atoms.set_momenta(atoms.get_momenta() * np.sqrt(T_target / atoms.get_temperature()))
    return atoms


def test_temperatures_from_trajectory_mean():
    frames = [_frame_at_temperature(300.0, seed=i) for i in range(5)]
    out = EngineOutput(
        final_structure=frames[-1], final_energy=0.0, converged=True, structures=frames
    )
    T = temperatures_from_trajectory.node_function(out, last_n=5)
    assert abs(T - 300.0) < 1e-6


def test_pressures_from_trajectory_virial_plus_kinetic():
    n, a = 64, 10.0
    V = a**3
    frame = _frame_at_temperature(300.0, n=n, a=a, seed=1)
    p_vir = 0.01  # eV/A^3
    svoigt = np.array([-p_vir, -p_vir, -p_vir, 0.0, 0.0, 0.0])
    out = EngineOutput(
        final_structure=frame,
        final_energy=0.0,
        converged=True,
        structures=[frame],
        stresses=[svoigt],
    )
    P = pressures_from_trajectory.node_function(out, last_n=1)
    p_kin = n * units.kB * 300.0 / V  # eV/A^3
    expected = (p_vir + p_kin) * 160.21766208  # GPa
    assert abs(P - expected) < 1e-6


def test_pressures_accepts_full_3x3_stress():
    n, a = 64, 10.0
    frame = _frame_at_temperature(300.0, n=n, a=a, seed=2)
    p_vir = 0.02
    full = np.diag([-p_vir, -p_vir, -p_vir])
    out = EngineOutput(
        final_structure=frame,
        final_energy=0.0,
        converged=True,
        structures=[frame],
        stresses=[full],
    )
    P = pressures_from_trajectory.node_function(out, last_n=1)
    p_kin = n * units.kB * 300.0 / (a**3)
    assert abs(P - (p_vir + p_kin) * 160.21766208) < 1e-6
