import os, time, numpy as np
from ase.build import bulk
from pyiron_workflow_lammps.engine import LammpsEngine
from pyiron_workflow_atomistics.engine import CalcInputMD, calculate
from pyiron_workflow_atomistics.analysis.trajectory import (
    temperatures_from_trajectory, pressures_from_trajectory,
)

OUT = "/ptmp/hmai/pwa_melting/_verify_runs/lammps_grace_smoke"
os.makedirs(OUT, exist_ok=True)
LMP = "/ptmp/hmai/lammps/build/lmp"
MODEL = "/ptmp/hmai/grace_cache/GRACE-1L-OAM"

atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))  # 108 atoms
md = CalcInputMD(mode="NVT", thermostat="nose-hoover", temperature=600.0,
                 n_ionic_steps=20, n_print=2, time_step=2.0,
                 thermostat_time_constant=100.0, seed=1, initial_temperature=1200.0)
eng = LammpsEngine(
    EngineInput=md,
    working_directory=OUT,
    command=f"{LMP} -in in.lmp -log log.lammps",
    path_to_model=MODEL,
    input_script_pair_style="grace",
)
t0 = time.time()
out = calculate.node_function(atoms, engine=eng)
dt = time.time() - t0
print("converged:", out.converged)
nframes = None if out.structures is None else len(out.structures)
print("n trajectory frames:", nframes)
print("n stresses:", None if out.stresses is None else len(out.stresses))
if out.structures:
    v = out.structures[-1].get_velocities()
    print("final frame has nonzero velocities:", bool(np.abs(v).sum() > 0))
    print("final frame get_temperature (K):", round(out.structures[-1].get_temperature(), 1))
try:
    T = temperatures_from_trajectory.node_function(out, last_n=5)
    P = pressures_from_trajectory.node_function(out, last_n=5)
    print("mean T (K):", round(T, 1), "| mean P (GPa):", round(P, 4))
except Exception as e:
    print("analysis ERR:", repr(e))
print("wall_seconds:", round(dt, 1))
print("DONE_LAMMPS_SMOKE")
