"""V4: one interface-method coexistence iteration through the LAMMPS engine
(pair_style grace). Demonstrates the engine-agnostic melting algorithm running
unchanged on LAMMPS. Run on a GPU node (module load cuda)."""

import json
import os
import time

from ase.build import bulk

from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_atomistics.physics.melting.coexistence import coexistence_iteration
from pyiron_workflow_lammps.engine import LammpsEngine

OUT = "/ptmp/hmai/pwa_melting/_verify_runs/lammps_grace_coex"
os.makedirs(OUT, exist_ok=True)
LMP = "/ptmp/hmai/lammps/build/lmp"
MODEL = "/ptmp/hmai/grace_cache/GRACE-1L-OAM"

atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 6))  # 216 atoms
eng = LammpsEngine(
    EngineInput=CalcInputStatic(),
    working_directory=OUT,
    command=f"{LMP} -in in.lmp -log log.lammps",
    path_to_model=MODEL,
    input_script_pair_style="grace",
)
t0 = time.time()
rec = coexistence_iteration.node_function(
    atoms, eng, temperature=900.0, crystalstructure="fcc", fit_range=0.05,
    n_strain_points=5, nvt_steps=200, nve_steps=200, npt_steps=200, timestep=2.0,
    delta_t_melt=1000.0, ratio_boundary=0.4, boundary_value=0.25, seed=1,
    npt_thermostat="nose-hoover", subdir="coex",
)
out = {
    "engine": "LammpsEngine(pair_style grace, GRACE-1L-OAM)",
    "n_atoms": len(atoms),
    "temperature_in_K": rec.temperature_in,
    "temperature_next_K": rec.temperature_next,
    "strains": rec.strains,
    "solid_fraction": [round(r, 3) for r in rec.ratios],
    "pressures_GPa": [round(p, 4) for p in rec.pressures],
    "temperatures_K": [round(t, 1) for t in rec.temperatures],
    "wall_seconds": round(time.time() - t0, 1),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("DONE_LAMMPS_COEX ->", os.path.join(OUT, "summary.json"))
