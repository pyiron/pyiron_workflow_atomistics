"""V3: Step-1 melting-temperature estimate through the ASE engine with a GRACE
foundation-model calculator. Run on a GPU node."""

import json
import os
import time

os.environ.setdefault("GRACE_CACHE", "/ptmp/hmai/grace_cache")

from ase.build import bulk  # noqa: E402

from pyiron_workflow_atomistics.analysis.structure_descriptors import (  # noqa: E402
    analyse_reference_structure,
)
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic  # noqa: E402
from pyiron_workflow_atomistics.physics.melting.initial_guess import (  # noqa: E402
    estimate_melting_temperature,
)
from tensorpotential.calculator import grace_fm  # noqa: E402

OUT = "/ptmp/hmai/pwa_melting/_verify_runs/grace_ase_step1"
os.makedirs(OUT, exist_ok=True)

calc = grace_fm("GRACE-1L-OAM")
atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))  # 108 atoms
key_max, _, half = analyse_reference_structure.node_function(atoms)
eng = ASEEngine(EngineInput=CalcInputStatic(), calculator=calc, working_directory=OUT)
t0 = time.time()
t_guess, struct = estimate_melting_temperature.node_function(
    atoms, eng, key_max=key_max, distribution_half=half, crystalstructure="fcc",
    temperature_left=0.0, temperature_right=1400.0, strain_run_steps=150,
    timestep=2.0, seed=1, t_step_min=50.0, max_iterations=10,
    npt_thermostat="berendsen",
)
out = {
    "engine": "ASEEngine(grace_fm GRACE-1L-OAM)",
    "n_atoms": len(atoms),
    "t_guess_K": t_guess,
    "wall_seconds": round(time.time() - t0, 1),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("DONE_GRACE_ASE ->", os.path.join(OUT, "summary.json"))
