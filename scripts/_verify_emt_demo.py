import time, json, os
from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from pyiron_workflow_atomistics.physics.melting import MeltingInput, calculate_melting_point

OUT = "/ptmp/hmai/pwa_melting/_verify_runs/emt_full"
os.makedirs(OUT, exist_ok=True)
mi = MeltingInput(
    element="Al", crystalstructure="fcc", a=4.05, n_atoms=500,
    temperature_left=0.0, temperature_right=1500.0, strain_run_steps=200,
    timestep_lst=[2.0], fit_range_lst=[0.05], nve_steps_lst=[300],
    nvt_run_steps=300, npt_run_steps=300, n_strain_points=7,
    ratio_boundary=0.3, boundary_value=0.25, seed=1,
)
eng = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=OUT)
t0 = time.time()
res = calculate_melting_point.node_function(eng, mi)
dt = time.time() - t0
d = res.to_dict()
# strip heavy iteration structures for the JSON summary
summary = {
    "element": d["element"], "crystalstructure": d["crystalstructure"],
    "n_atoms": d["n_atoms"], "initial_guess_K": d["initial_guess"],
    "melting_temperature_K": d["melting_temperature"], "converged": d["converged"],
    "n_iterations": d["n_iterations"], "wall_seconds": round(dt, 1),
    "iterations": [
        {"T_in": it.temperature_in, "T_next": it.temperature_next,
         "ratios": [round(r, 3) for r in it.ratios],
         "pressures_GPa": [round(p, 4) for p in it.pressures],
         "temperatures_K": [round(t, 1) for t in it.temperatures]}
        for it in d["iterations"]
    ],
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("EMT FULL-METHOD DEMO:")
print(json.dumps(summary, indent=2))
print("DONE_EMT_DEMO ->", os.path.join(OUT, "summary.json"))
