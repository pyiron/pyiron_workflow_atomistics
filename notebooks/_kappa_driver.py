"""Single-calculator driver for `calculate_phonon_thermal_conductivity`.

Invoked as a subprocess by ``phonon_thermal_conductivity.ipynb`` so each
calculator (EMT / EAM / GRACE / MACE) runs in the conda env where its
Python deps live. Result is pickled to ``--out-pkl`` for the notebook to
load and plot alongside the others.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Quiet TF (matters only for GRACE) and persist its PTX cache
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_CACHE_PATH", str(Path.home() / ".nv" / "ComputeCache"))

# When invoked from a jupyter kernel, MPLBACKEND is set to
# 'module://matplotlib_inline.backend_inline'. matplotlib-inline is not
# installed in every env (notably the `mace` env), and mace-torch pulls
# matplotlib in transitively, so the import explodes before any compute.
# Force a headless backend in the subprocess.
os.environ["MPLBACKEND"] = "Agg"


def build_calculator(spec: dict):
    kind = spec["kind"]
    if kind == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    if kind == "eam":
        from ase.calculators.eam import EAM

        return EAM(potential=spec["potential"])
    if kind == "grace":
        from tensorpotential.calculator.foundation_models import grace_fm

        return grace_fm(spec["model"])
    if kind == "mace":
        from mace.calculators import mace_mp

        return mace_mp(
            model=spec.get("model", "small"),
            device=spec.get("device", "cpu"),
            default_dtype=spec.get("dtype", "float64"),
        )
    raise ValueError(f"unknown calculator kind: {kind!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--calc-spec", required=True, help="JSON-encoded calculator spec")
    p.add_argument("--out-pkl", required=True, type=Path)
    p.add_argument("--working-dir", required=True, type=Path)
    p.add_argument("--lattice-a", type=float, default=4.05)
    p.add_argument("--fc2-supercell", type=int, default=3, help="N for N*eye(3)")
    p.add_argument("--fc3-supercell", type=int, default=2)
    p.add_argument("--n-snapshots", type=int, default=100)
    p.add_argument("--q-mesh", type=int, default=11)
    p.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[100.0, 200.0, 300.0, 500.0, 700.0],
    )
    p.add_argument("--displacement-distance", type=float, default=0.03)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    import numpy as np
    from ase.build import bulk

    # Make the worktree root importable so that envs with an older pwa in
    # site-packages (e.g. the mace env at 0.0.4) pick up the local source.
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.phonons import (
        calculate_phonon_thermal_conductivity,
    )

    spec = json.loads(args.calc_spec)
    args.working_dir.mkdir(parents=True, exist_ok=True)

    structure = bulk("Al", "fcc", a=args.lattice_a, cubic=True)  # 4 atoms / cell
    calc = build_calculator(spec)

    # one-shot energy/force at the as-built lattice constant
    structure.calc = calc
    e0 = float(structure.get_potential_energy())
    f0_max = float(np.linalg.norm(structure.get_forces(), axis=1).max())
    structure.calc = None

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=calc,
        working_directory=str(args.working_dir),
        write_to_disk=False,
        properties=("energy", "forces"),
    )

    fc2_sc = (args.fc2_supercell * np.eye(3)).astype(int)
    fc3_sc = (args.fc3_supercell * np.eye(3)).astype(int)

    print(
        f"[{args.label}] start: e0={e0:.4f} eV, |F|max={f0_max:.2e} eV/A, "
        f"fc2={args.fc2_supercell}**3 ({len(structure) * args.fc2_supercell ** 3} atoms), "
        f"fc3={args.fc3_supercell}**3 ({len(structure) * args.fc3_supercell ** 3} atoms), "
        f"n_snap={args.n_snapshots}",
        flush=True,
    )

    t0 = time.time()
    wf = calculate_phonon_thermal_conductivity(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=fc2_sc,
        fc3_supercell_matrix=fc3_sc,
        temperatures=list(args.temperatures),
        q_mesh=tuple([args.q_mesh] * 3),
        displacement_distance=args.displacement_distance,
        number_of_snapshots=args.n_snapshots,
        random_seed=args.random_seed,
        fc_calculator="symfc",
        mode_resolved=True,
        harmonic_observables=True,
    )
    wf.run()
    out = wf.outputs.phonon_output.value
    dt = time.time() - t0

    # Pickle only what the notebook plots — keep the dump small and
    # cross-env-loadable (no live phono3py / TF / torch handles).
    payload = {
        "label": args.label,
        "spec": spec,
        "elapsed_s": dt,
        "energy_0": e0,
        "fmax_0": f0_max,
        "lattice_a": args.lattice_a,
        "n_atoms_primitive": len(structure),
        "fc2_supercell": args.fc2_supercell,
        "fc3_supercell": args.fc3_supercell,
        "n_snapshots": args.n_snapshots,
        "temperatures": np.asarray(out.temperatures),
        "kappa": np.asarray(out.kappa),
        "converged": bool(out.converged),
        "band_structure": out.band_structure,
        "dos": out.dos,
        "free_energy": out.free_energy,
    }
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(payload, f)
    kappa_diag_300 = None
    for T, K in zip(out.temperatures, out.kappa):
        if abs(T - 300.0) < 1e-9:
            kappa_diag_300 = np.diag(K).tolist()
    print(
        f"[{args.label}] done in {dt:.1f}s, "
        f"kappa_300_diag={kappa_diag_300} W/(m·K), wrote {args.out_pkl}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
