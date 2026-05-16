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
    p.add_argument(
        "--eos-strain-range",
        type=float,
        nargs=2,
        default=(-0.02, 0.02),
        metavar=("LOW", "HIGH"),
    )
    p.add_argument("--eos-num-points", type=int, default=11)
    args = p.parse_args()

    import numpy as np
    from ase.build import bulk

    # Make the worktree root importable so that envs with an older pwa in
    # site-packages (e.g. the mace env at 0.0.4) pick up the local source.
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from pyiron_workflow import Workflow

    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.bulk import (
        optimise_cubic_lattice_parameter,
    )
    from pyiron_workflow_atomistics.physics.phonons import (
        calculate_phonon_thermal_conductivity,
    )

    spec = json.loads(args.calc_spec)
    args.working_dir.mkdir(parents=True, exist_ok=True)

    # Initial guess; each potential will optimise its own a0 via EOS scan.
    structure = bulk("Al", "fcc", a=args.lattice_a, cubic=True)  # 4 atoms / cell
    calc = build_calculator(spec)

    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=calc,
        working_directory=str(args.working_dir),
        write_to_disk=False,
        properties=("energy", "forces", "volume"),
    )

    fc2_sc = (args.fc2_supercell * np.eye(3)).astype(int)
    fc3_sc = (args.fc3_supercell * np.eye(3)).astype(int)

    # Build a single Workflow that wires EOS-optimised structure straight
    # into the phonon macro. Channels are connected via wf.opt.outputs.*,
    # so wf.run() executes EOS first, then the phonon macro on the relaxed
    # cell, in one graph rather than two manual calls.
    print(
        f"[{args.label}] start: "
        f"fc2={args.fc2_supercell}**3 ({len(structure) * args.fc2_supercell ** 3} atoms), "
        f"fc3={args.fc3_supercell}**3 ({len(structure) * args.fc3_supercell ** 3} atoms), "
        f"n_snap={args.n_snapshots}, initial a={args.lattice_a:.4f} Å",
        flush=True,
    )
    t_total = time.time()
    wf = Workflow(f"al_kappa_{spec['kind']}")
    wf.opt = optimise_cubic_lattice_parameter(
        structure=structure,
        name="Al",
        crystalstructure="fcc",
        engine=engine.with_working_directory("eos"),
        strain_range=tuple(args.eos_strain_range),
        num_points=args.eos_num_points,
        eos_type="birchmurnaghan",
    )
    wf.phonon = calculate_phonon_thermal_conductivity(
        structure=wf.opt.outputs.equil_struct,
        engine=engine.with_working_directory("phonon"),
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
    dt = time.time() - t_total

    a0 = float(wf.opt.outputs.a0.value)
    bulk_modulus_GPa = float(wf.opt.outputs.B.value)
    e0_per_atom = float(wf.opt.outputs.equil_energy_per_atom.value)
    v0_per_atom = float(wf.opt.outputs.equil_volume_per_atom.value)
    eos_volumes = [float(v) for v in wf.opt.outputs.volumes.value]
    eos_energies = [float(e) for e in wf.opt.outputs.energies.value]
    structure_relaxed = wf.opt.outputs.equil_struct.value
    out = wf.phonon.outputs.phonon_output.value

    # Cheap sanity check at the relaxed lattice (not part of the workflow graph).
    structure_relaxed.calc = calc
    e0 = float(structure_relaxed.get_potential_energy())
    f0_max = float(np.linalg.norm(structure_relaxed.get_forces(), axis=1).max())
    structure_relaxed.calc = None
    dt_eos = 0.0  # subsumed by single wf.run() — kept in payload for back-compat

    # Pickle only what the notebook plots — keep the dump small and
    # cross-env-loadable (no live phono3py / TF / torch handles).
    payload = {
        "label": args.label,
        "spec": spec,
        "elapsed_eos_s": dt_eos,
        "elapsed_phonon_s": dt,
        "energy_0": e0,
        "fmax_0": f0_max,
        "lattice_a_initial": args.lattice_a,
        "lattice_a0": a0,
        "bulk_modulus_GPa": bulk_modulus_GPa,
        "energy_per_atom": e0_per_atom,
        "volume_per_atom": v0_per_atom,
        "eos_volumes": eos_volumes,
        "eos_energies": eos_energies,
        "n_atoms_primitive": len(structure_relaxed),
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
        f"[{args.label}] workflow done in {dt:.1f}s: "
        f"a0={a0:.4f} Å, B={bulk_modulus_GPa:.1f} GPa, "
        f"kappa_300_diag={kappa_diag_300} W/(m·K), wrote {args.out_pkl}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
