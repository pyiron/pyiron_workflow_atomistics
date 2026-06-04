"""Fetch Materials Project elastic references and merge into the per-material JSONs.

Run this on an INTERNET-CONNECTED node (the Raven login node) — the GPU compute
nodes have no outbound network, so elastic_grace_vs_mp.py cannot reach the MP API
from inside the SLURM job. This patches the ``mp`` field of each
verification/results/<MAT>.json (leaving the GRACE results untouched).

    source /ptmp/hmai/.mp_api_key
    /ptmp/hmai/pwa_elastic/.venv/bin/python verification/fetch_mp_references.py
"""
import json
import os

from elastic_grace_vs_mp import MATERIALS, mp_reference  # local import (no TF at module load)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    key = os.environ["MP_API_KEY"]
    for mat, spec in MATERIALS.items():
        p = os.path.join(RESULTS_DIR, f"{mat}.json")
        if not os.path.exists(p):
            print(f"{mat}: no results JSON yet, skipping")
            continue
        with open(p) as f:
            rec = json.load(f)
        rec["mp"] = mp_reference(spec["mp_id"], key)
        with open(p, "w") as f:
            json.dump(rec, f, indent=2, default=float)
        mp = rec["mp"]
        print(f"{mat:3} {spec['mp_id']:7} state={mp.get('state')} "
              f"reliable={mp.get('reliable')} K_VRH={mp.get('K_VRH')} "
              f"G_VRH={mp.get('G_VRH')} nwarn={mp.get('n_warnings')}")


if __name__ == "__main__":
    main()
