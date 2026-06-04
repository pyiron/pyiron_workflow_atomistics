"""GRACE-2L-SMAX-large elastic constants vs Materials Project DFT.

For ONE material: relax + compute the full elastic tensor with the GRACE-2L-SMAX-large
foundation model (via the ASE engine), then pull the Materials Project elasticity doc
and record it WITH its quality flags (``state`` / ``warnings``) so failed MP fits
(e.g. Al / mp-134, whose MP elastic tensor is flagged mechanically unstable) are not
silently trusted. Writes per-material JSON + CSV.

The GRACE computation needs no reference, so this runs on GPU first; ``aggregate.py``
later combines GRACE + MP + experimental references into the final table.

Usage:
    GRACE_CACHE=/ptmp/hmai/grace_cache MP_API_KEY=... TF_FORCE_GPU_ALLOW_GROWTH=true \
      /ptmp/hmai/pwa_elastic/.venv/bin/python verification/elastic_grace_vs_mp.py \
      --material Cu --out-dir verification/results
"""
import argparse
import csv
import json
import os
import time
import traceback

import numpy as np

# symbol -> ASE bulk build args + MP material id
MATERIALS = {
    "Al": dict(name="Al", crystalstructure="fcc", a=4.05, mp_id="mp-134"),
    "Cu": dict(name="Cu", crystalstructure="fcc", a=3.615, mp_id="mp-30"),
    "Si": dict(name="Si", crystalstructure="diamond", a=5.43, mp_id="mp-149"),
    "Fe": dict(name="Fe", crystalstructure="bcc", a=2.87, mp_id="mp-13"),
    "Ni": dict(name="Ni", crystalstructure="fcc", a=3.52, mp_id="mp-23"),
    "W": dict(name="W", crystalstructure="bcc", a=3.16, mp_id="mp-91"),
}

GRACE_MODEL = "GRACE-2L-SMAX-large"


def build_structure(spec):
    from ase.build import bulk

    return bulk(spec["name"], spec["crystalstructure"], a=spec["a"], cubic=True)


def load_grace_calculator(model_name=GRACE_MODEL):
    """Load the GRACE foundation calculator.

    Prefers the canonical ``grace_fm`` entry point. The installed
    ``tensorpotential`` build ships a hardcoded ``MODELS_NAME_LIST`` allowlist
    that predates GRACE-2L-SMAX-large, so ``grace_fm`` rejects the name even
    though the saved_model is present in ``$GRACE_CACHE``. In that case we load
    the cached saved_model directly via ``TPCalculator`` (the same object
    grace_fm returns) using the path grace_fm itself would resolve
    (``$GRACE_CACHE/<model_name>``). Metadata/version gap, not a physics hack.
    """
    from tensorpotential.calculator import grace_fm

    try:
        return grace_fm(model_name)
    except AssertionError:
        from tensorpotential.calculator.asecalculator import TPCalculator

        cache_dir = os.environ.get("GRACE_CACHE") or os.path.expanduser("~/.cache/grace")
        model_path = os.path.join(cache_dir, model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"{model_name} not in tensorpotential allowlist and not cached "
                f"at {model_path}; set GRACE_CACHE to the cache root."
            )
        print(f"Using cached GRACE model from {model_path} (direct TPCalculator load)")
        return TPCalculator(model=model_path)


def grace_elastic(structure, workdir):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants

    calc = load_grace_calculator()
    engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=calc,
        working_directory=os.path.abspath(workdir),
    )
    wf = calculate_elastic_constants(structure=structure, engine=engine, relax_initial=True)
    out = wf.run()
    return out["elastic_constants"]


def _vrh(x):
    """VRH value from an MP modulus field (dict, pydantic object, or scalar)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        return x.get("vrh")
    return getattr(x, "vrh", None)


def mp_reference(mp_id, api_key):
    """MP elasticity doc WITH quality flags. ``reliable`` is True only when the
    MP fit succeeded and carries no warnings (so Al/mp-134's failed fit is flagged)."""
    from mp_api.client import MPRester

    with MPRester(api_key) as m:
        docs = m.materials.elasticity.search(
            material_ids=[mp_id],
            fields=[
                "material_id", "formula_pretty", "state", "warnings",
                "bulk_modulus", "shear_modulus", "youngs_modulus",
                "homogeneous_poisson", "universal_anisotropy", "elastic_tensor",
            ],
        )
    if not docs:
        return {"available": False, "reason": f"no elasticity doc for {mp_id}"}
    d = docs[0]
    state = str(getattr(d, "state", None))
    warnings = list(getattr(d, "warnings", None) or [])
    C = np.asarray(d.elastic_tensor.ieee_format) if getattr(d, "elastic_tensor", None) else None
    ym = getattr(d, "youngs_modulus", None)
    if ym is None:
        ym = getattr(d, "young_modulus", None)
    return {
        "available": True,
        "mp_id": mp_id,
        "state": state,
        "reliable": (state == "successful" and len(warnings) == 0),
        "n_warnings": len(warnings),
        "warnings": warnings,
        "K_VRH": _vrh(d.bulk_modulus),
        "G_VRH": _vrh(d.shear_modulus),
        "youngs_modulus": float(ym) if isinstance(ym, (int, float)) else _vrh(ym),
        "poisson_ratio": getattr(d, "homogeneous_poisson", None),
        "universal_anisotropy": getattr(d, "universal_anisotropy", None),
        "C11": float(C[0, 0]) if C is not None else None,
        "C12": float(C[0, 1]) if C is not None else None,
        "C44": float(C[3, 3]) if C is not None else None,
        "elastic_tensor_ieee": C.tolist() if C is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True, choices=list(MATERIALS))
    ap.add_argument("--out-dir", default="verification/results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spec = MATERIALS[args.material]
    record = {"material": args.material, "mp_id": spec["mp_id"]}

    t0 = time.time()
    structure = build_structure(spec)
    try:
        record["grace"] = grace_elastic(
            structure, os.path.join(args.out_dir, f"_work_{args.material}")
        )
        record["grace_ok"] = True
    except Exception:
        record["grace"] = None
        record["grace_ok"] = False
        record["grace_error"] = traceback.format_exc()
    record["grace_seconds"] = round(time.time() - t0, 1)

    try:
        record["mp"] = mp_reference(spec["mp_id"], os.environ["MP_API_KEY"])
    except Exception:
        record["mp"] = {"available": False, "reason": traceback.format_exc()}

    with open(os.path.join(args.out_dir, f"{args.material}.json"), "w") as f:
        json.dump(record, f, indent=2, default=float)

    g = record.get("grace") or {}
    mp = record.get("mp") or {}
    gC = np.asarray(g["elastic_tensor_ieee"]) if g.get("elastic_tensor_ieee") else None
    row = {
        "material": args.material, "mp_id": spec["mp_id"],
        "grace_ok": record["grace_ok"], "grace_seconds": record["grace_seconds"],
        "mp_state": mp.get("state"), "mp_reliable": mp.get("reliable"),
        "grace_stable": g.get("mechanically_stable"),
        "grace_K_VRH": g.get("K_VRH"), "mp_K_VRH": mp.get("K_VRH"),
        "grace_G_VRH": g.get("G_VRH"), "mp_G_VRH": mp.get("G_VRH"),
        "grace_E": g.get("youngs_modulus"), "mp_E": mp.get("youngs_modulus"),
        "grace_poisson": g.get("poisson_ratio"), "mp_poisson": mp.get("poisson_ratio"),
        "grace_A_U": g.get("universal_anisotropy"), "mp_A_U": mp.get("universal_anisotropy"),
        "grace_C11": gC[0, 0] if gC is not None else None, "mp_C11": mp.get("C11"),
        "grace_C12": gC[0, 1] if gC is not None else None, "mp_C12": mp.get("C12"),
        "grace_C44": gC[3, 3] if gC is not None else None, "mp_C44": mp.get("C44"),
    }
    with open(os.path.join(args.out_dir, f"{args.material}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2, default=float))


if __name__ == "__main__":
    main()
