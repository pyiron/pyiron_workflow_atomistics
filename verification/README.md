# Elastic-constants verification: GRACE-2L-SMAX-large vs Materials Project DFT

This directory validates `pyiron_workflow_atomistics.physics.elastic` by computing the
full elastic tensor with the **GRACE-2L-SMAX-large** foundation MLIP (via the ASE engine)
for six cubic elements and cross-checking against **Materials Project DFT** elastic data
and **experimental single-crystal** constants.

See [`results/SUMMARY.md`](results/SUMMARY.md) for the comparison tables and interpretation.

## Materials

| Element | Crystal | MP id | MP elastic fit |
|---------|---------|-------|----------------|
| Al | fcc | mp-134 | **failed** (MP marks it mechanically unstable → experiment used as reference) |
| Cu | fcc | mp-30 | successful |
| Si | diamond | mp-149 | successful |
| Fe | bcc (ferromagnetic) | mp-13 | successful |
| Ni | fcc (ferromagnetic) | mp-23 | successful |
| W | bcc | mp-91 | successful |

## Method

For each material: relax (cell + ions) with GRACE, apply the MP-standard deformation set
(normal strains ±0.5%/±1%, shear ±3%/±6% → 24 cells), relax ions at fixed cell, fit the
6×6 stiffness tensor, and derive K, G (Voigt/Reuss/Hill), Young's E, Poisson ν, and the
universal anisotropy A_U. This is the same `calculate_elastic_constants` macro the unit
tests exercise with EMT.

## How to run

The GRACE compute runs on GPU; the MP reference fetch must run where there is internet.

```bash
# 1. GRACE elastic constants on 4x A100 (gpudev). Compute nodes have NO outbound
#    internet, so the MP call inside the job is expected to fail — that's fine.
sbatch verification/submit_gpudev.sh          # ~2.5 min wall for all six materials

# 2. Fetch MP references on the (internet-connected) login node and merge into the JSONs.
source /ptmp/hmai/.mp_api_key                 # exports MP_API_KEY (kept outside the repo, mode 600)
.venv/bin/python verification/fetch_mp_references.py

# 3. Build the comparison tables (results/SUMMARY.md + results/combined.csv).
.venv/bin/python verification/aggregate.py
```

Environment: `GRACE_CACHE=/ptmp/hmai/grace_cache` (GRACE-2L-SMAX-large is pre-cached there;
the installed `tensorpotential` allowlist predates the name, so the loader falls back to a
direct `TPCalculator` load of the cached saved_model — see `load_grace_calculator`).

## Files

| File | Role |
|------|------|
| `elastic_grace_vs_mp.py` | per-material GRACE elastic calc + MP fetch → `results/<MAT>.json` + `.csv` |
| `submit_gpudev.sh` | SLURM job: GPU-visibility check, then all six materials across 4 A100s |
| `fetch_mp_references.py` | login-node MP fetch (captures `state`/`warnings`), merged into the JSONs |
| `aggregate.py` | builds `results/SUMMARY.md` and `results/combined.csv` (GRACE vs MP vs EXP) |
| `results/` | per-material JSON/CSV, `combined.csv`, `SUMMARY.md` |

## Experimental references (single-crystal, ~298 K)

Adversarially verified (internal VRH consistency + independent corroboration). Values in GPa.

| Element | C11 | C12 | C44 | K_VRH | G_VRH | Primary source |
|---------|----:|----:|----:|------:|------:|----------------|
| Al | 107.3 | 60.9 | 28.3 | 76.4 | 26.1 | Simmons & Wang 1971 (Vallin 1964; Kamm & Alers 1964) |
| Cu | 168.4 | 121.4 | 75.4 | 137.1 | 47.3 | Simmons & Wang 1971; Ledbetter & Naimon, JPCRD 3, 897 (1974) |
| Si | 165.6 | 63.9 | 79.5 | 97.8 | 66.5 | Hall, Phys. Rev. 161, 756 (1967); McSkimin & Andreatch 1964 |
| Fe | 233.0 | 135.0 | 117.5 | 167.7 | 82.6 | Rayne & Chandrasekhar, Phys. Rev. 122, 1714 (1961); Ledbetter & Reed 1973 |
| Ni | 254.0 | 155.0 | 123.0 | 188.0 | 85.0 | Ledbetter & Reed, JPCRD 2, 531 (1973); Neighbours et al. 1952 |
| W | 523.3 | 204.5 | 160.7 | 310.8 | 160.2 | Featherston & Neighbours, Phys. Rev. 130, 1324 (1963) |

## Headline result

GRACE-2L-SMAX-large reproduces the elastic constants well for most elements (bulk moduli
within ~0–15% of MP DFT / experiment; W and Cu excellent; W correctly nearly isotropic),
gives a physical, stable Al tensor where MP's own DFT fit failed, and most notably
**underestimates ferromagnetic bcc Fe** (K 135 vs MP 207 / exp 168 GPa) — the expected
magnetic-transition-metal weakness and a concrete fine-tuning target for Fe–X work.
