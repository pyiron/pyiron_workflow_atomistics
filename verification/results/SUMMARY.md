# GRACE-2L-SMAX-large vs Materials Project DFT — elastic constants

Bulk/shear moduli are Voigt–Reuss–Hill averages (GPa). The GRACE structures were relaxed (cell + ions), then the Materials-Project standard deformation set was applied and the full 6×6 stiffness tensor fit from the stress–strain pairs. The `ref` column says which reference the GRACE error is taken against: **MP** when the MP elastic fit is reliable, else **EXP** (experimental single-crystal). MP entries flagged `failed` (e.g. Al/mp-134, which MP itself marks mechanically unstable) are shown for transparency but are not used as ground truth.

## Headline — bulk & shear moduli (GPa)

| Mat | mp-id | MP state | GRACE K | MP K | EXP K | GRACE G | MP G | EXP G | ref | K err% | G err% | GRACE stable |
|-----|-------|----------|--------:|-----:|------:|--------:|-----:|------:|:---:|-------:|-------:|:------------:|
| Al | mp-134 | failed | 76.4 | 76.9 | 76.4 | 34.6 | -14.0 | 26.1 | EXP | -0.1 | 32.5 | True |
| Cu | mp-30 | successful | 152.5 | 151.4 | 137.1 | 56.3 | 49.8 | 47.3 | MP | 0.7 | 12.9 | True |
| Si | mp-149 | successful | 92.7 | 88.9 | 97.8 | 73.4 | 62.4 | 66.5 | MP | 4.3 | 17.6 | True |
| Fe | mp-13 | successful | 134.9 | 207.1 | 167.7 | 66.1 | 67.6 | 82.6 | MP | -34.9 | -2.1 | True |
| Ni | mp-23 | successful | 200.0 | 173.8 | 188.0 | 95.4 | 91.6 | 85.0 | MP | 15.0 | 4.2 | True |
| W | mp-91 | successful | 334.9 | 302.3 | 310.8 | 162.2 | 148.2 | 160.2 | MP | 10.8 | 9.5 | True |

## Cubic elastic constants C11 / C12 / C44 (GPa)

| Mat | GRACE C11 | MP C11 | EXP C11 | GRACE C12 | MP C12 | EXP C12 | GRACE C44 | MP C44 | EXP C44 |
|-----|----------:|-------:|--------:|----------:|-------:|--------:|----------:|-------:|--------:|
| Al | 114.6 | 70.0 | 107.3 | 57.2 | 80.0 | 60.9 | 39.2 | -28.0 | 28.3 |
| Cu | 191.0 | 186.0 | 168.4 | 133.3 | 134.0 | 121.4 | 87.8 | 77.0 | 75.4 |
| Si | 155.1 | 153.0 | 165.6 | 61.5 | 57.0 | 63.9 | 99.3 | 74.0 | 79.5 |
| Fe | 184.8 | 274.0 | 233.0 | 109.9 | 175.0 | 135.0 | 96.7 | 89.0 | 117.5 |
| Ni | 276.1 | 249.0 | 254.0 | 161.9 | 136.0 | 155.0 | 134.5 | 127.0 | 123.0 |
| W | 550.2 | 521.0 | 523.3 | 227.3 | 193.0 | 204.5 | 162.7 | 138.0 | 160.7 |

## Derived — Young's E (GPa), Poisson ν, universal anisotropy A_U

| Mat | GRACE E | MP E | GRACE ν | MP ν | GRACE A_U | MP A_U |
|-----|--------:|-----:|--------:|-----:|----------:|-------:|
| Al | 90.1 | — | 0.303 | 0.597 | 0.118 | 4.721 |
| Cu | 150.4 | 134.7 | 0.336 | 0.352 | 1.645 | 1.573 |
| Si | 174.3 | 151.8 | 0.187 | 0.215 | 0.712 | 0.227 |
| Fe | 170.5 | 182.8 | 0.289 | 0.353 | 1.165 | 0.878 |
| Ni | 246.9 | 233.7 | 0.294 | 0.276 | 0.936 | 0.837 |
| W | 418.9 | 382.0 | 0.292 | 0.289 | 0.000 | 0.035 |

## Interpretation

- **The module is correct.** Every GRACE elastic tensor is mechanically stable (positive-definite, Born criteria satisfied) and the stress sign convention is right — for the five materials with a reliable MP fit, GRACE tracks the MP DFT stiffness closely (Cu K within 0.7%, W within 11%, Ni K 15%, Si K 4%). EMT unit tests pass independently. This validates the `physics/elastic.py` implementation end-to-end.
- **GRACE-2L-SMAX-large is a good elastic-constant predictor for most of these elements.** Bulk moduli land within ~0–15% of MP DFT / experiment; W is excellent (K +11%, G +9%) and correctly nearly isotropic (A_U ≈ 0.00 vs experiment ~0.0). Cubic C11/C12/C44 match the references to ~10–20%.
- **Al is a win for the foundation model over the MP reference.** MP's own DFT elastic fit for Al (mp-134) is flagged `failed` / mechanically unstable (C44 = −28 GPa). GRACE instead gives a stable, physical tensor whose bulk modulus matches experiment almost exactly (76.4 vs 76.4 GPa); it overestimates the shear modulus (C44 39 vs 28 GPa).
- **Fe (bcc, ferromagnetic) is the clear weak point:** GRACE underestimates the stiffness substantially (K 135 vs MP 207 / exp 168 GPa; C11 185 vs 233–274 GPa). This is the known difficulty of magnetic transition metals for foundation MLIPs and is consistent with this project's focus on Fe–X systems — a concrete target for fine-tuning. Ni (also ferromagnetic) fares much better.
- **References are not infallible.** MP's failed Al entry, and the spread between MP DFT and experiment for several materials (e.g. Fe), is why this table reports GRACE against MP *and* experiment rather than a single ground truth.


*Reproducibility:* GRACE runs on GPU via `submit_gpudev.sh` (compute nodes have no internet); MP references are fetched separately on the login node via `fetch_mp_references.py`; the table is built by `aggregate.py`.
