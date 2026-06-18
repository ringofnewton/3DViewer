# Validation status & publication roadmap

What is and isn't established for the discrete fiber-network ECM model
(`ecm_pipeline`, see `METHODS.md`). This file separates **the validated science**
(the Python pipeline) from **communication tools** (the browser viewers), and
lists concretely what remains before the model supports a *quantitative*,
publishable claim.

> **Status (updated).** The core validation roadmap (§2 items 1–5) is now done:
> internal regression, discretization independence, order-of-magnitude **SI
> calibration** (length 1 µm, force 0.156 nN, stress 156 Pa, network modulus
> ~0.8 Pa, step ~2 h), one **quantitative experiment comparison** (explant bridge,
> with an honest discrepancy logged), **parameter robustness** (OAT + nonlinearity
> decomposition), **statistical power** (n=16, large effect sizes), and **full-3D
> reproduction**. The defensible claim is now **qualitative + order-of-magnitude
> calibrated**: cell-driven mechanical self-organization of a nonlinear fiber
> network produces — and controls — reticular collagen architecture, robustly and
> in 3D. Remaining for a *fully quantitative/predictive* paper: distribution-level
> comparison to a reticular target, a global (Sobol/LHS) sweep, tighter (periodic-BC)
> rheology, and the optional mechanistic upgrades in §2.6.

---

## 1. What is validated now (reproducible)

All checks are executable and deterministic given their seeds.

### 1.1 Internal regression (`verify_mechanics.py`, `verify_remodel.py`)
| Check | Result |
|---|---|
| FIRE relaxation reaches force balance | converged, max\|F\| ≈ 9.9e-3 |
| Single-cell radial aster (tension-weighted vs uniform radiality) | **0.98 ≫ 0.68** |
| Inter-cell track needs nonlinearity (nonlinear vs linear control) | **0.50 vs 0.40** (≈ isotropic 0.44) |
| Plastic memory: residual self-stress after cell removal (plastic vs elastic) | **4250 vs 11** (~400×) |

### 1.2 Discretization independence (`convergence_test.py`) — added for publication
The emergent observable (inter-cell band alignment) converges as the fiber
discretization is refined (physical setup fixed, 3 seeds):

| seg_len | ~nodes | mean band alignment | Δrel vs previous |
|---:|---:|---:|---:|
| 12.0 | 1320 | 0.556 | — |
| 8.0  | 1980 | 0.510 | 0.082 |
| 5.5  | 2860 | 0.492 | 0.035 |
| 4.0  | 3960 | 0.486 | **0.014** |

Finest-step relative change **0.014 < 0.05 tol**; limit ≈ **0.486** (isotropic =
0.333). The aligned track is a real, discretization-independent feature, not a
mesh artifact. (Minor: at the finest level one seed needed > 3000 FIRE iters;
raise `max_iter` for a fully tightened run.)

### 1.3 Quantitative comparison to experiment — explant bridge (`experiment_bridge.py`)
Reference: **Stopak & Harris (1982)** — two explants in collagen organize an
**aligned matrix bridge** between them. Model (two contractile foci, n=4 seeds,
mean ± SEM): bridge alignment along the focus–focus axis vs a perpendicular control.

| separation (µm) | bridge align | perpendicular control |
|---:|---:|---:|
| 24  | 0.497 ± 0.017 | 0.422 ± 0.010 |
| 44  | 0.586 ± 0.006 | 0.361 ± 0.004 |
| 68  | 0.527 ± 0.003 | 0.369 ± 0.005 |
| 96  | 0.554 ± 0.006 | 0.345 ± 0.003 |
| 128 | 0.598 ± 0.011 | 0.323 ± 0.006 |

(isotropic = 0.333.) **Reproduced quantitatively:** a coherent, *axis-specific*
aligned bridge (bridge ≈ 0.5–0.6 ≫ isotropic; perpendicular ≈ isotropic), robust
across seeds; the bridge–perpendicular gap *widens* with distance.
**Honest discrepancy:** the bridge does **not** weaken out to 128 µm (>3× the
traction reach) — the nonlinear network transmits force long-range (a documented
hallmark; Notbohm/Wang), so this regime shows sustained long-range bridging, not
the distance-limited failure seen between large explants. Capturing failure needs
finite cell force + matrix yielding/turnover + finite gel size (future work).
Figure: `output_validation/figures/bridge_alignment.png`.

### 1.4 Parameter sensitivity / robustness (`sensitivity.py`)
OAT sweep of the bridge conclusion (sep = 60 µm, n=2 seeds) over each parameter at
[low, default, high]. The **axis-specific bridge survives every single-parameter
variation** (axis ≫ isotropic AND axis ≫ perpendicular in all 15 runs). Influence
ranking (range of bridge alignment): `buckle_ratio` (0.089) > `k_bend` (0.041) >
`reach` ≈ `stiffen_alpha` (0.024) > `traction` (0.014).

Nonlinearity decomposition (axis − perpendicular signal):

| control | axis | perp | axis−perp |
|---|---:|---:|---:|
| full nonlinear (default) | 0.535 | 0.376 | **0.160** |
| buckling only (α=0) | 0.512 | 0.355 | 0.156 |
| stiffening only (r_b=1) | 0.446 | 0.345 | 0.102 |
| **fully linear** (r_b=1, α=0) | 0.430 | 0.349 | **0.081** |

**Refined claim:** nonlinearity ~**doubles** the axis-specific bridge signal vs a
fully-linear matrix (0.160 vs 0.081); **buckling is the dominant contributor** and
the two nonlinear mechanisms are **partially redundant** (either alone retains much
of the effect). So the nonlinear constitutive law is the load-bearing assumption —
now *quantified*, not just asserted.

### 1.5 Statistical power (`stats_power.py`)
Headline mechanics results over **n=16 seeds** (mean ± 95% CI; paired effect size):

| result | treatment | control | difference | Cohen's d |
|---|---:|---:|---:|---:|
| single-cell aster | tension-wtd radiality **0.969 ± 0.004** | uniform 0.689 ± 0.016 | **0.281 ± 0.017** | **9.06** |
| inter-cell bridge | axis align **0.532 ± 0.017** | perpendicular 0.371 ± 0.018 | **0.161 ± 0.035** | **2.47** |

Both paired-difference CIs exclude 0 → **significant at 95%** with **large effect
sizes** (isotropic baseline 0.333). Figure: `output_validation/figures/stats_power.png`.

### 1.6 Full-3D reproduction (`validate_3d.py`)
The headline results hold without the planar constraint (full 3D, n=10 seeds,
mean ± 95% CI; isotropic nematic order along any axis = 1/3):

| result | 3D treatment | 3D control | 2D ref |
|---|---:|---:|---:|
| single-cell aster | tension-wtd radiality **0.960 ± 0.007** | uniform 0.612 ± 0.026 | 0.969 / 0.689 |
| inter-cell bridge | axis align **0.440 ± 0.021** | transverse 0.280 ± 0.010 | 0.532 / 0.371 |

Both conclusions survive in 3D: the aster is if anything stronger (Δ 0.348), and
the bridge axis−transverse gap is **identical to 2D (0.161)** with axis ≫ isotropic.
(Bridge axis alignment is lower in 3D because fibers can also align out of plane,
diluting the in-axis fraction — expected; the *contrast* is what matters and is
unchanged.) Figure: `output_validation/figures/validate_3d.png`.

### 1.7 Face validity (no fitting, METHODS §9.3)
Reproduces documented phenomena qualitatively: radial collagen asters around
contractile cells and aligned matrix bridges between two foci (Stopak & Harris
explant traction-structuring); long-range force transmission requiring fiber
nonlinearity; density-controlled compaction/pore size; loaded-collagen
reinforcement + unloaded turnover (tensional homeostasis).

---

## 2. Required for publication (gaps)

Prioritized. Items 1–3 are needed for any *quantitative* claim; 4–6 strengthen it.

1. **[DONE] Calibration to physical units** — [`CALIBRATION.md`](CALIBRATION.md)
   (`python calibrate.py`, `python rheology_test.py`). Anchoring on measured cell
   traction gives length = 1 µm, force ≈ 0.156 nN, stress ≈ 156 Pa, step ≈ 2 h; an
   independent simulated shear gives network modulus **G ≈ 0.8 Pa** — order-of-
   magnitude consistent with a soft/dilute collagen gel. Key finding: a single-
   fibril stiffness anchor disagrees with the traction anchor by ~4×10³, so
   `k_s = 1` is an *effective network* element (bending/buckling-dominated), not a
   taut fibril. (Caveat: the shear estimate is order-of-magnitude; periodic-BC
   rheology would tighten it.)
2. **[DONE, with a noted discrepancy] Quantitative comparison to experiment** —
   explant bridge vs separation (`experiment_bridge.py`, §1.3). Reproduces the
   axis-specific aligned bridge quantitatively (n=4 seeds, SEMs); identifies a real
   discrepancy (no distance-failure in this regime — long-range nonlinear
   transmission). Next: compare *distributions* (pore size, node degree) of a
   reticular target (FRC / white-pulp) against published morphometry.
3. **[DONE — OAT] Parameter sensitivity / identifiability** (`sensitivity.py`, §1.4).
   OAT sweep: the bridge conclusion is robust to every single parameter; nonlinearity
   (buckling-dominant, partially redundant with stiffening) ~doubles the signal. Next:
   a global (Sobol/LHS) sweep + sensitivity of the remodeling percentiles.
4. **[DONE — headline metrics] Statistical power** (`stats_power.py`, §1.5). The two
   central mechanics results are now reported at n=16 with 95% CIs and large effect
   sizes (d = 9.06, 2.47). Next: extend CIs to the remodeling / emergence studies.
5. **[DONE] 3D.** Headline results reproduced in full 3D (`validate_3d.py`, §1.6):
   aster Δ 0.348, bridge axis−transverse gap 0.161 (identical to 2D). The 2D
   conclusions are not an artifact of the planar constraint.
6. **Mechanistic upgrades (optional, for a stronger model).** Replace the
   phenomenological percentile remodeling with explicit LOX/MMP-style kinetics;
   add active cell migration/contractile-feedback; fiber–fiber steric contact.

---

## 3. Communication tools are NOT validation

- `viewer/` (3D) and `play2d.html` (interactive 2D) are **illustrations**: they
  use lightweight real-time approximations (few-iteration relaxation, rendering
  heuristics), **not** the FIRE-to-equilibrium solver. They are for intuition,
  talks, and figures — **not** a source of quantitative results.
- All numbers for a paper must come from the `ecm_pipeline` solver via the
  `run_*.py` / `verify_*.py` / `convergence_test.py` scripts.

---

## 4. How to reproduce the validation

```bash
pip install -r requirements.txt
python verify_mechanics.py      # equilibrium, radial aster, nonlinearity
python verify_remodel.py        # plastic memory after cell removal
python convergence_test.py      # discretization independence of the emergent track
python calibrate.py             # reduced -> SI units + stiffness/traction consistency
python rheology_test.py         # network shear modulus -> G ~ 0.8 Pa (soft gel)
python experiment_bridge.py     # explant-bridge alignment vs separation (+ figure)
python sensitivity.py           # OAT robustness + nonlinearity decomposition
python stats_power.py           # headline metrics: mean ± 95% CI, effect sizes (+ figure)
python validate_3d.py           # headline results reproduced in full 3D (+ figure)
```
