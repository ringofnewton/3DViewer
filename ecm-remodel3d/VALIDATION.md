# Validation status & publication roadmap

What is and isn't established for the discrete fiber-network ECM model
(`ecm_pipeline`, see `METHODS.md`). This file separates **the validated science**
(the Python pipeline) from **communication tools** (the browser viewers), and
lists concretely what remains before the model supports a *quantitative*,
publishable claim.

> Scope reminder (METHODS §10): the model is in **reduced units and is not yet
> calibrated**. Its present, defensible claim is **qualitative**: cell-driven
> mechanical self-organization of a nonlinear fiber network produces — and lets
> us control — reticular collagen architecture. Quantitative/predictive claims
> require the work in §"Required for publication" below.

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

### 1.3 Face validity (no fitting, METHODS §9.3)
Reproduces documented phenomena qualitatively: radial collagen asters around
contractile cells and aligned matrix bridges between two foci (Stopak & Harris
explant traction-structuring); long-range force transmission requiring fiber
nonlinearity; density-controlled compaction/pore size; loaded-collagen
reinforcement + unloaded turnover (tensional homeostasis).

---

## 2. Required for publication (gaps)

Prioritized. Items 1–3 are needed for any *quantitative* claim; 4–6 strengthen it.

1. **Calibration to physical units.** Map reduced moduli → Pa (collagen network
   shear/Young's modulus), traction → nN per cell (measured single-cell tractions),
   and remodeling steps → hours (collagen turnover rates). Report all parameters in
   SI with sources. Until then, only qualitative/relative claims are defensible.
2. **Quantitative comparison to experiment.** Pick ≥1 dataset with measurable
   geometry (e.g. two-explant bridge alignment vs separation; FRC/white-pulp
   reticular pore-size and node-degree distributions) and compare *distributions*
   (not just means) with error bars and an effect-size/statistical test.
3. **Parameter sensitivity / identifiability.** Sweep the key parameters
   (`buckle_ratio`, `stiffen_alpha`, `k_bend`, `traction`, `reach`,
   reinforce/slack percentiles) and report which conclusions are robust vs
   parameter-dependent. A one-at-a-time + a global (e.g. Sobol/LHS) sweep.
4. **Statistical power.** Report every emergent metric as mean ± CI over ≥10–20
   seeds; current studies often use few seeds.
5. **3D.** The force law is already 3D; rerun the headline studies in 3D (higher
   cost) to show the 2D conclusions survive, or scope the paper explicitly to 2D.
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
```
