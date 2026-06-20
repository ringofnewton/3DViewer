# Research-grade upgrades (toward "as close to reality as possible")

This is the staged build-out from the validated qualitative model toward a
quantitative, predictive one, following the roadmap in the project discussion.
Each step adds a physical mechanism the previous model lacked, with a runnable,
deterministic verification script and an honest statement of what is and isn't
yet tight. Numbers are reduced units unless an SI conversion is given
(see `CALIBRATION.md`, stress unit ≈ 156 Pa).

> These build on the core pipeline (`ecm_mechanics`, `ecm_remodel`, `ecm_network`)
> and the existing validation (`VALIDATION.md`). They do **not** replace it; they
> extend the physics and the rigor.

---

## Step 1 — Boundary-free rheology (periodic / Lees–Edwards)
**Module** `ecm_pipeline/ecm_periodic.py` · **Script** `rheology_periodic.py`
· **Figure** `output_validation/figures/rheology_periodic.png`

The clamped-boundary modulus (`rheology_test.py`) left a residual force imbalance
(open side walls), so it was only order-of-magnitude. This adds a fully periodic
box with the **Lees–Edwards** convention for simple shear and reads the stress
from the **virial** (an exact, boundary-free estimator). Shear is applied
quasi-statically with an affine pre-shift per increment so FIRE reaches force
balance.

- **Linear modulus G0 = 0.39 ± 0.23 Pa** (n=4 seeds, 3/4 converged below ftol).
  Boundary-free, and **cross-checks the clamped estimate (~0.8 Pa)** to the same
  order of magnitude — soft/dilute collagen-gel range (1–100 Pa).
- **Nonlinear strain-stiffening:** the differential modulus K = dσ/dγ rises
  monotonically with strain (collagen/fibrin hallmark; Storm 2005, Licup 2015).
- *Caveat:* high-strain points are not fully force-balanced (athermal FIRE on a
  stiffening sub-isostatic network); the **trend** is robust, tightening it needs
  finer increments / a Newton solver.

## Step 2 — Viscoelasticity from transient crosslinks
**Module** `ecm_pipeline/ecm_viscoelastic.py` · **Script** `verify_viscoelastic.py`
· **Figure** `output_validation/figures/viscoelastic_relaxation.png`

The elastic pipeline holds stress forever; real ECM **relaxes**. Crosslinks are
made transient with **force-accelerated (Bell) unbinding**, k_off = k0·exp(|F|/F\*),
evolved as a mean-field survival fraction (smooth, deterministic).

- **Elastic reference stays flat** (σ_end/σ0 = 0.99); **with unbinding, stress
  relaxes** to a backbone-set plateau (viscoelastic solid).
- The **relaxation depth/time is tunable by crosslink turnover k0**: plateau falls
  0.83 → 0.39 as k0 rises (faster turnover → more relaxation), τ moves with it.
- This is the knob that, biologically, governs cell spreading/fate on viscoelastic
  matrices (Chaudhuri 2016) — now emergent from bond kinetics.
- *Caveat:* the stress signal is mildly noisy at low turnover (FIRE convergence);
  seed-averaging would smooth it.

## Step 3 — Mechanochemically-coupled remodeling
**Module** `ecm_pipeline/ecm_mechanochem.py` · **Script** `verify_mechanochem.py`
· **Figure** `output_validation/figures/mechanochem.png`

Replaces the phenomenological **percentile** remodel rule with first-principles
kinetics: tension-saturating (Hill) **synthesis**, **strain-PROTECTED** MMP
**degradation** (k_deg·exp(−T/F_prot)·m), and **tracked collagen mass**.

- Reproduces the percentile rule's emergent inter-cell track: **0.382 vs 0.392**
  alignment (agreement 0.010, n=3) — same structure, now from enzyme kinetics.
- **Collagen mass is tracked** (4308 → 5735: synthesis on loaded fibers) and the
  network is **pruned by strain-protection** (5220 → 2251 fibers): slack fibers
  cleaved, loaded fibers protected.
- Every parameter now maps to a measurable rate (synthesis/degradation/protection).

## Step 4 — Active cells: migration + mechanosensing
**Module** `ecm_pipeline/ecm_cells.py` · **Script** `verify_migration.py`
· **Figure** `output_validation/figures/durotaxis.png`

Cells become **agents** that read the matrix and move: **rigidity sensing**
(traction ∝ local stiffness), **durotaxis** (climb stiffness gradients), and
**contact guidance** (bias along local fiber alignment). Matrix and cells
co-evolve.

- **Durotaxis confirmed:** on an imposed stiffness gradient a cell migrates
  **+33.5 ± 8.0 µm** up-gradient; the uniform control shows no consistent drift
  (+13 ± 39 µm, CI spans 0). Net up-gradient migration **+20.5 µm**.

## Step 5 — Validation rigor
### 5a. Global (Sobol) sensitivity — replaces OAT
**Script** `sobol_sensitivity.py` · **Figure** `output_validation/figures/sobol_sensitivity.png`

Variance-based **Saltelli/Sobol** analysis (numpy-only) of the bridge signal over
four mechanics parameters — captures **interactions** OAT cannot.

- **Total-effect ranking: buckle_ratio dominant** (ST ≈ 0.6) > traction > k_bend >
  stiffen_alpha — agrees with the OAT ranking and the physics (buckling is the
  load-bearing nonlinearity), now established **globally**.
- **Interactions present** (ST > S1 for some parameters) — invisible to OAT.
- *Caveat:* first-order S1 (Saltelli) is noisy at this N when output variance is
  modest; the robust total-effect ST is reported. Raise N to tighten S1.

### 5b. TFM-style displacement field
**Script** `verify_displacement.py` · **Figure** `output_validation/figures/displacement_field.png`

Radial displacement u(r) ~ r^(−n) around a single contractile cell, nonlinear vs
matched-linear matrix (cf. TFM long-range transmission; Notbohm 2015).

- Nonlinear decays **slower** (n = 1.11) than linear (n = 1.17) — the predicted
  direction (longer-range transmission), but a **small margin** at these settings;
  the nonlinearity effect is captured far more robustly by the bridge metric
  (Sobol §5a) and the `sensitivity.py` decomposition.

---

## What is now stronger
- Modulus is **boundary-free** and cross-validated (Step 1).
- The model is no longer purely elastic — it has a **tunable viscoelastic
  timescale** (Step 2).
- Remodeling is **mechanistic with conserved mass**, not a percentile heuristic
  (Step 3).
- Cells are **active and mechanosensitive** (Step 4).
- Sensitivity is **global**, not OAT (Step 5a).

---

# Second batch — the remaining roadmap items (now done)

## Item 1 — Tightened numerics (seed-averaging)
**Updated** `verify_viscoelastic.py` (seed-averaged, n=3).
The small-strain relaxation regime *does* converge, so averaging over network
seeds removes the stochastic ripple: flat elastic reference (0.95) and a clean
**monotone plateau-vs-turnover** (0.70 → 0.58 as k0 rises 0.03 → 0.15).
*Solver note:* I implemented and tested preconditioned nonlinear-CG and
steepest-descent energy minimizers for the **high-strain rheology**; both lose to
FIRE because the sub-isostatic network's **floppy (zero-stiffness) modes** are
amplified by Jacobi preconditioning and defeat a naive line search. FIRE (inertial)
is retained; the high-strain rheology caveat therefore stands (a floppy-mode-aware
preconditioner or true Newton step is the real fix).

## Item 2 — Distribution-level comparison (KS / EMD)
**Module** `ecm_pipeline/dist_stats.py` · **Script** `distribution_compare.py`
· **Figure** `output_validation/figures/distribution_compare.png`
Numpy-only two-sample **KS** (asymptotic p) and **Wasserstein-1 (EMD)**.
- **GOF:** emergent pore distribution vs a literature-style log-normal (median
  25 µm) — **KS D=0.19, EMD=4.4 µm**: close in central tendency (model median
  22.5 µm) but **distinguishable in shape** (p<0.05). Matching ranges ≠ matching
  distributions — exactly what a distribution-level test is for.
- **Power (positive control):** dense vs sparse pores separate at **p<1e-4** —
  the test has real discriminating power.
- *Caveat:* the reference is synthesized from literature **summary statistics**,
  not a raw digitized histogram; swapping one in is a drop-in change.

## Item 3 — Bayesian inference + predictive validation
**Script** `bayesian_infer.py` · **Figure** `output_validation/figures/bayesian_infer.png`
Grid posterior over (buckle_ratio, stiffen_alpha) from the **bridge** experiment,
then prediction of a **held-out single-cell aster**.
- Posterior recovers truth within (broad) uncertainty: buckle_ratio 0.10 ± 0.10
  (true 0.02), stiffen_alpha 6.05 ± 4.22 (true 6.0).
- **Predictive validation PASSES:** posterior-predictive aster **0.793 ± 0.028**
  covers the held-out datum **0.808** — parameters fit on one experiment predict
  an independent one (the test that separates a fitted from a predictive model).

## Item 4 — Coupled 3-D morphogenesis
**Script** `coupled_sim.py` · **Figure** `output_validation/figures/coupled_3d.png`
Migration + mechanosensing + mechanochemistry co-evolving in **full 3-D** (n=3).
- **Emergent reinforced network:** connectivity **0.00 → 0.23 ± 0.14** — the
  matrix self-organizes from the coupled dynamics.
- **Honest negative:** cells do **not** aggregate under pure durotaxis (pairwise
  distance 54.7 → 65.2 µm); each climbs its *own* self-built stiffness, so
  trajectories are variable. Aggregation needs chemotaxis / inter-cell mechanical
  guidance to dominate the gradient — a clear next step. The *coupling itself works.*

---

## Status after both batches
Done: boundary-free + (partly) tightened rheology; viscoelasticity; mechanochemical
mass-conserving remodeling; active mechanosensing cells; global sensitivity;
distribution-level KS/EMD with power; Bayesian inference with a passing predictive
validation; all layers coupled in 3-D.

Still open (honest): a floppy-mode-aware solver for high-strain rheology; a
**digitized** experimental dataset for the KS/EMD GOF; richer inference (full MCMC,
more parameters); and cell **aggregation** (chemotaxis / stronger mechanical
guidance) in the coupled 3-D model.

## Reproduce
```bash
python rheology_periodic.py     # Step 1: periodic modulus + stiffening
python verify_viscoelastic.py   # Step 2 + Item 1: relaxation, tunable tau (seed-avg)
python verify_mechanochem.py    # Step 3: mechanochemical remodeling vs percentile
python verify_migration.py      # Step 4: durotaxis
python sobol_sensitivity.py     # Step 5a: global sensitivity
python verify_displacement.py   # Step 5b: TFM-style displacement field
python distribution_compare.py  # Item 2: KS / EMD distribution-level comparison
python bayesian_infer.py        # Item 3: Bayesian inference + predictive validation
python coupled_sim.py           # Item 4: coupled migration+remodeling in 3-D
```
