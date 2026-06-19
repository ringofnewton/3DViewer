# Manuscript skeleton — cell-driven mechanical self-organization of reticular ECM

A submission-oriented outline tying the validated results into a paper. Methods
detail is in `METHODS.md`; the evidence and its caveats are in `VALIDATION.md`;
every number here is reproducible via the listed script.

---

## Working title
**Mechanical self-organization of reticular collagen: how cell placement controls
emergent extracellular-matrix architecture in a nonlinear fiber-network model.**

## One-line contribution
A reproducible discrete fiber-network platform in which simply-placed contractile
cells *secrete, contract, and plastically remodel* collagen so that reticular
architecture (asters, inter-cell bridges, spanning webs) **emerges from mechanics
and is controlled by the seeding geometry** — demonstrated, calibrated to physical
units at order of magnitude, and shown robust (parameters, seeds, 2D→3D).

## Abstract (draft)
Connective-tissue architecture is not drawn cell-by-cell; it self-organizes as a
mechanically nonlinear collagen network is contracted and remodeled by cells. We
present a discrete fiber-network model (tension-bearing, buckling, strain-
stiffening fibers with bending rigidity; FIRE equilibrium; tension-driven plastic
reinforcement and turnover) in which cells secrete and contract collagen. The
model reproduces, without fitting, the radial collagen asters around single
contractile cells and the aligned matrix bridges between cell pairs (Stopak &
Harris), with large effect sizes (Cohen's d = 9.1 and 2.5, n=16). We show the
emergent reticular network is **controlled by spheroid spacing** [RESULT], and
that the nonlinear constitutive law is the load-bearing assumption (it ~doubles
the axis-specific bridge signal; buckling dominant). Results are discretization-
independent, reproduce in full 3D, and the emergent morphometry (trivalent branch
points, tens-of-µm pores) matches reticular-network ranges. An order-of-magnitude
SI calibration (force 0.16 nN, stress 156 Pa, network modulus ~0.8 Pa) places the
model at the soft-collagen-gel scale.

## Claims (each backed by a reproducible result)
| # | Claim | Evidence | Script |
|---|---|---|---|
| C1 | Single contractile cells build radial collagen asters | tension-wtd radiality 0.97 vs 0.69, d=9.1 (n=16) | `stats_power.py` |
| C2 | Cell pairs build axis-specific aligned bridges | axis 0.53 vs perp 0.37, d=2.5 (n=16); widens with distance | `stats_power.py`, `experiment_bridge.py` |
| C3 | **Seeding geometry controls emergent connectivity** (headline) | spacing sweep [RESULT] | `emergence_control.py` |
| C4 | Nonlinearity is the driver | nonlinear ~2× linear; buckling dominant; partially redundant | `sensitivity.py` |
| C5 | Plastic memory: structure outlasts the cells | residual self-stress 4250 vs 11 (~400×) | `verify_remodel.py` |
| C6 | Emergent morphometry matches reticular networks | coordination 3.09 (92% trivalent), pores ~22 µm | `morphometry_compare.py` |

## Robustness & rigor
- **Discretization independence** (`convergence_test.py`): emergent track stable to <1.5% as nodes 1.3k→4k.
- **Statistical power** (`stats_power.py`): headline metrics with 95% CIs, large d.
- **Parameter sensitivity** (`sensitivity.py`): conclusion robust to every single parameter (OAT).
- **3D** (`validate_3d.py`): conclusions reproduce without the planar constraint.
- **SI calibration** (`calibrate.py`, `rheology_test.py`): order-of-magnitude units; model sits at soft-gel modulus (~0.8 Pa).

## Figures (in `output_validation/figures/` and `output_*/figures/`)
1. Model schematic + constitutive law (from METHODS).
2. Single-cell aster + cell-pair bridge, with stats (`stats_power.png`).
3. **Headline:** emergent connectivity vs spheroid spacing (`emergence_control.png`).
4. Bridge alignment vs separation (`bridge_alignment.png`).
5. Nonlinearity decomposition (sensitivity) + convergence.
6. Reticular morphometry: pore-size + coordination (`morphometry.png`).
7. 3D reproduction (`validate_3d.png`).

## Limitations (from VALIDATION §2, stated plainly)
Reduced/order-of-magnitude calibration (not Pa-precise); phenomenological traction
and percentile remodeling (not LOX/MMP kinetics); one experiment compared at
range/face level (no distribution-level KS/EMD yet); OAT (not global Sobol/LHS)
sensitivity; quasi-static, athermal, single matrix species.

## Remaining before submission
- Distribution-level comparison (KS/EMD) vs a digitized FRC/white-pulp histogram.
- Global (Sobol/LHS) sensitivity.
- Periodic-BC rheology to tighten the modulus.
- (Optional) explicit remodeling kinetics; active migration.

## Reproducibility
`pip install -r requirements.txt`; every claim above runs from a single script
(listed). Deterministic given seeds. See `VALIDATION.md` §4 for the full command list.
