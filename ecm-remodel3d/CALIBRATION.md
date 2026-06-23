# Unit calibration (reduced → SI) — VALIDATION.md §2.1

Maps the model's reduced units to SI and reports a consistency check. Run:

```bash
python calibrate.py
```

> Anchors are **order-of-magnitude** literature values (see `ecm_pipeline/calibration.py`
> for attribution). Confirm exact numbers against primary sources before publication.

## Anchors

| quantity | value | source (representative) |
|---|---|---|
| single collagen fibril modulus `E` | ~10⁸ Pa (0.1 GPa, lower end of 0.1–1 GPa) | Wenger 2007; Shen 2008 |
| model fiber radius `r` | 0.12 µm | `NetworkParams` default |
| segment rest length `ℓ₀` | 7 µm | `build_random_network` default |
| single-fibroblast total traction | ~50 nN (range 10–100) | Munevar 2001; Legant 2010 |
| collagen gel plateau shear modulus | ~1–100 Pa (1–3 mg/mL) | Storm 2005; Licup 2015 |
| remodeling step | ~2 h | Grinnell 2003 |

## Headline result — the two anchors are inconsistent by ~4×10³

| anchor | reduced force unit | consequence |
|---|---|---|
| **A: single-fibril stiffness** (`k_s = E·A/ℓ₀ ≈ 0.65 N/m`) | **646 nN** | predicts cell traction ≈ **2×10⁵ nN** — **~4150× too large** vs measured ~50 nN |
| **B: measured cell traction** (50 nN) | **0.156 nN** | implies effective fiber modulus ≈ **24 kPa** — **~4150× softer** than a single fibril |

**Interpretation.** A taut single collagen fibril (0.1 GPa) cannot be the meaning
of the reduced `k_s = 1`: it would make cells ~10³–10⁴× stronger than measured.
The consistent reading is that `k_s = 1` is an **effective, network-scale element**
whose compliance is dominated by **fiber bending, buckling, and reorientation**
(the model is deliberately sub-isostatic, METHODS §3.2) — not by axial fibril
stretching. This is the expected physics of soft collagen networks, and it means
the model must be **anchored on the network/traction scale, not the single fibril.**

## Resulting SI scales (Anchor B, recommended)

| unit | value |
|---|---|
| length | 1 model unit = **1 µm** |
| force | 1 reduced force = **0.156 nN** |
| stress | 1 reduced stress = **156 Pa** |
| time | 1 remodeling step ≈ **2 h** |

## Independent consistency check — network shear modulus (`rheology_test.py`)

Anchor B fixes the force/stress units from cell traction; an **independent** check
measures the model's **network shear modulus** by simulated clamped simple shear
(fix top/bottom layers, impose shear γ, relax, read the top-plate reaction stress):

| γ | seeds | `G_reduced` | `G_SI = G_reduced × 156 Pa` |
|---|---|---|---|
| 0.01–0.02 | 2 each | **0.0050 ± 0.0002** | **≈ 0.77 Pa** |

The result is γ-independent and seed-robust. **G_SI ≈ 0.8 Pa is order-of-magnitude
consistent with the soft/dilute end of the collagen-gel range (~1–100 Pa)** — i.e.
the calibration is self-consistent and the model network corresponds to a soft
(~0.5–1 mg/mL) collagen matrix. Raising fiber density + crosslinking would move it
up the gel-stiffness range.

> Caveat: the clamped/open-side relaxation left a residual `max|F|` above `f_tol`
> (a sub-isostatic network has soft shear modes), so 0.8 Pa is an **order-of-
> magnitude** estimate. A periodic (Lees–Edwards) shear on a larger system would
> tighten the number; the order of magnitude is already informative.

**Bottom line:** anchoring on measured cell traction gives length = 1 µm, force ≈
0.156 nN, stress ≈ 156 Pa, step ≈ 2 h, and an independent shear measurement lands
the network modulus at ≈ 0.8 Pa — consistent with a soft collagen gel. The model
is now calibrated to SI at order-of-magnitude fidelity.
