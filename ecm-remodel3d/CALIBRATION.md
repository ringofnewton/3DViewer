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

## Not yet closed (next step)

Anchor B fixes the force/stress units from cell traction; an **independent** check
is still needed: measure the model's **network shear modulus** `G_reduced` with a
rheology simulation (fix top/bottom boundaries, apply small shear, relax to
equilibrium, read the reaction stress) and verify

```
G_reduced × 156 Pa  ∈  ~1–100 Pa   (collagen gel plateau modulus)
```

A first attempt at this rheology measurement did **not** reach force-balance
convergence with the current open-side boundary setup, so no modulus is reported
here — closing this loop (periodic/Lees–Edwards shear, larger system, tighter
`f_tol`) is the next calibration task before any quantitative (Pa-level) claim.
