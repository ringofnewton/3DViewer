#!/usr/bin/env python3
"""Print the reduced→SI calibration and the stiffness/traction consistency check.

    python calibrate.py

See `CALIBRATION.md` for the interpretation. The headline result: a single-fibril
stiffness anchor and a measured-cell-traction anchor are mutually inconsistent by
~10^3–10^4×, which shows the reduced `k_s = 1` represents an *effective network*
stiffness (bending/buckling-dominated), not a taut single fibril — so the model
should be anchored on the network/traction scale.
"""

from __future__ import annotations

from ecm_pipeline.calibration import Anchors, calibrate


def main():
    a = Anchors()
    r = calibrate(a)
    print("ECM model unit calibration (reduced → SI)\n")
    print("Anchors (order-of-magnitude; confirm vs primary sources):")
    print(f"  single fibril E      = {a.E_fibril_Pa:.1e} Pa")
    print(f"  fiber radius         = {a.fiber_radius_um} µm")
    print(f"  segment rest length  = {a.seg_len_um} µm")
    print(f"  cell traction (total)= {a.cell_traction_nN} nN")
    print(f"  remodeling step      = {a.step_hours} h\n")

    print(f"reduced traction per cell (Σm)            = {r['sum_m']:.1f}\n")

    print("Anchor A — stiffness from single fibril:")
    print(f"  single-segment spring k_s              = {r['k_s_SI_fibril']:.3e} N/m")
    print(f"  ⇒ reduced force unit                   = {r['force_unit_A_nN']:.0f} nN")
    print(f"  ⇒ PREDICTED cell traction              = {r['pred_traction_nN']:.2e} nN")
    print(f"     (measured ≈ {a.cell_traction_nN:.0f} nN  →  off by {r['discrepancy']:.0f}×)\n")

    print("Anchor B — measured cell traction (recommended):")
    print(f"  reduced force unit                     = {r['force_unit_B_nN']:.3f} nN")
    print(f"  reduced stress unit                    = {r['stress_unit_Pa']:.1f} Pa")
    print(f"  length unit                            = {r['length_unit_um']:.0f} µm")
    print(f"  1 remodeling step                      ≈ {r['step_hours']:.1f} h")
    print(f"  ⇒ implied effective fiber modulus      = {r['E_eff_Pa']:.2e} Pa")
    print(f"     (vs single fibril {a.E_fibril_Pa:.1e} Pa  →  ~{a.E_fibril_Pa/r['E_eff_Pa']:.0f}× softer)\n")

    print("Reading: k_s = 1 is NOT a taut single fibril; it is an effective,")
    print("bending/buckling-dominated network element. Anchor on the network scale.")
    print("Next step to CLOSE the loop: measure the model's network shear modulus")
    print("(rheology sim) and check  G_reduced × stress_unit ∈ ~1–100 Pa (collagen gel).")


if __name__ == "__main__":
    main()
