#!/usr/bin/env python3
"""Convergence / discretization-independence check (publication validation).

A discrete-network result is only trustworthy if the *emergent observable* does
not drift as the fiber discretization is refined. We fix the physical setup (a
contractile cell pair in a collagen slab) and refine the segment length
`ℓ_s` (more nodes / bending triples per fiber), then measure the inter-cell
**band alignment** (nematic order along the cell–cell axis). If the model is
discretization-independent, the observable should approach a limit and the
successive relative change should shrink.

    python convergence_test.py

Reports a table over resolutions × seeds and PASS/FAIL on the convergence of the
mean observable (successive relative change below a tolerance at the finest step).
"""

from __future__ import annotations

import numpy as np

from ecm_pipeline import ecm_mechanics as mech

DOMAIN = 200.0
ZC = DOMAIN / 2.0
SEG_LENS = [12.0, 8.0, 5.5, 4.0]     # coarse -> fine discretization
SEEDS = [1, 2, 3]
TOL = 0.05                            # |Δobservable|/observable at the finest step


def band_alignment(seg_len, seed):
    """Relax a 2-cell collagen slab at this discretization; return band alignment."""
    # keep the physical network (fiber count, length, density) fixed; only the
    # segment length — i.e. how finely each fiber is discretized — changes.
    net0 = mech.build_random_network(DOMAIN, n_fibers=220, fiber_len=70.0,
                                     seg_len=seg_len, crosslink_dist=4.5,
                                     seed=seed, slab_thickness=26)
    c = DOMAIN / 2
    cells = np.array([[c - 28, c, ZC], [c + 28, c, ZC]])
    P = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0,
                        buckle_ratio=0.02, max_iter=3000)
    net, info = mech.relax(net0, cells, P)
    axis = cells[1] - cells[0]
    val, _ = mech.axis_alignment(net, axis, (cells[0], cells[1]), band=22.0)
    return val, info.get("converged", False), len(net0.nodes)


def main():
    print("Convergence of inter-cell band alignment vs fiber discretization")
    print(f"(finer = smaller seg_len; physical setup fixed; {len(SEEDS)} seeds)\n")
    print(f"{'seg_len':>8} {'~nodes':>7} {'mean align':>11} {'std':>7} {'Δrel vs prev':>13}")
    means = []
    for sl in SEG_LENS:
        vals, nodes = [], 0
        for sd in SEEDS:
            v, conv, n = band_alignment(sl, sd)
            vals.append(v); nodes = n
            if not conv:
                print(f"  (warning: seg_len={sl} seed={sd} did not fully converge)")
        m, s = float(np.mean(vals)), float(np.std(vals))
        drel = abs(m - means[-1]) / (abs(means[-1]) + 1e-9) if means else float("nan")
        means.append(m)
        drel_s = "    —" if np.isnan(drel) else f"{drel:12.3f}"
        print(f"{sl:8.1f} {nodes:7d} {m:11.3f} {s:7.3f} {drel_s}")

    final_drel = abs(means[-1] - means[-2]) / (abs(means[-2]) + 1e-9)
    isotropic = 1.0 / 3.0
    print(f"\nfinest-step relative change = {final_drel:.3f} (tol {TOL})")
    print(f"limit alignment ≈ {means[-1]:.3f}  (isotropic = {isotropic:.3f}; "
          f">{isotropic:.2f} means a real aligned track)")
    ok = final_drel < TOL and means[-1] > isotropic + 0.05
    print("\nPASS  observable is discretization-independent and shows a real track"
          if ok else "\nFAIL  observable still drifting or no track — investigate")


if __name__ == "__main__":
    main()
