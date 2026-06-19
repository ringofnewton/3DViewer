#!/usr/bin/env python3
"""Headline result: spheroid spacing CONTROLS emergent reticular connectivity.

This is the paper's central *finding* (not just validation): with a uniform
collagen matrix and simply-placed spheroids, the emergent strut-and-pore network
is controlled by the spheroid layout. We sweep the spacing of 4 ring-placed
spheroids over seeds and measure the reinforced-network connectivity (largest
connected fraction) and branch-point count — mean ± 95% CI.

Claim under test (METHODS §9.1): too-tight spacing fuses into one clump (low
spanning connectivity); wider spacing yields a more connected spanning web.

    python emergence_control.py
"""
from __future__ import annotations
import os
import numpy as np
import run_emergent as RE                       # reuse gel/ring_centers/P/RP/metrics
from ecm_pipeline import ecm_remodel as rem, seeding, morpho_metrics as mm

SPACINGS = [(34.0, "tight"), (52.0, "medium"), (72.0, "wide")]
SEEDS = [1, 2, 3, 4, 5]
T_CRIT = {5: 2.776, 6: 2.571, 8: 2.365}


def emergent_seeded(centers, seed):
    cells = seeding.spheroids(centers, n_cells=14, radius=12.0, seed=seed)
    net0 = RE.gel(seed=seed)
    _, final = rem.remodel(net0, cells, RE.P, RE.RP)
    return final


def stat(x):
    x = np.asarray(x, float)
    return x.mean(), T_CRIT[len(x)] * x.std(ddof=1) / np.sqrt(len(x))


def main():
    print(f"Spheroid-spacing control of emergent connectivity "
          f"(4 spheroids, n={len(SEEDS)} seeds, mean ± 95% CI)\n")
    print(f"{'spacing':>8} {'R(µm)':>6} {'connectivity':>14} {'branch points':>16}")
    results = {}
    for R, name in SPACINGS:
        conn, branch = [], []
        for s in SEEDS:
            net = emergent_seeded(RE.ring_centers(4, R=R), s)
            conn.append(mm.connectivity(net)); branch.append(mm.branch_points(net))
        mc, cc = stat(conn); mb, cb = stat(branch)
        results[name] = (R, mc, cc, mb, cb)
        print(f"{name:>8} {R:6.0f} {mc:8.2f} ± {cc:.2f} {mb:10.0f} ± {cb:.0f}")

    conns = {k: v[1] for k, v in results.items()}
    best = max(conns, key=conns.get)
    tight, wide = conns["tight"], conns["wide"]
    print(f"\nMost spanning network: '{best}' spacing (connectivity {conns[best]:.2f})")
    print(f"tight {tight:.2f} vs wide {wide:.2f}  →  "
          f"{'wider spacing spans more (claim holds)' if wide > tight + 0.05 else 'difference small'}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [n for _, n in SPACINGS]
        Rs = [results[n][0] for n in names]
        mc = [results[n][1] for n in names]; cc = [results[n][2] for n in names]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.errorbar(Rs, mc, yerr=cc, fmt="o-", color="#c0392b", capsize=4, lw=2)
        for R, n in zip(Rs, names):
            ax.annotate(n, (R, results[n][1]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=10)
        ax.set_xlabel("spheroid spacing R (µm)")
        ax.set_ylabel("reticular connectivity (largest connected fraction)")
        ax.set_title("Spheroid spacing controls emergent connectivity\n"
                     f"(4 spheroids; mean ± 95% CI, n={len(SEEDS)})")
        ax.grid(alpha=0.3); fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "emergence_control.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'emergence_control.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
