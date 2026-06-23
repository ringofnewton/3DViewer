#!/usr/bin/env python3
"""Reticular morphometry vs literature ranges (VALIDATION.md §2.2 follow-up).

Quantifies the emergent reticular network's architecture over seeds and checks it
against representative ranges reported for collagen / lymphoid-reticular (FRC)
networks:

  * branch-point coordination — reticular/collagen networks are predominantly
    **trivalent** (3-way junctions), coordination ≈ 3 (Mikado/biopolymer theory;
    FRC conduit network morphology).
  * pore (inter-fiber) size — reticular mesh openings are on the order of **tens
    of µm** (with 1 model length unit = 1 µm, per CALIBRATION.md).

This is a comparison to literature *ranges*, not a digitized dataset; a rigorous
distribution-level test (KS/EMD against a published histogram) is the remaining
step and is noted as such.

    python morphometry_compare.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_network, metrics

N = 8
SEEDS = list(range(1, N + 1))
T_CRIT = {6: 2.571, 8: 2.365, 10: 2.262}


def grow(seed):
    p = ecm_network.NetworkParams(n_cells=24, steps=70, n_frames=2, seg_length=3.0,
                                  crosslink_dist=4.5, branch_prob=0.07,
                                  max_nodes=9000, max_tips=300, seed=seed)
    fr = ecm_network.simulate_network(p)[0][-1]
    return fr, p.domain


def morphometry(seed):
    fr, domain = grow(seed)
    nodes = fr["nodes"]
    edges = np.array([[a, b] for a, b, st, bi in fr["edges"]])
    deg = np.bincount(edges.ravel(), minlength=len(nodes))
    deg = deg[deg > 0]
    branch = deg[deg >= 3]
    coord = float(branch.mean()) if len(branch) else float("nan")
    trivalent_frac = float(np.mean(branch == 3)) if len(branch) else float("nan")
    fib = dict(p1=nodes[edges[:, 0]], p2=nodes[edges[:, 1]],
               radius=np.full(len(edges), 0.5))
    pores = metrics.pore_size_distribution(fib, domain, n_samples=2500, seed=0)
    conn = metrics.network_connectivity(fib)
    return dict(coord=coord, trivalent=trivalent_frac,
                pore_med=float(np.median(pores)), pore_mean=float(pores.mean()),
                largest=conn["largest_fraction"], pores=pores, deg=deg)


def stat(x):
    x = np.asarray(x, float); return x.mean(), T_CRIT[N] * x.std(ddof=1) / np.sqrt(len(x))


def main():
    res = [morphometry(s) for s in SEEDS]
    coord = [r["coord"] for r in res]; tri = [r["trivalent"] for r in res]
    pmed = [r["pore_med"] for r in res]; lar = [r["largest"] for r in res]
    mc, cc = stat(coord); mt, ct = stat(tri); mp, cp = stat(pmed); ml, cl = stat(lar)

    print(f"Reticular network morphometry (n={N} seeds, mean ± 95% CI)\n")
    print(f"  branch-point coordination   {mc:.2f} ± {cc:.2f}   (reticular ≈ 3, trivalent)")
    print(f"  trivalent fraction          {mt:.2f} ± {ct:.2f}   (expect majority 3-way)")
    print(f"  pore size (median, µm)      {mp:.1f} ± {cp:.1f}   (reticular mesh ~ tens of µm)")
    print(f"  largest connected fraction  {ml:.2f} ± {cl:.2f}")

    coord_ok = abs(mc - 3.0) < 0.5
    tri_ok = mt > 0.6
    pore_ok = 5.0 <= mp <= 60.0
    print("\nComparison to reticular/FRC literature ranges:")
    print(f"  coordination ≈ 3 (trivalent)        : {'YES' if coord_ok else 'NO'}")
    print(f"  predominantly 3-way junctions       : {'YES' if tri_ok else 'NO'}")
    print(f"  pore size in tens-of-µm range       : {'YES' if pore_ok else 'NO'}")
    ok = coord_ok and tri_ok and pore_ok
    print("\nPASS  emergent morphometry is consistent with reticular-network ranges"
          if ok else "\nPARTIAL  see numbers above")
    print("NOTE: comparison is to literature *ranges*; a distribution-level test")
    print("      (KS/EMD vs a digitized FRC histogram) remains as the next step.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pores = np.concatenate([r["pores"] for r in res])
        degs = np.concatenate([r["deg"] for r in res])
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].hist(pores, bins=30, color="#2c6fbb", alpha=0.85)
        ax[0].axvline(np.median(pores), color="#c0392b", ls="--",
                      label=f"median {np.median(pores):.0f} µm")
        ax[0].set_xlabel("pore size (µm)"); ax[0].set_ylabel("count")
        ax[0].set_title("Pore-size distribution"); ax[0].legend(fontsize=8)
        vals, cnts = np.unique(degs[degs <= 6], return_counts=True)
        ax[1].bar(vals, cnts / cnts.sum(), color="#27ae60")
        ax[1].set_xlabel("node degree"); ax[1].set_ylabel("fraction")
        ax[1].set_title("Node coordination\n(branch points ≈ trivalent)")
        fig.suptitle(f"Emergent reticular morphometry (pooled, n={N} seeds)")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "morphometry.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'morphometry.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
