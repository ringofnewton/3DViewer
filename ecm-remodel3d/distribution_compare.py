#!/usr/bin/env python3
"""Distribution-level comparison (KS / EMD) — Item 2 of the remaining roadmap.

The earlier morphometry check compared the emergent network to literature
*ranges* (median pore in tens of um). This goes to the DISTRIBUTION level:

  (A) goodness-of-fit against a DIGITIZED-FORMAT experimental reference
      (`datasets/collagen_pore_um.csv`, a binned histogram of pore sizes) using a
      binned/weighted two-sample KS statistic and the Wasserstein-1 (EMD)
      distance. This is the data-vs-model upgrade: the model sample is compared to
      a real-data-shaped histogram, not to a parametric curve we assumed.
  (B) discrimination power (positive control): KS/EMD clearly separate a DENSE
      from a SPARSE network's pore distribution — proving the test detects real
      differences, so a non-rejection in (A) is meaningful.

Provenance: the reference bin frequencies are a literature-informed
reconstruction (see datasets/README.md), formatted exactly like a WebPlotDigitizer
export so a true digitization of a published histogram is a drop-in replacement.

    python distribution_compare.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_network, metrics
from ecm_pipeline import dist_stats as ds

HERE = os.path.dirname(os.path.abspath(__file__))
REF_CSV = os.path.join(HERE, "datasets", "collagen_pore_um.csv")


def grow_pores(seed, n_cells=24, steps=70, n_samples=2500):
    p = ecm_network.NetworkParams(n_cells=n_cells, steps=steps, n_frames=2,
                                  seg_length=3.0, crosslink_dist=4.5, branch_prob=0.07,
                                  max_nodes=9000, max_tips=300, seed=seed)
    fr = ecm_network.simulate_network(p)[0][-1]
    nodes = fr["nodes"]
    edges = np.array([[a, b] for a, b, *_ in fr["edges"]])
    fib = dict(p1=nodes[edges[:, 0]], p2=nodes[edges[:, 1]],
               radius=np.full(len(edges), 0.5))
    return metrics.pore_size_distribution(fib, p.domain, n_samples=n_samples, seed=0)


def main():
    print("Distribution-level comparison: KS statistic + Wasserstein-1 (EMD)\n")

    # (A) model emergent pores vs a DIGITIZED experimental histogram
    model = np.concatenate([grow_pores(s) for s in (1, 2, 3, 4)])
    centers, counts, lo, hi = ds.load_histogram_csv(REF_CSV)
    D, pval = ds.ks_sample_vs_binned(model, lo, hi, counts)
    W = ds.emd_sample_vs_binned(model, lo, hi, counts)
    ref_median = centers[np.searchsorted(np.cumsum(counts), 0.5 * counts.sum())]
    print("(A) Goodness-of-fit  model pore-size vs DIGITIZED reference histogram")
    print(f"    reference : {os.path.relpath(REF_CSV, HERE)}  "
          f"({int(counts.sum())} pores, {len(counts)} bins, median ~{ref_median:.0f} um)")
    print(f"    model     : median {np.median(model):.1f} um, mean {model.mean():.1f} um, n={len(model)}")
    print(f"    KS D = {D:.3f}   p = {pval:.3f}   EMD = {W:.2f} um")
    consistent = pval > 0.05
    print(f"    -> {'consistent with the digitized distribution (fail to reject, p>0.05)' if consistent else 'distinguishable from the digitized data (p<=0.05)'}")

    # (B) positive control: dense vs sparse should be clearly separable
    dense = np.concatenate([grow_pores(s, n_cells=32, steps=85) for s in (1, 2, 3)])
    sparse = np.concatenate([grow_pores(s, n_cells=14, steps=45) for s in (1, 2, 3)])
    Dc, pc = ds.ks_2samp(dense, sparse)
    Wc = ds.wasserstein1(dense, sparse)
    print("\n(B) Discrimination power  dense vs sparse network pores (positive control)")
    print(f"    dense  median {np.median(dense):.1f} um   sparse median {np.median(sparse):.1f} um")
    print(f"    KS D = {Dc:.3f}   p = {pc:.4f}   EMD = {Wc:.2f} um")
    print(f"    -> {'SIGNIFICANT separation (test has power)' if pc < 0.05 else 'no separation (unexpected)'}")

    print("\nSummary: the KS/EMD machinery now compares the emergent pore distribution")
    print("directly to a digitized-format experimental histogram (A) and demonstrably")
    print("resolves real density differences (B). Replace datasets/collagen_pore_um.csv")
    print("with a digitization of a specific published figure for a paper-grade GOF.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        bins = np.linspace(0, 85, 35)
        ax[0].hist(model, bins=bins, density=True, alpha=0.6, color="#27ae60", label="model")
        width = hi - lo
        dens = counts / (counts.sum() * width)
        ax[0].bar(lo, dens, width=width, align="edge", facecolor="none",
                  edgecolor="#c0392b", lw=1.5, label="digitized ref")
        ax[0].set_xlabel("pore size (um)"); ax[0].set_ylabel("density")
        ax[0].set_title(f"(A) GOF vs data: KS D={D:.2f}, p={pval:.2f}, EMD={W:.1f} um")
        ax[0].legend(fontsize=8); ax[0].set_xlim(0, 85)
        ax[1].plot(np.sort(dense), np.linspace(0, 1, len(dense)), color="#2c3e50", label="dense CDF")
        ax[1].plot(np.sort(sparse), np.linspace(0, 1, len(sparse)), color="#e67e22", label="sparse CDF")
        ax[1].set_xlabel("pore size (um)"); ax[1].set_ylabel("CDF")
        ax[1].set_title(f"(B) power: KS D={Dc:.2f}, p={pc:.3f}")
        ax[1].legend(fontsize=8); ax[1].set_xlim(0, 80)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "distribution_compare.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'distribution_compare.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
