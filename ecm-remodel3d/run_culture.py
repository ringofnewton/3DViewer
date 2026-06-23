#!/usr/bin/env python3
"""Cell-density / DLP-pattern → ECM network morphology (open vs dense pores).

Reproduces, in simulation, the kind of contrast seen between an open coarse
reticular network (low cell density) and a dense fine network (high density) —
the naive-vs-fibrotic spleen morphology. Mechanism: contractile cells condense
the collagen network into struts around cell-free pores; cell SPACING sets the
pore size, so density controls the morphology. Cells are seeded by pattern, as a
DLP printer would gel chosen regions of a GelMA + cell suspension.

Outputs (output_culture/figures/):
  density_morphology.png   low / medium / high density → ECM network (dark field)
  density_poresize.png     pore-size distributions shrink with density
  dlp_patterns.png         DLP-printed seeding patterns → patterned ECM

  python run_culture.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_culture")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 220.0, 110.0

P = mech.MechParams(reach=30, reach_decay=17, planar=True, traction=1.5)
RP = rem.RemodelParams(n_steps=4, compaction=0.06, reinforce_rate=0.3,
                       reinforce_pct=60, slack_pct=30, slack_steps=3)


def condense(cells, seed=3):
    """Cells deposit ECM locally, then contract it into struts around pores."""
    net0 = mech.build_deposited_network(cells, DOMAIN, fibers_per_cell=12,
                                        secretion_radius=32.0, seed=seed)
    _, final = rem.remodel(net0, cells, P, RP)
    return final


def pore_size_2d(net, n=3000, seed=1):
    p1 = net.nodes[net.edges[:, 0]][:, :2]
    p2 = net.nodes[net.edges[:, 1]][:, :2]
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, DOMAIN, size=(n, 2))
    d = metrics.point_segment_distance(pts, p1, p2)
    return d.min(axis=1)


def draw_darkfield(ax, net, cells=None, show_cells=False):
    """Microscopy-like: bright collagen network on black, pores = dark gaps."""
    ax.set_facecolor("black")
    k = net.k_scale
    kmax = max(k.max(), 1.5)
    order = np.argsort(k)
    for idx in order:
        e = net.edges[idx]
        pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        v = (k[idx] - 1.0) / (kmax - 1.0)
        # brighter + thicker where reinforced (the condensed struts)
        g = 0.30 + 0.70 * min(max(v, 0), 1)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=(g, g, min(1, g + 0.05)),
                lw=0.22 + 0.9 * min(max(v, 0), 1), alpha=0.55 + 0.4 * min(max(v, 0), 1),
                solid_capstyle="round")
    if show_cells and cells is not None:
        ax.scatter(cells[:, 0], cells[:, 1], s=14, c="#35d6d0", alpha=0.5, zorder=5)
    ax.set_xlim(20, DOMAIN - 20); ax.set_ylim(20, DOMAIN - 20)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def density_study():
    print("[1/2] density → pore morphology …")
    conditions = [("low density\n(~16 cells)", 16, 6),
                  ("medium density\n(~40 cells)", 40, 6),
                  ("high density\n(~72 cells)", 72, 6)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.patch.set_facecolor("#0a0a0a")
    pores = {}
    for ax, (label, n, sd) in zip(axes, conditions):
        cells = seeding.uniform(DOMAIN, n=n, z=Z, seed=sd)
        net = condense(cells)
        ps = pore_size_2d(net)
        pores[label] = ps
        draw_darkfield(ax, net, cells)
        ax.set_title(f"{label}\nmedian pore {np.median(ps):.0f} µm",
                     color="white", fontsize=12, fontweight="bold")
        print(f"      {label.splitlines()[0]:16s} median pore = {np.median(ps):.1f} µm "
              f"· {len(net.edges)} fibers")
    fig.suptitle("Cell density sets ECM pore size — low = open coarse network, "
                 "high = dense fine network", color="white", fontweight="bold",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(os.path.join(FIG, "density_morphology.png"), dpi=140,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # pore-size distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = {"low": "#75e4a8", "medium": "#7aa2ff", "high": "#ff8d8d"}
    for label, ps in pores.items():
        key = label.split()[0]
        ax.hist(ps, bins=45, range=(0, 55), histtype="step", lw=2.2,
                color=cols.get(key, "#999"), label=f"{key}  (median {np.median(ps):.0f} µm)")
    ax.set(title="Pore-size distribution shrinks as cell density rises",
           xlabel="pore radius (µm)", ylabel="count")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "density_poresize.png"), dpi=140)
    plt.close(fig)


def dlp_study():
    print("[2/2] DLP-printed seeding patterns …")
    patterns = [
        ("stripes", seeding.in_mask(DOMAIN, 46, seeding.stripes_mask(DOMAIN, period=52, duty=0.5), z=Z, seed=2)),
        ("ring", seeding.in_mask(DOMAIN, 44, seeding.ring_mask(DOMAIN, 40, 72), z=Z, seed=2)),
        ("two blocks", seeding.in_mask(DOMAIN, 44, seeding.two_blocks_mask(DOMAIN, gap=44, block=46), z=Z, seed=2)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.patch.set_facecolor("#0a0a0a")
    for ax, (name, cells) in zip(axes, patterns):
        net = condense(cells)
        draw_darkfield(ax, net, cells, show_cells=True)
        ax.set_title(f"DLP pattern: {name}\n({len(cells)} cells)",
                     color="white", fontsize=12, fontweight="bold")
        print(f"      {name:12s} {len(cells)} cells · {len(net.edges)} fibers")
    fig.suptitle("DLP-printed cell patterns template the ECM structure "
                 "(cells = teal, collagen = white)", color="white", fontweight="bold",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(os.path.join(FIG, "dlp_patterns.png"), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    density_study()
    dlp_study()
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
