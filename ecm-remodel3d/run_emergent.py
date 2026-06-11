#!/usr/bin/env python3
"""Emergent cotton-candy ECM: how should spheroids be placed to self-organize a
spleen-white-pulp-like divergent reticular network?

The structure is NOT painted by the seeding. Spheroids are placed simply in a
continuous collagen matrix (the GelMA collagen); their contraction pulls the
matrix into radially divergent struts and inter-spheroid bridges, and pruning of
the unloaded matrix leaves an emergent strut-and-pore reticular web. We sweep the
NUMBER and SPACING of spheroids and auto-score each layout for how spleen-like
the emergent network is (branching + connectivity + spanning coverage).

  python run_emergent.py    →  output_emergent/figures/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding
from ecm_pipeline import morpho_metrics as mm, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_emergent")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0
c = DOMAIN / 2

P = mech.MechParams(reach=26, reach_decay=16, planar=True, traction=1.6, max_iter=1200)
RP = rem.RemodelParams(n_steps=6, compaction=0.05, reinforce_rate=0.3,
                       reinforce_pct=60, slack_pct=33, slack_steps=2)


def gel(seed=4, n_fibers=230):
    """Continuous collagen matrix the spheroids are embedded in."""
    return mech.build_random_network(DOMAIN, n_fibers=n_fibers, fiber_len=64,
                                     seg_len=6.0, crosslink_dist=4.2,
                                     seed=seed, slab_thickness=0.0)


def ring_centers(k, R=58.0, with_center=False):
    if k == 1:
        return [np.array([c, c, Z])]
    pts = []
    if with_center:
        pts.append(np.array([c, c, Z])); k -= 1
    for i in range(k):
        a = 2 * np.pi * i / k + np.pi / 2
        pts.append(np.array([c + R * np.cos(a), c + R * np.sin(a), Z]))
    return pts


def emergent(centers, n_cells=14, radius=12.0):
    cells = seeding.spheroids(centers, n_cells=n_cells, radius=radius, seed=1)
    net0 = gel()
    _, final = rem.remodel(net0, cells, P, RP)
    return final, cells


def metrics(net):
    area = (DOMAIN - 40) ** 2
    return dict(
        branch=mm.branch_points(net),
        connectivity=mm.connectivity(net),
        coverage=mm.coverage(net, DOMAIN, radius=12),
        pore=float(np.median(mm.pore_sizes(net, DOMAIN))),
    )


def darkfield(ax, net, cells=None):
    """Light-mode render with the blue→red reinforcement colour scale."""
    render.draw_network(ax, net, theme="light", cmap="coolwarm", lw=0.9,
                        show_cells=False)
    if cells is not None:
        ax.scatter(cells[:, 0], cells[:, 1], s=5, c="#0d3b4f", alpha=0.8,
                   edgecolors="none", zorder=6)
    ax.set_xlim(15, DOMAIN - 15); ax.set_ylim(15, DOMAIN - 15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def composite(rows):
    """Normalize metrics across layouts → spleen-likeness score (0..1)."""
    def norm(key):
        vals = np.array([r[key] for r in rows], float)
        lo, hi = vals.min(), vals.max()
        return (vals - lo) / (hi - lo + 1e-9)
    nb, nc, ncov = norm("branch"), [r["connectivity"] for r in rows], norm("coverage")
    score = 0.34 * np.array(nc) + 0.33 * ncov + 0.33 * nb
    for r, s in zip(rows, score):
        r["score"] = float(s)
    return rows


def count_sweep():
    print("[1/2] spheroid-count sweep …")
    layouts = [(2, ring_centers(2, R=46)), (3, ring_centers(3, R=52)),
               (4, ring_centers(4, R=56)), (6, ring_centers(6, R=60)),
               (9, ring_centers(8, R=64, with_center=True))]
    nets, cellss, rows = [], [], []
    for k, ctr in layouts:
        net, cells = emergent(ctr)
        m = metrics(net); m["k"] = k
        nets.append(net); cellss.append(cells); rows.append(m)
        print(f"      {k} spheroids: branch={m['branch']} conn={m['connectivity']:.2f} "
              f"coverage={m['coverage']:.2f} pore={m['pore']:.1f}")
    rows = composite(rows)
    best = int(np.argmax([r["score"] for r in rows]))

    fig, axes = plt.subplots(1, len(layouts), figsize=(16, 4.4))
    for i, (ax, net, cells, m) in enumerate(zip(axes, nets, cellss, rows)):
        darkfield(ax, net, cells)
        star = "  ★ best" if i == best else ""
        ax.set_title(f"{m['k']} spheroids\nspleen-score {m['score']:.2f}{star}",
                     color=("#c47f00" if i == best else "black"), fontsize=11, fontweight="bold")
        if i == best:
            for s in ax.spines.values():
                s.set_color("#e8a000"); s.set_linewidth(2.5); s.set_visible(True)
    fig.suptitle("Emergent reticular network vs spheroid number — auto-scored for spleen-likeness",
                 fontweight="bold", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 0.91, 0.86])
    render.add_colorbar(fig, cmap="coolwarm")
    fig.savefig(os.path.join(FIG, "emergent_count.png"), dpi=140)
    plt.close(fig)

    # score / metrics figure
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(ks, [r["score"] for r in rows], "o-", color="#d04b4b", lw=2)
    ax[0].plot(ks[best], rows[best]["score"], "*", ms=20, color="#ffb000")
    ax[0].set(title="Spleen-likeness score vs spheroid count", xlabel="number of spheroids", ylabel="composite score")
    ax[0].grid(alpha=0.3)
    w = 0.25; x = np.arange(len(ks))
    ax[1].bar(x - w, [r["connectivity"] for r in rows], w, label="connectivity", color="#2457ff")
    ax[1].bar(x, mm_norm([r["coverage"] for r in rows]), w, label="coverage (norm)", color="#00a8a8")
    ax[1].bar(x + w, mm_norm([r["branch"] for r in rows]), w, label="branching (norm)", color="#8948ff")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{k}" for k in ks])
    ax[1].set(title="Component metrics", xlabel="number of spheroids", ylabel="0..1")
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("How spheroid number shapes the emergent reticular structure", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "emergent_scores.png"), dpi=140)
    plt.close(fig)
    print(f"      → most spleen-like: {rows[best]['k']} spheroids (score {rows[best]['score']:.2f})")


def mm_norm(vals):
    vals = np.array(vals, float)
    return (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)


def spacing_sweep():
    print("[2/2] spacing sweep (fixed 4 spheroids) …")
    spacings = [(34, "tight"), (52, "medium"), (72, "wide")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (R, name) in zip(axes, spacings):
        net, cells = emergent(ring_centers(4, R=R))
        m = metrics(net)
        darkfield(ax, net, cells)
        ax.set_title(f"{name} spacing (R={R})\nbranch {m['branch']} · conn {m['connectivity']:.2f} · "
                     f"pore {m['pore']:.0f}µm", color="black", fontsize=11, fontweight="bold")
        print(f"      {name}: branch={m['branch']} conn={m['connectivity']:.2f} pore={m['pore']:.1f}")
    fig.suptitle("Spheroid spacing: too close = one clump, too far = isolated asters, "
                 "intermediate = connected web", fontweight="bold", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 0.91, 0.88])
    render.add_colorbar(fig, cmap="coolwarm")
    fig.savefig(os.path.join(FIG, "emergent_spacing.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    count_sweep()
    spacing_sweep()
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
