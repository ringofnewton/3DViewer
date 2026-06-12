#!/usr/bin/env python3
"""Boundary condition matters: anchored gel vs floating (free) gel.

In a wall-ANCHORED gel, contractile foci pull struts toward the rigid boundary,
so an aster network dominates and arbitrary geometric graphs do not form. In a
FLOATING gel (no wall anchoring; only sparse peripheral adhesion), foci pull on
each other, so inter-focus BRIDGES dominate and a cleaner geometric strut graph
emerges between the foci (the classic two-explant bridge of Stopak & Harris).

We compare the two boundary conditions for the same foci layouts (triangle,
square), so you can see which culture condition enables designed connectivity.

  python run_freegel.py    →  output_freegel/figures/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_freegel")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0
c = DOMAIN / 2

P = mech.MechParams(reach=30, reach_decay=18, planar=True, traction=1.8, max_iter=1000)
RP = rem.RemodelParams(n_steps=4, compaction=0.06, reinforce_rate=0.36,
                       reinforce_pct=73, slack_pct=40, slack_steps=2)


def anchored_gel(seed=4, nf=150):
    """Default build = a band of fixed nodes around the wall."""
    return mech.build_random_network(DOMAIN, n_fibers=nf, fiber_len=64, seg_len=6.0,
                                     crosslink_dist=4.2, seed=seed, slab_thickness=0.0)


def free_gel(seed=4, nf=150, frac=0.10, r_periph=80.0):
    """Floating gel: NO wall band; only sparse anchors in the outer periphery,
    so the interior (where the foci sit) is free and foci pull on each other."""
    net = anchored_gel(seed, nf)
    r = np.linalg.norm(net.nodes[:, :2] - c, axis=1)
    rng = np.random.default_rng(7)
    net.fixed[:] = (rng.random(len(net.nodes)) < frac) & (r > r_periph)
    return net


def foci(pts, n_cells=18, radius=12):
    return seeding.spheroids([np.array([x, y, Z]) for x, y in pts],
                             n_cells=n_cells, radius=radius, seed=1)


def polygon(n, R, rot=np.pi / 2):
    a = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(c + R * np.cos(t), c + R * np.sin(t)) for t in a]


def draw(ax, net, cells, title=None, target=None, vmax=None):
    # shared renderer: constant line width + continuous blue→red gradient
    render.draw_network(ax, net, theme="light", cmap="coolwarm", lw=1.1,
                        vmax=vmax, show_cells=False)
    if target is not None:
        for (x0, y0), (x1, y1) in target:
            ax.plot([x0, x1], [y0, y1], color="#f0b000", lw=1.2, ls="--", zorder=1)
    ax.scatter(cells[:, 0], cells[:, 1], s=8, c="#0d3b4f", alpha=0.85, zorder=6)
    ax.set_xlim(15, DOMAIN - 15); ax.set_ylim(15, DOMAIN - 15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def main():
    os.makedirs(FIG, exist_ok=True)
    layouts = [("triangle", polygon(3, 54)), ("square", polygon(4, 56, np.pi / 4))]

    rows = []
    for name, pts in layouts:
        cells = foci(pts)
        edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        print(f"  {name}: {len(pts)} foci, {len(cells)} cells")
        _, na = rem.remodel(anchored_gel(), cells, P, RP)
        _, nf = rem.remodel(free_gel(), cells, P, RP)
        rows.append((name, cells, edges, na, nf))

    vmax = max(max(max(na.k_scale.max(), nf.k_scale.max()) for _, _, _, na, nf in rows), 1.6)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    for r, (name, cells, edges, na, nf) in enumerate(rows):
        draw(axes[r, 0], na, cells, f"{name} — ANCHORED gel\n(struts radiate to walls)",
             target=edges, vmax=vmax)
        draw(axes[r, 1], nf, cells, f"{name} — FREE/floating gel\n(bridges between foci)",
             target=edges, vmax=vmax)
    fig.suptitle("Boundary condition decides what you can build: free gel → "
                 "clean inter-focus geometric graph", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    render.add_colorbar(fig, cmap="coolwarm")
    fig.savefig(os.path.join(FIG, "freegel_vs_anchored.png"), dpi=145); plt.close(fig)
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
