#!/usr/bin/env python3
"""Clean inter-focus graphs in a floating gel (rim-anchored, local secretion).

Refines the free-gel result into clean geometric strut graphs:
  - anchor ONLY a thin outer rim (interior fully free → foci pull on each other),
  - cells secrete locally (small secretion radius) so fibers start near the foci,
  - traction reach spans the inter-focus distance so bridges carry the load,
  - selective reinforcement so only the load-bearing bridges mature into struts.

  python run_cleangraph.py    →  output_cleangraph/figures/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_cleangraph")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0
c = DOMAIN / 2

P = mech.MechParams(reach=34, reach_decay=20, planar=True, traction=1.9, max_iter=1100)
RP = rem.RemodelParams(n_steps=4, compaction=0.06, reinforce_rate=0.4,
                       reinforce_pct=74, slack_pct=42, slack_steps=2)
PIN = 16.0   # the matrix is held ONLY at the foci (within this radius of a cell)


def build(cells):
    # continuous floating gel held ONLY at the foci: with nothing else anchored,
    # the matrix between foci condenses into clean bridges (no outward spikes,
    # because there is nothing outside to pull toward) — a closed, Stopak-Harris-
    # like floating gel.
    net = mech.build_random_network(DOMAIN, n_fibers=150, fiber_len=64, seg_len=6.0,
                                    crosslink_dist=4.2, seed=4, slab_thickness=0.0)
    d = np.min(np.linalg.norm(net.nodes[:, None, :2] - cells[None, :, :2], axis=2), axis=1)
    net.fixed[:] = d < PIN
    return net


def emergent(cells):
    _, net = rem.remodel(build(cells), cells, P, RP)
    return net


def foci(pts, n_cells=20, radius=11):
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
            ax.plot([x0, x1], [y0, y1], color="#f0b000", lw=1.3, ls="--", zorder=1)
    ax.scatter(cells[:, 0], cells[:, 1], s=10, c="#0d3b4f", alpha=0.9, zorder=6)
    ax.set_xlim(15, DOMAIN - 15); ax.set_ylim(15, DOMAIN - 15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def graphs():
    return {
        "triangle (3 foci)": (polygon(3, 50), True),
        "square (4 foci)": (polygon(4, 52, np.pi / 4), True),
        "pentagon (5 foci)": (polygon(5, 54), True),
        "hub + 5 spokes": ([(c, c)] + polygon(5, 56), False),
    }


def main():
    os.makedirs(FIG, exist_ok=True)
    items = graphs()
    results = []
    for name, (pts, ring) in items.items():
        cells = foci(pts)
        if ring:
            target = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        else:  # hub-spoke: center to each rim point
            target = [(pts[0], pts[i]) for i in range(1, len(pts))]
        print(f"  {name}: {len(pts)} foci, {len(cells)} cells")
        results.append((name, emergent(cells), cells, target))

    vmax = max(max(net.k_scale.max() for _, net, _, _ in results), 1.6)  # shared colour scale
    fig, axes = plt.subplots(1, 4, figsize=(16.6, 4.6))
    for ax, (name, net, cells, target) in zip(axes, results):
        draw(ax, net, cells, name, target=target, vmax=vmax)
    render.add_colorbar(fig, axes[-1], cmap="coolwarm")
    fig.suptitle("Clean inter-focus strut graphs (floating gel held only at the foci) — "
                 "orange dashed = intended edges", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "clean_graphs.png"), dpi=145); plt.close(fig)
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
