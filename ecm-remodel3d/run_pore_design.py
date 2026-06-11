#!/usr/bin/env python3
"""Pore-lattice design: place cell foci so a TARGET pore structure emerges.

Cell-free regions stay open, so foci on a lattice tile the space with pores and
the struts condense between them. The foci SPACING sets the pore size/period, so
you can design a target pore lattice (and read off the spacing from a calibration
curve). Square and honeycomb foci arrangements are shown.

  pore_lattice.png   square foci lattice at 3 spacings → pore lattices
  pore_curve.png     foci spacing → measured pore size (the design rule)
  pore_hex.png       honeycomb foci → hexagonal pore lattice

  python run_pore_design.py    →  output_pore/figures/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding
from ecm_pipeline import morpho_metrics as mm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_pore")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0

P = mech.MechParams(reach=26, reach_decay=15, planar=True, traction=1.7, max_iter=850)
RP = rem.RemodelParams(n_steps=3, compaction=0.06, reinforce_rate=0.36,
                       reinforce_pct=72, slack_pct=40, slack_steps=2)


def gel(seed=4, n_fibers=140):
    return mech.build_random_network(DOMAIN, n_fibers=n_fibers, fiber_len=64,
                                     seg_len=6.0, crosslink_dist=4.2, seed=seed,
                                     slab_thickness=0.0)


def emergent(centers, n_cells=10, radius=9):
    cells = seeding.spheroids([np.array([x, y, Z]) for x, y in centers],
                              n_cells=n_cells, radius=radius, seed=1)
    _, net = rem.remodel(gel(), cells, P, RP)
    return net, cells


def square_foci(spacing, n=4):
    """A fixed n×n grid centered in the domain; spacing sets the period."""
    total = (n - 1) * spacing
    start = (DOMAIN - total) / 2
    xs = start + np.arange(n) * spacing
    return [(x, y) for x in xs for y in xs], start, start + total


def hex_foci(spacing, margin=40.0):
    h = spacing * np.sqrt(3) / 2
    pts, row = [], 0
    y = margin
    while y < DOMAIN - margin:
        x0 = margin + (spacing / 2 if row % 2 else 0)
        x = x0
        while x < DOMAIN - margin:
            pts.append((x, y)); x += spacing
        y += h; row += 1
    return pts


def draw(ax, net, cells, title=None):
    ax.set_facecolor("white")
    k = net.k_scale; kmax = max(k.max(), 1.6)
    cmap = plt.get_cmap("coolwarm")
    for idx in np.argsort(k):
        e = net.edges[idx]; pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        v = min(max((k[idx] - 1) / (kmax - 1), 0), 1)
        if v > 0.12:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cmap(0.5 + 0.5 * v),
                    lw=0.7 + 1.7 * v, alpha=0.95, zorder=4, solid_capstyle="round")
        else:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#cdd6e3", lw=0.4,
                    alpha=0.5, zorder=2)
    ax.scatter(cells[:, 0], cells[:, 1], s=6, c="#0d3b4f", alpha=0.8, zorder=6,
               edgecolors="none")
    ax.set_xlim(15, DOMAIN - 15); ax.set_ylim(15, DOMAIN - 15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def main():
    os.makedirs(FIG, exist_ok=True)

    print("[1/2] 4×4 foci grid at several spacings …")
    spacings = [34, 44, 54, 64]
    nets, cellss, pores = [], [], []
    for s in spacings:
        centers, lo, hi = square_foci(s)
        net, cells = emergent(centers)
        p = float(np.median(mm.pore_sizes(net, DOMAIN, lo=lo, hi=hi)))  # interior only
        nets.append(net); cellss.append(cells); pores.append(p)
        print(f"      spacing={s}µm  foci={len(centers)}  median pore={p:.1f}µm")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, s, net, cells, p in zip(axes, spacings, nets, cellss, pores):
        draw(ax, net, cells, f"foci spacing {s} µm\n→ median pore {p:.0f} µm")
    fig.suptitle("Pore lattice designed by foci spacing — tighter foci, smaller pores",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "pore_lattice.png"), dpi=140); plt.close(fig)

    # design rule curve
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(spacings, pores, "o-", color="#d04b4b", lw=2)
    fit = np.polyfit(spacings, pores, 1)
    ax.plot(spacings, np.polyval(fit, spacings), "--", color="#888",
            label=f"pore ≈ {fit[0]:.2f}·spacing + {fit[1]:.0f}")
    ax.set(title="Design rule: foci spacing → resulting pore size",
           xlabel="foci spacing (µm)", ylabel="median pore radius (µm)")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pore_curve.png"), dpi=140); plt.close(fig)

    print("[2/2] honeycomb foci → hexagonal pore lattice …")
    net, cells = emergent(hex_foci(54), n_cells=10, radius=9)
    ph = float(np.median(mm.pore_sizes(net, DOMAIN)))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw(ax, net, cells, f"honeycomb foci (spacing 54 µm)\n→ hexagonal pore lattice, "
                         f"median pore {ph:.0f} µm")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pore_hex.png"), dpi=140); plt.close(fig)

    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
