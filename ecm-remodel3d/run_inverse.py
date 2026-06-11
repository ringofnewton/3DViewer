#!/usr/bin/env python3
"""Inverse design: where to seed cells to make a TARGET ECM pattern.

The emergent rule is: struts form as BRIDGES between cell foci, and pores stay
where there are no cells. So you design a pattern by placing foci at the strut
junctions/endpoints — not by tracing the struts. Three explorations:

  inverse_principle.png   outline seeding (literal) vs vertex-foci (emergent edges)
  inverse_foci_graph.png  foci at vertices → triangle / square / star / hub-spoke
  inverse_bridge_rule.png how far apart two foci can be and still bridge

Light mode, blue→red reinforcement colour. Foci drawn as dark dots.

  python run_inverse.py    →  output_inverse/figures/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_inverse")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0
c = DOMAIN / 2

# Strong traction + reinforce only the TOP-tension fibers, so the inter-focus
# bridges (which carry the most tension) reinforce into clean struts while the
# weaker radial background stays faint.
P = mech.MechParams(reach=30, reach_decay=17, planar=True, traction=1.9, max_iter=900)
RP = rem.RemodelParams(n_steps=4, compaction=0.06, reinforce_rate=0.36,
                       reinforce_pct=76, slack_pct=40, slack_steps=2)


def gel(seed=4, n_fibers=140):
    return mech.build_random_network(DOMAIN, n_fibers=n_fibers, fiber_len=64,
                                     seg_len=6.0, crosslink_dist=4.2, seed=seed,
                                     slab_thickness=0.0)


def emergent(cells, gel_seed=4):
    _, final = rem.remodel(gel(gel_seed), cells, P, RP)
    return final


def foci(centers, n_cells=16, radius=12.0):
    return seeding.spheroids([np.array([x, y, Z]) for x, y in centers],
                             n_cells=n_cells, radius=radius, seed=1)


def draw(ax, net, cells, title=None, target=None):
    """High-contrast: reinforced struts bold red, background fibers very faint."""
    ax.set_facecolor("white")
    k = net.k_scale
    kmax = max(k.max(), 1.6)
    cmap = plt.get_cmap("coolwarm")
    for idx in np.argsort(k):
        e = net.edges[idx]
        pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        v = min(max((k[idx] - 1) / (kmax - 1), 0), 1)
        if v > 0.12:                              # reinforced strut
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cmap(0.5 + 0.5 * v),
                    lw=0.7 + 1.8 * v, alpha=0.95, solid_capstyle="round", zorder=4)
        else:                                     # faint background
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#cdd6e3",
                    lw=0.4, alpha=0.5, solid_capstyle="round", zorder=2)
    if target is not None:                       # faint guide of the intended pattern
        for (x0, y0), (x1, y1) in target:
            ax.plot([x0, x1], [y0, y1], color="#f0b000", lw=1.2, ls="--", zorder=1)
    ax.scatter(cells[:, 0], cells[:, 1], s=7, c="#0d3b4f", alpha=0.85, zorder=6,
               edgecolors="none")
    ax.set_xlim(15, DOMAIN - 15); ax.set_ylim(15, DOMAIN - 15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def polygon_pts(n, R, rot=np.pi / 2):
    a = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(c + R * np.cos(t), c + R * np.sin(t)) for t in a]


def reinforced_band_length(net, a, b, band=22.0):
    sel = net.k_scale > 1.4
    if sel.sum() < 1:
        return 0.0
    mid = 0.5 * (net.nodes[net.edges[sel, 0]] + net.nodes[net.edges[sel, 1]])[:, :2]
    a, b = np.array(a), np.array(b)
    ab = b - a
    t = np.clip(((mid - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
    perp = np.linalg.norm(mid - (a + t[:, None] * ab), axis=1)
    d = net.nodes[net.edges[sel, 1]] - net.nodes[net.edges[sel, 0]]
    return float(np.sum(np.linalg.norm(d, axis=1)[perp < band]))


# --------------------------------------------------------------------------- #
def study_principle():
    print("[1/3] outline vs vertex-foci …")
    R = 52
    verts = polygon_pts(3, R)
    outline = seeding.polygon_outline([c, c, Z], r=R, n_sides=3, per_side=20,
                                      rotation=np.pi / 2, seed=1)
    vtx = foci(verts, n_cells=20, radius=13)
    edges = [(verts[i], verts[(i + 1) % 3]) for i in range(3)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    draw(axes[0], emergent(outline), outline,
         "Seed cells ALONG the outline\n→ ECM just copies the seeding (trivial)")
    draw(axes[1], emergent(vtx), vtx,
         "Seed foci at the 3 VERTICES only\n→ radial asters + bridges, NOT a clean triangle",
         target=edges)
    fig.suptitle("The limit: in an anchored continuum, foci make radial asters + "
                 "inter-focus bridges — arbitrary clean lines do NOT transfer",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "inverse_principle.png"), dpi=140); plt.close(fig)


def study_pore():
    print("[2/3] pore control (cell-free regions) …")
    layouts = {
        "ring of foci\n→ one central pore": polygon_pts(8, 46),
        "wider ring\n→ bigger central pore": polygon_pts(8, 64),
        "3×3 grid of foci\n→ lattice of pores": [(c + dx, c + dy)
            for dx in (-46, 0, 46) for dy in (-46, 0, 46)],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (name, pts) in zip(axes, layouts.items()):
        cells = foci(pts, n_cells=12, radius=10)
        print(f"      {name.splitlines()[0]}: {len(pts)} foci, {len(cells)} cells")
        draw(ax, emergent(cells), cells, name)
    fig.suptitle("What you CAN control cleanly: pores. Cell-free regions stay open; "
                 "foci surround / tile them", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "inverse_pore.png"), dpi=140); plt.close(fig)


def study_bridge_rule():
    print("[3/3] bridge-length rule …")
    gaps = [40, 60, 80, 100, 120]
    lengths = []
    panels = []
    for g in gaps:
        a, b = (c - g / 2, c), (c + g / 2, c)
        cells = foci([a, b], n_cells=18, radius=12)
        net = emergent(cells)
        L = reinforced_band_length(net, a, b)
        lengths.append(L); panels.append((net, cells, g, L))
        print(f"      gap={g}µm: bridge reinforced length={L:.0f}")

    fig = plt.figure(figsize=(16, 6))
    for i, (net, cells, g, L) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1)
        draw(ax, net, cells, f"gap = {g} µm · bridge {L:.0f}")
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(gaps, lengths, "o-", color="#d04b4b", lw=2)
    ax.set(title="Inter-focus bridge vs distance", xlabel="focus spacing (µm)",
           ylabel="reinforced length in band (µm)")
    ax.grid(alpha=0.3)
    fig.suptitle("How far apart two foci can be and still build a connecting strut",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "inverse_bridge_rule.png"), dpi=140); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    study_principle()
    study_pore()
    study_bridge_rule()
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
