#!/usr/bin/env python3
"""Mimic the spleen white-pulp reticular skeleton — vs a random control.

White pulp is organized around a central arteriole: a periarteriolar sheath
(PALS), B-cell follicles hanging off it, and a denser marginal-zone ring at the
boundary — all knit together by a reticular (FRC) collagen meshwork. We seed the
reticular cells in that layout, let them deposit + condense ECM, and compare the
resulting scaffold to the SAME number of randomly seeded cells (the control).

Rendered light-mode with a continuous blue→red colour scale for reinforcement
(constant line width). Quantified by the radial density profile (white pulp shows
structured rings; the control is flat), pore size, and connectivity.

  python run_whitepulp.py    →  output_whitepulp/figures/
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
OUT = os.path.join(HERE, "output_whitepulp")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0
c = DOMAIN / 2
CENTER = np.array([c, c, Z])

P = mech.MechParams(reach=24, reach_decay=15, planar=True, traction=1.5, max_iter=1100)
RP = rem.RemodelParams(n_steps=3, compaction=0.06, reinforce_rate=0.34,
                       reinforce_pct=60, slack_pct=34, slack_steps=2)
CMAP = "coolwarm"


WP_RADIUS = 68.0


def white_pulp_cells():
    """A bounded, organized compartment (how to mimic spleen white pulp):

      base       a uniform FRC reticular meshwork filling the white-pulp disk
      arteriole  a central line (the organizing axis)
      follicles  two dense spheroids (B-cell follicles) inside the disk
      marginal   a denser cell ring at the disk boundary (marginal zone)
    """
    base = seeding.in_mask(DOMAIN, 40, seeding.disk_mask(DOMAIN, r=WP_RADIUS), z=Z, seed=7)
    arteriole = seeding.line_cells([c, c - 24, Z], [c, c + 24, Z], n=6, seed=1)
    foll = np.concatenate([
        seeding.spheroid([c - 30, c + 20, Z], n_cells=11, radius=11, seed=3),
        seeding.spheroid([c + 32, c - 10, Z], n_cells=11, radius=11, seed=4),
    ])
    margin = seeding.annulus(CENTER, WP_RADIUS - 8, WP_RADIUS + 2, n=34, z=Z, seed=6)
    return np.concatenate([base, arteriole, foll, margin])


def condense(cells, seed=3):
    net0 = mech.build_deposited_network(cells, DOMAIN, fibers_per_cell=4,
                                        secretion_radius=14.0, crosslink_dist=3.5,
                                        z=Z, seed=seed)
    _, final = rem.remodel(net0, cells, P, RP)
    return final


def radial_density(net, nbins=16):
    r, dens = mm.density_profile(net, CENTER, DOMAIN, nbins=nbins)
    return r, dens


def main():
    os.makedirs(FIG, exist_ok=True)

    wp_cells = white_pulp_cells()
    n = len(wp_cells)
    ctrl_cells = seeding.uniform(DOMAIN, n=n, z=Z, margin=26, seed=9)
    print(f"white pulp: {n} cells  |  control: {len(ctrl_cells)} cells (matched)")

    print("condensing white-pulp scaffold …")
    wp = condense(wp_cells)
    print("condensing control …")
    ctrl = condense(ctrl_cells)

    # ---- comparison figure (light mode, gradient colour) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    # seeding map
    axes[0].set_facecolor("white")
    axes[0].scatter(wp_cells[:, 0], wp_cells[:, 1], s=12, c="#1186a8")
    axes[0].add_patch(plt.Circle((c, c), WP_RADIUS, fill=False, ls="--", color="#aaa"))
    axes[0].set_title("White-pulp seeding\n(arteriole + PALS + follicles + marginal zone)",
                      fontsize=11, fontweight="bold")
    axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_xlim(20, DOMAIN - 20); axes[0].set_ylim(20, DOMAIN - 20)

    vmax = max(wp.k_scale.max(), ctrl.k_scale.max(), 1.6)
    for ax, net, ttl in [(axes[1], wp, "White-pulp mimic — ECM scaffold"),
                         (axes[2], ctrl, "Random control — same cell count")]:
        render.draw_network(ax, net, theme="light", cmap=CMAP, lw=0.9, vmax=vmax,
                            show_cells=False)
        ax.set_xlim(20, DOMAIN - 20); ax.set_ylim(20, DOMAIN - 20)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
    render.add_colorbar(fig, axes[2], cmap=CMAP)
    fig.suptitle("Reproducing the spleen white-pulp reticular skeleton vs a random control",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "whitepulp_vs_control.png"), dpi=140)
    plt.close(fig)

    # ---- quantitative comparison ----------------------------------------
    rw, dw = radial_density(wp)
    rc, dc = radial_density(ctrl)
    pore_w = mm.pore_sizes(wp, DOMAIN); pore_c = mm.pore_sizes(ctrl, DOMAIN)

    def cv(d):
        d = np.asarray(d); return float(np.std(d) / (np.mean(d) + 1e-9))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(rw, dw, "o-", color="#d04b4b", label="white pulp")
    ax[0].plot(rc, dc, "s--", color="#2457ff", label="control")
    ax[0].axvspan(60, 70, color="#ffd27a", alpha=0.25, label="marginal zone")
    ax[0].set(title="Radial ECM density profile\n(white pulp = structured rings, control = flat)",
              xlabel="distance from center (µm)", ylabel="reinforced length / area")
    ax[0].grid(alpha=0.3); ax[0].legend(frameon=False, fontsize=9)

    ax[1].hist(pore_w, bins=40, range=(0, 40), histtype="step", lw=2.2, color="#d04b4b",
               label=f"white pulp (median {np.median(pore_w):.0f})")
    ax[1].hist(pore_c, bins=40, range=(0, 40), histtype="step", lw=2.2, color="#2457ff",
               label=f"control (median {np.median(pore_c):.0f})")
    ax[1].set(title="Pore-size distribution", xlabel="pore radius (µm)", ylabel="count")
    ax[1].grid(alpha=0.3); ax[1].legend(frameon=False, fontsize=9)

    labels = ["radial-profile\nstructure (CV)", "connectivity", "radial\nalignment"]
    wp_vals = [cv(dw), mm.connectivity(wp), mm.radial_alignment(wp, CENTER, reinforced_only=False)]
    ctrl_vals = [cv(dc), mm.connectivity(ctrl), mm.radial_alignment(ctrl, CENTER, reinforced_only=False)]
    x = np.arange(len(labels))
    ax[2].bar(x - 0.2, wp_vals, 0.4, color="#d04b4b", label="white pulp")
    ax[2].bar(x + 0.2, ctrl_vals, 0.4, color="#2457ff", label="control")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9)
    ax[2].set_title("Organization metrics"); ax[2].grid(alpha=0.3, axis="y")
    ax[2].legend(frameon=False, fontsize=9)

    fig.suptitle("White-pulp mimic vs control — quantitative comparison", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "whitepulp_metrics.png"), dpi=140)
    plt.close(fig)

    print(f"  radial-profile structure (CV): white pulp {cv(dw):.2f} vs control {cv(dc):.2f}")
    print(f"  connectivity: {mm.connectivity(wp):.2f} vs {mm.connectivity(ctrl):.2f}")
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
