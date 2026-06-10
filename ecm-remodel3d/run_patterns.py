#!/usr/bin/env python3
"""Grow ECM into a chosen cell-seeding pattern, OVER TIME (cells secrete it).

t = 0 shows only the SEEDED CELLS (how you laid them down); then the cells
secrete collagen gradually and contract it, so the ECM grows into the pattern.
Three studies:

  1. patterns_grid.png      regularly aligned seeds → a lattice ECM
  2. patterns_shapes.png    circle / triangle / square outlines → shaped ECM
  3. patterns_text.png      a text pattern ("REZA ABDI LAB") → ECM in that shape

Light mode, continuous blue→red reinforcement colour, constant line width;
seeded cells drawn as dark dots at every frame.

  python run_patterns.py    →  output_patterns/figures/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_patterns")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 240.0, 120.0
c = DOMAIN / 2
CMAP = "coolwarm"

P = mech.MechParams(reach=20, reach_decay=13, planar=True, traction=1.5, max_iter=1100)
RP = rem.RemodelParams(n_steps=4, compaction=0.06, reinforce_rate=0.4,
                       reinforce_pct=58, slack_pct=34, slack_steps=2)
N_STEPS = 4   # growth steps (montage columns = N_STEPS + 1, incl. t=0 cells-only)


def grow(cells, fibers_per_cell=5, secretion_radius=14.0, crosslink_dist=3.6,
         p_mech=None, p_rem=None):
    net_full, birth = mech.build_deposited_network(
        cells, DOMAIN, fibers_per_cell=fibers_per_cell, secretion_radius=secretion_radius,
        crosslink_dist=crosslink_dist, z=Z, seed=3, return_birth=True)
    frames = rem.grow_and_remodel(net_full, birth, cells, p_mech or P, p_rem or RP,
                                  n_steps=N_STEPS)
    vmax = max([f.k_scale.max() for f in frames if f is not None] + [1.6])
    return frames, vmax


def draw_frame(ax, frame, cells, vmax, title=None):
    ax.set_facecolor("white")
    if frame is not None:
        render.draw_network(ax, frame, theme="light", cmap=CMAP, lw=0.9, vmax=vmax,
                            show_cells=False)
    ax.scatter(cells[:, 0], cells[:, 1], s=5, c="#0d3b4f", alpha=0.85, zorder=6,
               edgecolors="none")
    ax.set_xlim(12, DOMAIN - 12); ax.set_ylim(12, DOMAIN - 12)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def montage_row(axes_row, frames, cells, vmax, header=False):
    for j, (ax, frame) in enumerate(zip(axes_row, frames)):
        title = ("t = 0\n(seeded cells)" if j == 0 else f"t = {j}") if header else None
        draw_frame(ax, frame, cells, vmax, title)


def study_grid():
    print("[1/3] regular aligned grid …")
    cells = seeding.grid_lattice(DOMAIN, spacing=30, margin=34, z=Z)
    frames, vmax = grow(cells, fibers_per_cell=6, secretion_radius=17)
    fig, axes = plt.subplots(1, N_STEPS + 1, figsize=(16, 3.7))
    montage_row(axes, frames, cells, vmax, header=True)
    fig.suptitle(f"1 · Regularly aligned seeds ({len(cells)} cells) → ECM grows into a lattice",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "patterns_grid.png"), dpi=140); plt.close(fig)


def study_shapes():
    print("[2/3] geometric shapes …")
    shapes = {
        "circle": seeding.circle_outline([c, c, Z], r=66, n=70, seed=1),
        "triangle": seeding.polygon_outline([c, c, Z], r=74, n_sides=3, per_side=22, rotation=np.pi/2, seed=1),
        "square": seeding.polygon_outline([c, c, Z], r=76, n_sides=4, per_side=18, rotation=np.pi/4, seed=1),
    }
    fig, axes = plt.subplots(3, N_STEPS + 1, figsize=(16, 10))
    for r, (name, cells) in enumerate(shapes.items()):
        print(f"      {name}: {len(cells)} cells")
        frames, vmax = grow(cells)
        montage_row(axes[r], frames, cells, vmax, header=(r == 0))
        axes[r, 0].set_ylabel(name, fontsize=13, fontweight="bold")
    fig.suptitle("2 · Geometric seeding patterns → ECM grows into the shape over time",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "patterns_shapes.png"), dpi=135); plt.close(fig)


def study_text():
    print("[3/3] text pattern (REZA ABDI LAB) …")
    cells = seeding.text_points(["REZA", "ABDI", "LAB"], domain=DOMAIN, n=320, z=Z,
                                fontsize=58, seed=1)
    print(f"      {len(cells)} cells")
    # gentle, LOCAL traction so the matrix stays on the glyphs (no clumping)
    p_text = mech.MechParams(reach=13, reach_decay=9, planar=True, traction=1.1, max_iter=1000)
    rp_text = rem.RemodelParams(n_steps=4, compaction=0.04, reinforce_rate=0.4,
                                reinforce_pct=55, slack_pct=30, slack_steps=2)
    frames, vmax = grow(cells, fibers_per_cell=2, secretion_radius=8, crosslink_dist=2.4,
                        p_mech=p_text, p_rem=rp_text)
    fig, axes = plt.subplots(1, N_STEPS + 1, figsize=(16, 4.4))
    montage_row(axes, frames, cells, vmax, header=True)
    fig.suptitle("3 · Custom pattern (REZA ABDI LAB) → ECM grows into the design",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "patterns_text.png"), dpi=140); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    study_grid()
    study_shapes()
    study_text()
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
