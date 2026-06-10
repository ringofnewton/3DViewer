#!/usr/bin/env python3
"""Grow ECM into a chosen cell-seeding pattern, over time.

Seed cells along any geometric pattern; they deposit collagen locally and then
contract/reinforce it, so over time the ECM matures into that pattern (a faint
diffuse matrix → sharp reinforced struts along the design). Three studies:

  1. patterns_grid.png      regularly aligned seeds → a lattice ECM, over time
  2. patterns_shapes.png    circle / triangle / square outlines → shaped ECM
  3. patterns_complex.png   a complex geometric motif (honeycomb) → ECM

Light mode, continuous blue→red reinforcement colour, constant line width.

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

P = mech.MechParams(reach=22, reach_decay=14, planar=True, traction=1.5, max_iter=1100)
RP = rem.RemodelParams(n_steps=4, compaction=0.06, reinforce_rate=0.36,
                       reinforce_pct=58, slack_pct=34, slack_steps=2)


def timecourse(cells, fibers_per_cell=5, secretion_radius=15.0):
    net0 = mech.build_deposited_network(cells, DOMAIN, fibers_per_cell=fibers_per_cell,
                                        secretion_radius=secretion_radius,
                                        crosslink_dist=3.6, z=Z, seed=3)
    frames, _ = rem.remodel(net0, cells, P, RP)
    seq = [net0] + [f["net"] for f in frames]
    vmax = max(max(n.k_scale.max() for n in seq), 1.6)
    return seq, vmax


def montage(seq, vmax, axes, row=None):
    n = len(seq)
    cols = np.linspace(0, n - 1, axes.shape[-1] if axes.ndim > 1 else len(axes), dtype=int)
    row_axes = axes[row] if row is not None else axes
    for ax, ci in zip(row_axes, cols):
        render.draw_network(ax, seq[ci], theme="light", cmap=CMAP, lw=0.9,
                            vmax=vmax, show_cells=False)
        ax.set_xlim(15, DOMAIN - 15); ax.set_ylim(15, DOMAIN - 15)
        if row in (None, 0):
            ax.set_title("t = 0 (seeded)" if ci == 0 else f"t = {ci}",
                         fontsize=11, fontweight="bold")
    return cols


def study_grid():
    print("[1/3] regular aligned grid …")
    cells = seeding.grid_lattice(DOMAIN, spacing=30, margin=34, z=Z)
    seq, vmax = timecourse(cells, fibers_per_cell=6, secretion_radius=17)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    montage(seq, vmax, axes)
    fig.suptitle(f"1 · Regularly aligned seeds ({len(cells)} cells) → lattice ECM growing over time",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "patterns_grid.png"), dpi=140)
    plt.close(fig)


def study_shapes():
    print("[2/3] geometric shapes …")
    shapes = {
        "circle": seeding.circle_outline([c, c, Z], r=66, n=70, seed=1),
        "triangle": seeding.polygon_outline([c, c, Z], r=74, n_sides=3, per_side=22, rotation=np.pi/2, seed=1),
        "square": seeding.polygon_outline([c, c, Z], r=76, n_sides=4, per_side=18, rotation=np.pi/4, seed=1),
    }
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for r, (name, cells) in enumerate(shapes.items()):
        print(f"      {name}: {len(cells)} cells")
        seq, vmax = timecourse(cells)
        montage(seq, vmax, axes, row=r)
        axes[r, 0].set_ylabel(name, fontsize=13, fontweight="bold")
    fig.suptitle("2 · Geometric seeding patterns → ECM grows into the shape over time",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "patterns_shapes.png"), dpi=135)
    plt.close(fig)


def study_complex():
    print("[3/3] complex geometric motif (honeycomb) …")
    segs = seeding.hex_lattice_segments(DOMAIN, a=30, margin=30)
    cells = seeding.along_segments(segs, spacing=10, z=Z, jitter=1.4, seed=1)
    print(f"      {len(cells)} cells, {len(segs)} edges")
    seq, vmax = timecourse(cells, fibers_per_cell=4, secretion_radius=13)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    montage(seq, vmax, axes)
    fig.suptitle(f"3 · Complex geometric pattern ({len(cells)} cells) → ECM grows into the motif",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "patterns_complex.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    study_grid()
    study_shapes()
    study_complex()
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
