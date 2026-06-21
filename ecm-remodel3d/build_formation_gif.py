#!/usr/bin/env python3
"""Animated GIF of ECM network FORMATION — cells growing collagen outward.

Uses the real growth model (ecm_network): cells secrete fibers that advance,
branch and crosslink, so a connected network emerges OUTWARD from the cells and
bridges form between them. GIFs render inline on mobile (no file to open).

    python build_formation_gif.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from ecm_pipeline import ecm_network as net

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output_validation", "figures", "ecm_formation.gif")


def main():
    p = net.NetworkParams(domain=150.0, n_cells=20, steps=92, n_frames=34,
                          seg_length=2.8, branch_prob=0.09, crosslink_dist=5.0,
                          max_tips=420, max_nodes=14000, persistence=0.80,
                          align_radius=24.0, traction_pull=0.08, seed=5)
    frames = net.simulate_network(p)[0]
    D = p.domain

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    def draw(i):
        ax.clear()
        ax.set_xlim(0, D); ax.set_ylim(0, D); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f7fafc")
        fr = frames[i]
        nodes = np.asarray(fr["nodes"], float)
        edges = fr["edges"]
        if len(edges):
            segs, cols = [], []
            for e in edges:
                a, b, st = e[0], e[1], e[2]
                segs.append([(nodes[a, 0], nodes[a, 1]), (nodes[b, 0], nodes[b, 1])])
                cols.append("#94a3b8" if st == "crosslink" else "#2563eb")
            lc = LineCollection(segs, colors=cols,
                                linewidths=[0.6 if c == "#94a3b8" else 1.0 for c in cols],
                                alpha=0.8)
            ax.add_collection(lc)
        cells = fr["cells"]
        cx = [c["x"] for c in cells]; cy = [c["y"] for c in cells]
        ax.scatter(cx, cy, s=130, c="#f97316", edgecolors="white", linewidths=1.6,
                   zorder=5)
        n_fib = sum(1 for e in edges if e[2] != "crosslink")
        n_xl = sum(1 for e in edges if e[2] == "crosslink")
        ax.set_title(f"Cells forming the ECM network  ·  step {fr['step']:>2}   "
                     f"fibers {n_fib}  ·  crosslinks {n_xl}",
                     fontsize=12, color="#1f2933")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=180, blit=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=6))
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
