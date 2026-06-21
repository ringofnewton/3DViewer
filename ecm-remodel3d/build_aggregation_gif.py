#!/usr/bin/env python3
"""Animated GIF of cell aggregation: durotaxis vs durotaxis+chemotaxis.

Side-by-side, same seed: pure durotaxis disperses the cells; adding cell-cell
chemotaxis makes them converge along the reinforced collagen they build. GIFs
render inline on mobile.

    python build_aggregation_gif.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import coupled_sim as cs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output_validation", "figures", "ecm_aggregation.gif")
SUB = 5                       # interpolation sub-frames between recorded steps


def trail(traj):
    """linearly interpolate (T,M,3) cell trajectory to (T*SUB, M, 3)."""
    T = len(traj); out = []
    for i in range(T - 1):
        for s in range(SUB):
            out.append(traj[i] + (traj[i + 1] - traj[i]) * s / SUB)
    out.append(traj[-1])
    return np.array(out)


def main():
    ctrl = cs.run(1, chemo=0.0)
    trt = cs.run(1, chemo=cs.CHEMO)
    tc = trail(ctrl["traj"]); tt = trail(trt["traj"])
    D = cs.DOMAIN
    nF = len(tc)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    fig.patch.set_facecolor("white")

    def panel(ax, net, tr, upto, title, col):
        ax.clear(); ax.set_xlim(0, D); ax.set_ylim(0, D); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#f7fafc")
        e = net.edges; m = net.k_scale > 1.5            # reinforced sub-network
        nodes = net.nodes
        for (a, b) in e[m][:1400]:
            ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color="#93b4e6", lw=0.5, alpha=0.55)
        for ci in range(tr.shape[1]):
            ax.plot(tr[:upto + 1, ci, 0], tr[:upto + 1, ci, 1], color=col, lw=1.4, alpha=0.7)
            ax.scatter(tr[upto, ci, 0], tr[upto, ci, 1], s=90, c="#f97316",
                       edgecolors="white", linewidths=1.4, zorder=5)
        d = []
        cells = tr[upto]
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                d.append(np.linalg.norm(cells[i] - cells[j]))
        ax.set_title(f"{title}\nmean cell–cell distance: {np.mean(d):.0f} µm",
                     fontsize=12, color="#1f2933")

    def draw(k):
        panel(axes[0], ctrl["net"], tc, k, "durotaxis only  →  disperse", "#94a3b8")
        panel(axes[1], trt["net"], tt, k, "+ chemotaxis  →  aggregate", "#db2777")
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=140, blit=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=7))
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB, {nF} frames)")


if __name__ == "__main__":
    main()
