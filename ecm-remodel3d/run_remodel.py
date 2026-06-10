#!/usr/bin/env python3
"""Plastic ECM remodeling: a track that hardens over time and OUTLASTS the cells.

Alternates mechanical equilibrium with tension-driven remodeling (reinforce +
compact loaded fibers, degrade persistently slack ones). Over time the matrix is
pruned to its load-bearing collagen track and that track is reinforced into a
permanent structure — so when the cells are removed at the end, the track stays
(an elastic-only matrix springs back). Figures land in output_remodel/.

  python run_remodel.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_remodel")
FIG = os.path.join(OUT, "figures")
DOMAIN, ZC = 200.0, 100.0
_TEAL, _PURPLE = "#00a8a8", "#8948ff"


def draw(ax, net, cells, p, title, mode="tension"):
    if mode == "tension":
        _, t, _ = mech.compute_forces(net, cells, p)
        tmax = max(np.percentile(np.abs(t), 97), 1e-6)
        order = np.argsort(np.abs(t))
        for idx in order:
            e, te = net.edges[idx], t[idx]
            pa, pb = net.nodes[e[0]], net.nodes[e[1]]
            if te >= 0:
                c = plt.cm.inferno(0.32 + 0.62 * min(te / tmax, 1)); lw = 0.4 + 2.8 * min(te / tmax, 1); al = 0.95
            else:
                c = "#33507d"; lw = 0.4; al = 0.3
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=c, lw=lw, alpha=al, solid_capstyle="round")
    else:  # color by plastic reinforcement (after cells are gone)
        kmax = max(net.k_scale.max(), 1.0)
        order = np.argsort(net.k_scale)
        for idx in order:
            e, k = net.edges[idx], net.k_scale[idx]
            pa, pb = net.nodes[e[0]], net.nodes[e[1]]
            if k > 1.4:
                c = plt.cm.inferno(0.3 + 0.65 * min(k / kmax, 1)); lw = 0.5 + 2.6 * min(k / kmax, 1); al = 0.95
            else:
                c = "#33507d"; lw = 0.4; al = 0.3
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=c, lw=lw, alpha=al, solid_capstyle="round")
    if len(cells):
        ax.scatter(cells[:, 0], cells[:, 1], s=150, c=_TEAL, edgecolors="white", linewidths=1.4, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(20, DOMAIN - 20); ax.set_ylim(20, DOMAIN - 20)


def main():
    os.makedirs(FIG, exist_ok=True)
    c = 100
    cells = np.array([[c - 28, c, ZC], [c + 28, c, ZC]])
    P = mech.MechParams(reach=40, reach_decay=22)
    RP = rem.RemodelParams(n_steps=10, compaction=0.05, reinforce_rate=0.4,
                           slack_pct=30, slack_steps=2)

    print("[1/4] plastic remodeling over time …")
    net0 = mech.build_random_network(DOMAIN, n_fibers=220, seed=3, slab_thickness=26)
    frames, final = rem.remodel(net0, cells, P, RP)
    for f in frames[::2]:
        print(f"      step {f['step']:2d}  edges={f['n_edges']:4d}  "
              f"reinforced={f['frac_reinforced']:.2f}  band align={f['alignment']:.2f}")

    print("[2/4] montage + build-up curves …")
    idxs = np.linspace(0, len(frames) - 1, 6, dtype=int)
    fig = plt.figure(figsize=(16, 6))
    for k, fi in enumerate(idxs):
        ax = fig.add_subplot(2, 3, k + 1)
        fr = frames[fi]
        draw(ax, fr["net"], cells, P, f"step {fr['step']} · {fr['n_edges']} fibers · "
                                       f"align {fr['alignment']:.2f}")
    fig.suptitle("Plastic remodeling: matrix is pruned to its load-bearing collagen track "
                 "(warm = tension)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "remodel_montage.png")); plt.close(fig)

    steps = [f["step"] for f in frames]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, [f["alignment"] for f in frames], "o-", color=_PURPLE, label="band alignment")
    ax1.plot(steps, [f["frac_reinforced"] for f in frames], "s--", color=_TEAL, label="fraction reinforced")
    ax1.axhline(0.333, color="#888", ls=":", lw=1, label="isotropic")
    ax1.set_xlabel("remodeling step"); ax1.set_ylabel("alignment / fraction"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps, [f["n_edges"] for f in frames], "^-", color="#d04b4b", label="fiber count")
    ax2.set_ylabel("fiber count", color="#d04b4b")
    ax1.legend(loc="center right", frameon=False)
    ax1.set_title("Track hardens while slack fibers are pruned", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "remodel_curves.png")); plt.close(fig)

    print("[3/4] permanence test (remove the cells) …")
    elastic, _ = mech.relax(net0, cells, P)
    res_plastic, plastic_relaxed = rem.residual_self_stress(final, P)
    res_elastic, elastic_relaxed = rem.residual_self_stress(elastic, P)
    al_init = mech.axis_alignment(net0, cells[1] - cells[0], (cells[0], cells[1]), 22.0)[0]
    print(f"      residual self-stress after cell removal: "
          f"plastic={res_plastic:.0f}  elastic={res_elastic:.0f}")

    fig = plt.figure(figsize=(15, 5.2))
    ax = fig.add_subplot(1, 3, 1); draw(ax, net0, cells, P, "initial matrix")
    ax = fig.add_subplot(1, 3, 2)
    draw(ax, elastic_relaxed, np.zeros((0, 3)), P,
         f"ELASTIC, cells removed\nself-stress={res_elastic:.0f} — springs back", mode="reinforce")
    ax = fig.add_subplot(1, 3, 3)
    draw(ax, plastic_relaxed, np.zeros((0, 3)), P,
         f"PLASTIC, cells removed\nself-stress={res_plastic:.0f} — track remains", mode="reinforce")
    fig.suptitle("The remodeled collagen track outlasts the cells (plastic memory: "
                 "residual self-stress)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "remodel_permanence.png")); plt.close(fig)

    print("[4/4] done. figures in", FIG)


if __name__ == "__main__":
    main()
