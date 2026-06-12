#!/usr/bin/env python3
"""2D morphogenesis: place cells, watch ECM mature along time → morphology differs.

A flat (2D) collagen sheet with contractile cells. Over time the matrix is
remodeled plastically (loaded fibers reinforce/compact, slack fibers are pruned),
so the ECM *matures* into load-bearing collagen. Different cell arrangements mature
into different collagen morphologies — the point of the plot.

Output (output_morph2d/figures/):
  morphology_grid.png   rows = cell arrangements, cols = time → ECM maturation
  morphology_metrics.png  alignment / reinforced-fiber curves over time

  python run_morphology2d.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_morph2d")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 200.0, 100.0
c = DOMAIN / 2


def arrangements():
    """Each: name -> (M,3) contractile-cell positions in the z=Z plane."""
    ring = [[c + 46 * np.cos(t), c + 46 * np.sin(t), Z]
            for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    return {
        "pair": np.array([[c - 30, c, Z], [c + 30, c, Z]]),
        "line (4)": np.array([[c - 54, c, Z], [c - 18, c, Z], [c + 18, c, Z], [c + 54, c, Z]]),
        "triangle": np.array([[c, c + 42, Z], [c - 36, c - 22, Z], [c + 36, c - 22, Z]]),
        "ring (6)": np.array(ring),
    }


def draw(ax, net, cells):
    """2D top-view; fibers colored by plastic reinforcement (maturity)."""
    k = net.k_scale
    kmax = max(k.max(), 1.5)
    order = np.argsort(k)
    for idx in order:
        e = net.edges[idx]
        pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        v = (k[idx] - 1.0) / (kmax - 1.0)
        if v > 0.04:
            col = plt.cm.inferno(0.28 + 0.64 * min(v, 1)); lw = 0.5 + 2.6 * min(v, 1); al = 0.95
        else:
            col = "#9fb3cc"; lw = 0.5; al = 0.45
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col, lw=lw, alpha=al, solid_capstyle="round")
    for p in cells:
        ax.add_patch(plt.Circle((p[0], p[1]), 9, color="#35d6d0", ec="white", lw=1.2, zorder=5, alpha=0.9))
    ax.set_xlim(24, DOMAIN - 24); ax.set_ylim(24, DOMAIN - 24)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def global_order(net):
    """Nematic order of reinforced fibers (0=isotropic .. 1=one direction)."""
    k = net.k_scale
    sel = k > 1.4
    if sel.sum() < 3:
        return 0.0
    d = net.nodes[net.edges[sel, 1]] - net.nodes[net.edges[sel, 0]]
    d = d[:, :2]
    L = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(L, 1e-9)
    # 2D nematic order parameter
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sqrt(np.mean(np.cos(2 * ang)) ** 2 + np.mean(np.sin(2 * ang)) ** 2))


def main():
    os.makedirs(FIG, exist_ok=True)
    P = mech.MechParams(reach=36, reach_decay=20, planar=True)
    # selective + gradual maturation: only the most-loaded fibers reinforce, slowly
    RP = rem.RemodelParams(n_steps=10, compaction=0.04, reinforce_rate=0.22,
                           reinforce_pct=72, slack_pct=35, slack_steps=2)
    arr = arrangements()
    col_steps = [0, 2, 4, 7, 10]          # time columns to show

    rows = len(arr)
    fig, axes = plt.subplots(rows, len(col_steps),
                             figsize=(3.0 * len(col_steps), 3.0 * rows))
    metrics = {}

    for r, (name, cells) in enumerate(arr.items()):
        print(f"  {name}: remodeling …")
        net0 = mech.build_random_network(DOMAIN, n_fibers=170, fiber_len=70,
                                         seg_len=7.0, crosslink_dist=4.5,
                                         seed=4, slab_thickness=0.0)
        # t=0 = relaxed but un-remodeled matrix (immature); then maturation frames
        net_t0, _ = mech.relax(net0, cells, P)
        frames, _ = rem.remodel(net0, cells, P, RP)
        seq = [net_t0] + [f["net"] for f in frames]
        order_t = [global_order(n) for n in seq]
        reinf_t = [float(np.mean(n.k_scale > 1.4)) for n in seq]
        metrics[name] = (order_t, reinf_t)
        for ccol, step in enumerate(col_steps):
            net = seq[min(step, len(seq) - 1)]
            ax = axes[r, ccol]
            draw(ax, net, cells)
            if r == 0:
                ax.set_title(f"t = {step}", fontsize=12, fontweight="bold")
            if ccol == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")

    fig.suptitle("2D ECM maturation over time — cell arrangement sets the collagen morphology "
                 "(color = collagen reinforcement)", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "morphology_grid.png"), dpi=130)
    plt.close(fig)

    # metric curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (order_t, reinf_t) in metrics.items():
        t = range(len(order_t))
        a1.plot(t, order_t, "o-", label=name)
        a2.plot(t, reinf_t, "o-", label=name)
    a1.set(title="ECM alignment (nematic order)", xlabel="time step", ylabel="0=isotropic · 1=aligned")
    a2.set(title="Matured (reinforced) fiber fraction", xlabel="time step", ylabel="fraction")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend(frameon=False, fontsize=10)
    fig.suptitle("Morphology metrics diverge by cell arrangement", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "morphology_metrics.png"), dpi=130)
    plt.close(fig)
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
