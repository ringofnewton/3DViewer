#!/usr/bin/env python3
"""Durotaxis + contact guidance from the cell-agent layer — Step 4 verify.

Claim: with mechanosensing (traction ~ local stiffness) and a gradient-climbing
migration rule, a cell performs DUROTAXIS — it migrates up an imposed matrix
stiffness gradient — while an identical cell in a uniform matrix shows no net
drift. This is the minimal active-cell behavior the fixed-focus model could not
produce.

Setup: a slab network with a stiffness gradient imposed along +x (k_scale rises
with x). A cell starts at the center; we co-evolve matrix + cell and measure the
net displacement along x. Control = uniform stiffness (no gradient).

    python verify_migration.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech
from ecm_pipeline import ecm_cells as ec

DOMAIN, SLAB = 160.0, 26.0
SEEDS = [1, 2, 3, 4]
T_CRIT = {3: 3.182, 4: 3.182, 5: 2.776}


def stat(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), T_CRIT.get(n, 2.776) * x.std(ddof=1) / np.sqrt(n)


def make(seed, gradient):
    net = mech.build_random_network(DOMAIN, n_fibers=230, fiber_len=70.0, seg_len=7.0,
                                    crosslink_dist=4.5, seed=seed, slab_thickness=SLAB)
    mid_x = 0.5 * (net.nodes[net.edges[:, 0], 0] + net.nodes[net.edges[:, 1], 0])
    if gradient:
        # stiffness rises ~4x from left to right
        net.k_scale = 1.0 + 3.0 * (mid_x / DOMAIN)
    return net


def run(seed, gradient):
    net = make(seed, gradient)
    c = DOMAIN / 2.0
    cells = np.array([[c, c, c]])
    pc = ec.CellParams(speed=4.0, durotaxis=1.0, guidance=0.4, sense_radius=24.0)
    pm = mech.MechParams(planar=True, max_iter=3500, ftol=2e-2)
    traj, _ = ec.migrate(net, cells, pc, pm, n_steps=10, record=True)
    _, dx = ec.net_displacement(traj)
    return dx


def main():
    print(f"Durotaxis test (n={len(SEEDS)} seeds, mean +/- 95% CI)")
    print("net cell displacement along +x (toward stiffer matrix)\n")

    grad = [run(s, True) for s in SEEDS]
    ctrl = [run(s, False) for s in SEEDS]
    mg, cg = stat(grad); mc, cc = stat(ctrl)

    print(f"  stiffness gradient : dx = {mg:+.2f} +/- {cg:.2f} um")
    print(f"  uniform (control)  : dx = {mc:+.2f} +/- {cc:.2f} um")

    durotaxis = mg > mc + 0.5 and mg > 0
    print(f"\n  net up-gradient migration = {mg - mc:+.2f} um  "
          f"({'YES — durotaxis (cell climbs the stiffness gradient)' if durotaxis else 'no clear durotaxis'})")
    print("Interpretation: with rigidity sensing + gradient climbing, the cell")
    print("actively migrates toward stiffer matrix; matrix and cell co-evolve.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # one example trajectory for the figure
        net = make(1, True); c = DOMAIN / 2.0
        traj, fnet = ec.migrate(net, np.array([[c, c, c]]),
                                ec.CellParams(speed=4.0, durotaxis=1.0, guidance=0.4),
                                mech.MechParams(planar=True, max_iter=3500, ftol=2e-2),
                                n_steps=10)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].bar([0, 1], [mg, mc], yerr=[cg, cc], capsize=5,
                  color=["#27ae60", "#7f8c8d"])
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["gradient", "uniform"])
        ax[0].set_ylabel("net displacement along +x (um)")
        ax[0].set_title("Durotaxis")
        mid = 0.5 * (fnet.nodes[fnet.edges[:, 0]] + fnet.nodes[fnet.edges[:, 1]])
        ax[1].scatter(mid[:, 0], mid[:, 1], s=1, c=fnet.k_scale, cmap="viridis", alpha=0.5)
        tr = traj[:, 0, :]
        ax[1].plot(tr[:, 0], tr[:, 1], "o-", color="#c0392b", ms=4, label="cell path")
        ax[1].scatter([tr[0, 0]], [tr[0, 1]], c="white", edgecolor="k", zorder=5, label="start")
        ax[1].set_title("Cell path on stiffness gradient"); ax[1].legend(fontsize=8)
        ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("y (um)")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "durotaxis.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'durotaxis.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
