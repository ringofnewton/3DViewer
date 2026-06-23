#!/usr/bin/env python3
"""Statistical power: headline metrics with CIs + effect sizes (VALIDATION.md §2.4).

Re-reports the two central mechanics results over many seeds as mean ± 95% CI with
a paired effect size (Cohen's d), so the claims carry uncertainty, not single runs:

  A. Single-cell radial aster — tension-weighted radiality vs uniform radiality.
  B. Inter-cell bridge        — axis alignment vs perpendicular control (sep=60 µm).

A claim is significant at 95% if the paired-difference CI excludes 0. Writes
output_validation/figures/stats_power.png. (No SciPy dependency: Student-t critical
values are tabulated for the seed counts used.)

    python stats_power.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech

DOMAIN, ZC, SLAB = 200.0, 100.0, 26.0
N = 16
SEEDS = list(range(1, N + 1))
T_CRIT = {8: 2.365, 10: 2.262, 12: 2.201, 14: 2.160, 16: 2.131, 20: 2.093}  # t_.975, df=N-1


def _net(seed):
    return mech.build_random_network(DOMAIN, n_fibers=200, fiber_len=70.0, seg_len=7.0,
                                     crosslink_dist=4.5, seed=seed, slab_thickness=SLAB)


def aster(seed):
    net0 = _net(seed)
    P = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0, max_iter=2200)
    single = np.array([[DOMAIN / 2, DOMAIN / 2, ZC]])
    net, _ = mech.relax(net0, single, P)
    _, tension, _ = mech.compute_forces(net, single, P)
    dirs, _ = mech.edge_dirs_lengths(net)
    mid = 0.5 * (net.nodes[net.edges[:, 0]] + net.nodes[net.edges[:, 1]])
    radial = mid - single[0]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    radiality = np.abs(np.einsum("ij,ij->i", dirs, radial))
    near = np.linalg.norm(mid - single[0], axis=1) < P.reach
    ten = np.clip(tension, 0, None)
    return (float(np.average(radiality[near], weights=ten[near] + 1e-9)),
            float(np.mean(radiality[near])))


def bridge(seed, sep=60.0):
    net0 = _net(seed)
    c = DOMAIN / 2.0
    a = np.array([c - sep / 2, c, ZC]); b = np.array([c + sep / 2, c, ZC])
    P = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0, buckle_ratio=0.02,
                        max_iter=2200)
    net, _ = mech.relax(net0, np.array([a, b]), P)
    al, _ = mech.axis_alignment(net, b - a, (a, b), band=20.0)
    pe, _ = mech.axis_alignment(net, np.array([0.0, 1.0, 0.0]), (a, b), band=20.0)
    return float(al), float(pe)


def stat(x):
    x = np.asarray(x); m = x.mean(); sd = x.std(ddof=1)
    ci = T_CRIT[N] * sd / np.sqrt(len(x))
    return m, ci, sd


def report(name, treat, ctrl, t_label, c_label):
    mt, cit, _ = stat(treat); mc, cic, _ = stat(ctrl)
    diff = np.asarray(treat) - np.asarray(ctrl)
    md, cid, sdd = stat(diff)
    d = md / sdd if sdd > 0 else float("inf")
    sig = (md - cid) > 0
    print(f"\n{name}")
    print(f"  {t_label:<26} {mt:.3f} ± {cit:.3f}")
    print(f"  {c_label:<26} {mc:.3f} ± {cic:.3f}")
    print(f"  difference                 {md:.3f} ± {cid:.3f}  (Cohen's d = {d:.2f})")
    print(f"  → {'SIGNIFICANT (95% CI excludes 0)' if sig else 'not significant'}")
    return (mt, cit, mc, cic, md, cid, d)


def main():
    print(f"Statistical power of the headline mechanics results (n={N} seeds, 95% CI)")
    aw, au = zip(*[aster(s) for s in SEEDS])
    bx, bp = zip(*[bridge(s) for s in SEEDS])
    A = report("A. Single-cell radial aster", aw, au,
               "tension-weighted radiality", "uniform radiality")
    B = report("B. Inter-cell bridge (sep=60µm)", bx, bp,
               "axis alignment", "perpendicular control")
    print(f"\n  (isotropic baseline for alignment = {1/3:.3f})")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].bar([0, 1], [A[0], A[2]], yerr=[A[1], A[3]], capsize=5,
                  color=["#c0392b", "#95a5a6"])
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["tension-wtd", "uniform"])
        ax[0].set_title(f"A. Single-cell aster radiality\nd={A[6]:.2f}"); ax[0].set_ylim(0, 1)
        ax[1].bar([0, 1], [B[0], B[2]], yerr=[B[1], B[3]], capsize=5,
                  color=["#c0392b", "#2c6fbb"])
        ax[1].axhline(1/3, color="0.5", ls=":", label="isotropic")
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["bridge axis", "perp"])
        ax[1].set_title(f"B. Inter-cell bridge\nd={B[6]:.2f}"); ax[1].set_ylim(0, 0.7)
        ax[1].legend(fontsize=8)
        for a in ax: a.set_ylabel("nematic alignment"); a.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Headline metrics: mean ± 95% CI (n={N} seeds)")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "stats_power.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'stats_power.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
