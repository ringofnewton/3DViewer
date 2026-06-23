#!/usr/bin/env python3
"""Morphogenesis experiment: cell placement -> collagen patterning, via mechanics.

This is the predictive core. We do NOT script "draw a fiber between cells". We
place contractile cells in a passive collagen network and let the network relax
to mechanical equilibrium under cell traction. The aligned, tensed collagen
"strap" between neighboring cells then EMERGES from the physics — reproducing
the classic Stopak & Harris traction-structuring observation.

Geometry is a quasi-2D collagen slab (the canonical explant/gel setup). Studies:

  A. Two cells: a tensed, axis-aligned collagen track emerges between them, and
     the nonlinear (buckling) constitutive law is REQUIRED — a linear control
     network diffuses the force and builds no track. Averaged over seeds.
  B. Cell-spacing sweep: inter-cell distance tunes track alignment and the
     fraction of tension carried between the cells (force-transmission length).
  C. Geometry: 3 cells in a line vs a triangle -> different collagen scaffolds,
     i.e. different morphogenetic templates from the same cells.

Readout: nematic alignment S = <(dir·axis)^2> in a band (1/3 isotropic, 1 fully
aligned), plus its anisotropy vs the perpendicular direction.
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_mech")
FIG = os.path.join(OUT, "figures")
DOMAIN = 200.0
SLAB = 26.0
ZC = DOMAIN / 2.0
_ACCENT, _PURPLE, _TEAL, _RED = "#2457ff", "#8948ff", "#00a8a8", "#d04b4b"

# Traction strong enough to finitely strain the matrix (as real cells do).
P = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0, buckle_ratio=0.02)
P_LIN = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0,
                        buckle_ratio=1.0, stiffen_alpha=0.0)


def net_slab(seed, n_fibers=220):
    return mech.build_random_network(DOMAIN, n_fibers=n_fibers, fiber_len=70,
                                     seg_len=7.0, crosslink_dist=4.5,
                                     seed=seed, slab_thickness=SLAB)


def _draw(ax, net, cells, p, title):
    _, tension, _ = mech.compute_forces(net, cells, p)
    t = tension
    tmax = max(np.percentile(np.abs(t), 97), 1e-6)
    # draw a top view (slab), tension warm/thick, compression cool/faint
    order = np.argsort(np.abs(t))            # strong fibers on top
    for idx in order:
        e, te = net.edges[idx], t[idx]
        pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        if te >= 0:
            c = plt.cm.inferno(0.32 + 0.62 * min(te / tmax, 1))
            lw = 0.4 + 2.8 * min(te / tmax, 1)
            al = 0.95
        else:
            c = "#33507d"; lw = 0.4; al = 0.3
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=c, lw=lw, alpha=al,
                solid_capstyle="round")
    ax.scatter(cells[:, 0], cells[:, 1], s=160, c=_TEAL, edgecolors="white",
               linewidths=1.4, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(20, DOMAIN - 20); ax.set_ylim(20, DOMAIN - 20)


def band_alignment(net, a, b, band=22.0):
    """Nematic order along the a-b axis, and anisotropy vs perpendicular."""
    axis = b - a
    al, n = mech.axis_alignment(net, axis, (a, b), band)
    perp = np.array([-axis[1], axis[0], 0.0])
    pal, _ = mech.axis_alignment(net, perp, (a, b), band)
    return al, pal, n


def study_A(rows):
    print("  [A] two-cell track + nonlinearity control (seed-averaged) …")
    seeds = [3, 11, 19, 27]
    s = 56.0
    c = DOMAIN / 2
    cells = np.array([[c - s / 2, c, ZC], [c + s / 2, c, ZC]])
    res = {"initial": [], "nonlinear": [], "linear": []}
    keep = None
    for sd in seeds:
        net0 = net_slab(sd)
        net_nl, _ = mech.relax(net0, cells, P)
        net_lin, _ = mech.relax(net0, cells, P_LIN)
        res["initial"].append(band_alignment(net0, cells[0], cells[1])[0])
        res["nonlinear"].append(band_alignment(net_nl, cells[0], cells[1])[0])
        res["linear"].append(band_alignment(net_lin, cells[0], cells[1])[0])
        if keep is None:
            keep = (net0, net_nl, net_lin)
    for k in res:
        print(f"      {k:10s} band alignment = {np.mean(res[k]):.2f} ± {np.std(res[k]):.2f}"
              f"   (isotropic ≈ 0.33)")

    net0, net_nl, net_lin = keep
    fig = plt.figure(figsize=(15, 5.2))
    for k, (net, ttl, p) in enumerate([
            (net0, f"initial matrix\nband alignment={np.mean(res['initial']):.2f}", P),
            (net_nl, f"collagen — nonlinear (buckling)\nband alignment="
                     f"{np.mean(res['nonlinear']):.2f}  ← track forms", P),
            (net_lin, f"linear control\nband alignment={np.mean(res['linear']):.2f}  "
                      f"← no track", P_LIN)]):
        ax = fig.add_subplot(1, 3, k + 1)
        _draw(ax, net, cells, p, ttl)
    fig.suptitle("A · A tensed, aligned collagen track emerges between two cells — "
                 "and requires the buckling nonlinearity (warm=tension, blue=buckled)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A_two_cell_track.png"))
    plt.close(fig)
    rows.append(dict(study="A", case="initial", spacing=s,
                     alignment=np.mean(res["initial"]), tension_band=np.nan))
    rows.append(dict(study="A", case="nonlinear", spacing=s,
                     alignment=np.mean(res["nonlinear"]), tension_band=np.nan))
    rows.append(dict(study="A", case="linear", spacing=s,
                     alignment=np.mean(res["linear"]), tension_band=np.nan))


def study_B(rows):
    print("  [B] cell-spacing sweep …")
    spacings = [28, 44, 64, 88, 116]
    aligns, tfracs = [], []
    panels = []
    for sp in spacings:
        al_s, tf_s = [], []
        keep = None
        for sd in (3, 11, 19):
            net0 = net_slab(sd)
            c = DOMAIN / 2
            cells = np.array([[c - sp / 2, c, ZC], [c + sp / 2, c, ZC]])
            net, _ = mech.relax(net0, cells, P)
            al, _, _ = band_alignment(net, cells[0], cells[1])
            tf = mech.tension_along_axis(net, cells, P, 22.0)
            al_s.append(al); tf_s.append(tf)
            if keep is None:
                keep = (net, cells)
        aligns.append(np.mean(al_s)); tfracs.append(np.mean(tf_s))
        panels.append((keep[0], keep[1], sp, np.mean(al_s)))
        rows.append(dict(study="B", case=f"spacing_{sp}", spacing=sp,
                         alignment=np.mean(al_s), tension_band=np.mean(tf_s)))
        print(f"      spacing={sp:3d} µm  align={np.mean(al_s):.2f}  tension fraction={np.mean(tf_s):.2f}")

    fig = plt.figure(figsize=(16, 6))
    for k, (net, cells, sp, al) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 1)
        _draw(ax, net, cells, P, f"spacing={sp} µm · align={al:.2f}")
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(spacings, aligns, "o-", color=_PURPLE, label="band alignment")
    ax.plot(spacings, tfracs, "s--", color=_TEAL, label="tension fraction in band")
    ax.axhline(0.333, color="#888", ls=":", lw=1, label="isotropic")
    ax.set_xlabel("cell spacing (µm)"); ax.set_ylabel("value"); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9); ax.set_title("spacing tunes the track", fontweight="bold")
    fig.suptitle("B · Inter-cell distance tunes the emergent collagen track",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "B_spacing_sweep.png"))
    plt.close(fig)


def study_C(rows):
    print("  [C] cell geometry: line vs triangle …")
    c = DOMAIN / 2
    configs = {
        "line (3 cells)": np.array([[c - 44, c, ZC], [c, c, ZC], [c + 44, c, ZC]]),
        "triangle (3 cells)": np.array([[c, c + 44, ZC],
                                        [c - 38, c - 22, ZC],
                                        [c + 38, c - 22, ZC]]),
    }
    fig = plt.figure(figsize=(12, 5.6))
    for k, (name, cells) in enumerate(configs.items()):
        als = []
        keep = None
        for sd in (7, 13, 21):
            net0 = net_slab(sd, n_fibers=240)
            net, _ = mech.relax(net0, cells, P)
            pair_al = []
            for a in range(len(cells)):
                for b in range(a + 1, len(cells)):
                    al, _, _ = band_alignment(net, cells[a], cells[b], band=18.0)
                    pair_al.append(al)
            als.append(np.mean(pair_al))
            if keep is None:
                keep = net
        mean_al = float(np.mean(als))
        rows.append(dict(study="C", case=name, spacing=np.nan,
                         alignment=mean_al, tension_band=np.nan))
        print(f"      {name:22s} mean pairwise track alignment={mean_al:.2f}")
        ax = fig.add_subplot(1, 2, k + 1)
        _draw(ax, keep, cells, P, f"{name}\nmean track align={mean_al:.2f}")
    fig.suptitle("C · Same cells, different placement → different collagen scaffold",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "C_geometry.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = []
    print("[mechanics] relaxing collagen slab under cell traction …")
    study_A(rows)
    study_B(rows)
    study_C(rows)
    with open(os.path.join(OUT, "mechanics_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["study", "case", "spacing",
                                           "alignment", "tension_band"])
        w.writeheader()
        w.writerows(rows)
    print("[mechanics] done. figures in", FIG)


if __name__ == "__main__":
    main()
