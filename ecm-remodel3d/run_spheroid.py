#!/usr/bin/env python3
"""A spheroid in a simple collagen gel — time-course remodeling + quantification.

Start simple: a uniform collagen sheet with one (or two) contractile spheroids.
Over time the spheroid condenses the surrounding matrix into radially aligned
fiber tracks and a dense rim, opening pores further out — the classic
spheroid-in-collagen result. Two spheroids build an aligned ECM bridge between
them. The emphasis here is the QUANTITATIVE time-course.

Outputs (output_spheroid/figures/):
  spheroid_timecourse.png   matrix condensing around the spheroid over time
  spheroid_metrics.png      radial alignment, near-rim density, pore, connectivity vs t
  two_spheroid_bridge.png   bridge formation + inter-spheroid alignment vs t

  python run_spheroid.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, seeding
from ecm_pipeline import morpho_metrics as mm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_spheroid")
FIG = os.path.join(OUT, "figures")
DOMAIN, Z = 220.0, 110.0
c = DOMAIN / 2

P = mech.MechParams(reach=26, reach_decay=16, planar=True, traction=1.5)
RP = rem.RemodelParams(n_steps=8, compaction=0.05, reinforce_rate=0.28,
                       reinforce_pct=62, slack_pct=32, slack_steps=3)
ISO = 0.637   # radial-alignment value of an isotropic matrix (2/pi)


def gel(seed=4, n_fibers=210):
    return mech.build_random_network(DOMAIN, n_fibers=n_fibers, fiber_len=60,
                                     seg_len=6.0, crosslink_dist=4.2,
                                     seed=seed, slab_thickness=0.0)


def draw(ax, net, cells=None):
    """Thin, microscopy-like dark field (fine bright fibers on black)."""
    ax.set_facecolor("black")
    k = net.k_scale
    kmax = max(k.max(), 1.5)
    order = np.argsort(k)
    for idx in order:
        e = net.edges[idx]
        pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        v = min(max((k[idx] - 1.0) / (kmax - 1.0), 0), 1)
        g = 0.30 + 0.70 * v                     # brightness by reinforcement
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=(g, g, min(1, g + 0.05)),
                lw=0.22 + 0.95 * v, alpha=0.55 + 0.4 * v, solid_capstyle="round")
    if cells is not None and len(cells):
        ax.scatter(cells[:, 0], cells[:, 1], s=5, c="#35d6d0", alpha=0.35, zorder=5)
    ax.set_xlim(20, DOMAIN - 20); ax.set_ylim(20, DOMAIN - 20)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def timecourse(cells):
    """t=0 is the raw isotropic gel; then plastic remodeling frames."""
    net0 = gel()
    frames, _ = rem.remodel(net0, cells, P, RP)
    return [net0] + [f["net"] for f in frames]


def reinforced_length(net, k_thresh=1.4):
    sel = net.k_scale > k_thresh
    if not sel.any():
        return 0.0
    d = net.nodes[net.edges[sel, 1]] - net.nodes[net.edges[sel, 0]]
    return float(np.sum(np.linalg.norm(d, axis=1)))


# --------------------------------------------------------------------------- #
def single_spheroid():
    print("[1/2] single spheroid time-course …")
    center = np.array([c, c, Z])
    cells = seeding.spheroid(center, n_cells=24, radius=15.0, seed=1)
    seq = timecourse(cells)
    cols = [0, 2, 4, 6, len(seq) - 1]

    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.0))
    fig.patch.set_facecolor("#0a0a0a")
    for ax, ci in zip(axes, cols):
        draw(ax, seq[ci], cells)
        lab = "t = 0 (isotropic gel)" if ci == 0 else f"t = {ci}"
        ax.set_title(lab, color="white", fontsize=12, fontweight="bold")
    fig.suptitle("Spheroid condenses the matrix into radial collagen tracks over time",
                 color="white", fontweight="bold", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "spheroid_timecourse.png"), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ---- quantitative time-course ---------------------------------------
    t = np.arange(len(seq))
    radial = [mm.radial_alignment(n, center, reinforced_only=False) for n in seq]
    reinf = [mm.reinforced_fraction(n) for n in seq]
    pore = [np.median(mm.pore_sizes(n, DOMAIN, n=1500)) for n in seq]
    conn = [mm.connectivity(n) for n in seq]
    rlen = [reinforced_length(n) for n in seq]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    def plot(a, y, title, ylab, col):
        a.plot(t, y, "o-", color=col); a.set(title=title, xlabel="time step", ylabel=ylab); a.grid(alpha=0.3)
    plot(ax[0, 0], radial, "Radial matrix alignment (aster)", "0.64 iso · 1 radial", "#8948ff")
    ax[0, 0].axhline(ISO, color="#888", ls=":", lw=1); ax[0, 0].text(0.2, ISO + 0.005, "isotropic", color="#888", fontsize=9)
    plot(ax[0, 1], reinf, "Reinforced (strut) fiber fraction", "fraction", "#00a8a8")
    plot(ax[0, 2], rlen, "Total reinforced collagen (strut length)", "µm", "#2457ff")
    plot(ax[1, 0], pore, "Median pore radius", "µm", "#d04b4b")
    plot(ax[1, 1], conn, "Strut-network connectivity", "largest comp. frac.", "#177245")

    # density profile vs distance, at three times (rim compaction)
    ax5 = ax[1, 2]
    for ci, lab, col in [(2, "early", "#9fb3cc"), (len(seq)//2, "mid", "#7aa2ff"), (len(seq)-1, "final", "#d04b4b")]:
        r, dens = mm.density_profile(seq[ci], center, DOMAIN)
        if len(r):
            ax5.plot(r, dens, "o-", color=col, label=lab)
    ax5.set(title="Reinforced-ECM density vs distance from spheroid", xlabel="distance (µm)", ylabel="fiber length / area")
    ax5.grid(alpha=0.3); ax5.legend(frameon=False)

    fig.suptitle("Spheroid remodeling — quantitative time-course", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "spheroid_metrics.png"), dpi=140)
    plt.close(fig)
    print(f"      radial alignment {radial[0]:.2f} → {radial[-1]:.2f} (iso={ISO}); "
          f"reinforced {reinf[0]:.2f} → {reinf[-1]:.2f}; pore {pore[0]:.1f} → {pore[-1]:.1f} µm")


def two_spheroids():
    print("[2/2] two-spheroid bridge …")
    a = np.array([c - 42, c, Z]); b = np.array([c + 42, c, Z])
    cells = seeding.spheroids([a, b], n_cells=20, radius=13.0, seed=1)
    seq = timecourse(cells)
    cols = [0, 2, 5, len(seq) - 1]

    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.2))
    fig.patch.set_facecolor("#0a0a0a")
    for ax, ci in zip(axes, cols):
        draw(ax, seq[ci], cells)
        ax.set_title(f"t = {ci}", color="white", fontsize=12, fontweight="bold")
    fig.suptitle("Two spheroids build an aligned collagen bridge between them",
                 color="white", fontweight="bold", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "two_spheroid_bridge.png"), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

    # reinforced collagen built in the band between the two spheroids
    axis = b - a
    def bridge_length(net):
        sel = net.k_scale > 1.4
        if sel.sum() < 1:
            return 0.0
        mid = 0.5 * (net.nodes[net.edges[sel, 0]] + net.nodes[net.edges[sel, 1]])[:, :2]
        ab = (b - a)[:2]
        t = np.clip(((mid - a[:2]) @ ab) / (ab @ ab), 0, 1)
        proj = a[:2] + t[:, None] * ab
        dperp = np.linalg.norm(mid - proj, axis=1)
        d = net.nodes[net.edges[sel, 1]] - net.nodes[net.edges[sel, 0]]
        L = np.linalg.norm(d, axis=1)
        return float(np.sum(L[dperp < 22]))
    bridge = [bridge_length(n) for n in seq]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(seq)), bridge, "o-", color="#8948ff")
    ax.set(title="Reinforced collagen built in the inter-spheroid bridge",
           xlabel="time step", ylabel="reinforced fiber length in band (µm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "two_spheroid_alignment.png"), dpi=140)
    plt.close(fig)
    print(f"      bridge reinforced length {bridge[0]:.0f} → {bridge[-1]:.0f} µm")


def main():
    os.makedirs(FIG, exist_ok=True)
    single_spheroid()
    two_spheroids()
    print("done. figures in", FIG)


if __name__ == "__main__":
    main()
