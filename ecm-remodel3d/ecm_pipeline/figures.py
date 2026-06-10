"""Publication figures (matplotlib, Agg backend, 300 dpi)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

_ACCENT = "#2457ff"
_TEAL = "#00a8a8"
_PURPLE = "#8948ff"


def fig_time_series(ts_rows, path):
    f = np.array([r["frame"] for r in ts_rows])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].plot(f, [r["volume_fraction"] for r in ts_rows], "o-", color=_ACCENT)
    ax[0, 0].set(title="ECM volume fraction", xlabel="frame", ylabel="fraction")

    ax[0, 1].plot(f, [r["alignment_index"] for r in ts_rows], "o-", color=_PURPLE, label="global")
    ax[0, 1].plot(f, [r["local_alignment"] for r in ts_rows], "s--", color=_TEAL, label="local (per-cell)")
    ax[0, 1].set(title="Fiber alignment index", xlabel="frame", ylabel="0=iso  1=aligned")
    ax[0, 1].legend(frameon=False)

    ax[1, 0].plot(f, [r["mean_contacts"] for r in ts_rows], "o-", color=_TEAL)
    ax[1, 0].set(title="Cell–ECM contact", xlabel="frame", ylabel="mean fibers in contact / cell")

    ax[1, 1].plot(f, [r["largest_component"] for r in ts_rows], "o-", color=_ACCENT)
    ax[1, 1].set(title="Network connectivity", xlabel="frame",
                 ylabel="largest component fraction")

    fig.suptitle("ECM remodeling — structural metrics over time", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_orientation_rose(fibers, path):
    """Azimuthal orientation distribution of fibers (polar histogram)."""
    d = fibers["direction"]
    az = np.arctan2(d[:, 1], d[:, 0])
    az = np.concatenate([az, az + np.pi])  # axial symmetry
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    counts, edges = np.histogram(az % (2 * np.pi), bins=36, range=(0, 2 * np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=2 * np.pi / 36, color=_PURPLE, alpha=0.8,
           edgecolor="white", linewidth=0.5)
    ax.set_title("Fiber orientation rose (final frame)", pad=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pore_size(fibers, domain, path):
    pores = metrics.pore_size_distribution(fibers, domain)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pores, bins=40, color=_TEAL, alpha=0.85, edgecolor="white")
    ax.axvline(np.median(pores), color=_ACCENT, ls="--",
               label=f"median = {np.median(pores):.1f} µm")
    ax.set(title="Pore-size distribution (final frame)",
           xlabel="nearest-fiber distance (µm)", ylabel="count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_remodeling(events_csv, path):
    rr = metrics.remodeling_rate(events_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(rr["frame"] - 0.2, rr["births"], width=0.4, color=_TEAL, label="births")
    ax.bar(rr["frame"] + 0.2, rr["deaths"], width=0.4, color="#d04b4b", label="deaths")
    ax.set(title="ECM remodeling events", xlabel="frame", ylabel="fiber events / snapshot")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_persistence(tracks_csv, path):
    pers = metrics.migration_persistence(tracks_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pers["per_cell"], bins=15, range=(0, 1), color=_ACCENT, alpha=0.85,
            edgecolor="white")
    ax.axvline(pers["mean_persistence"], color=_PURPLE, ls="--",
               label=f"mean = {pers['mean_persistence']:.2f}")
    ax.set(title="Migration persistence (net / path length)",
           xlabel="persistence ratio", ylabel="cells")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_scene_3d(cells, fibers, path):
    """Quick 3D preview (cells as points, fibers as line segments)."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"nascent": "#75e4a8", "mature": "#a8b6c9",
              "aligned": _PURPLE, "degraded": "#d04b4b"}
    for p1, p2, st in zip(fibers["p1"], fibers["p2"], fibers["state"]):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=colors.get(st, "#a8b6c9"), lw=0.8, alpha=0.7)
    ax.scatter(cells["pos"][:, 0], cells["pos"][:, 1], cells["pos"][:, 2],
               s=60, c=_ACCENT, edgecolors="white", depthshade=True)
    ax.set_title("Cells + ECM fibers (final frame)", fontweight="bold")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
