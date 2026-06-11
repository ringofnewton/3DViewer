"""Shared 2D network renderer — light/dark theme + continuous colour scale.

Line WIDTH is constant; the plastic reinforcement (k_scale, a continuous value =
how much a fiber has matured/bundled into a load-bearing strut) is shown by a
continuous colour map (default blue→red), not by discrete thickness.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

THEMES = {
    "light": dict(bg="white", cells="#0bb/", cell_color="#1186a8", base_alpha=0.85),
    "dark": dict(bg="#0a0a0a", cell_color="#35d6d0", base_alpha=0.9),
}


def reinforcement_norm(net, vmax=None):
    """Map k_scale (>=1) to 0..1 for the colour map (0 = soft, 1 = reinforced)."""
    k = np.asarray(net.k_scale)
    hi = vmax if vmax is not None else max(np.percentile(k, 98), 1.6)
    return np.clip((k - 1.0) / (hi - 1.0), 0, 1), hi


def draw_network(ax, net, cells=None, theme="light", cmap="coolwarm",
                 lw=1.0, vmax=None, show_cells=True, cell_size=8):
    bg = "white" if theme == "light" else "#0a0a0a"
    cell_color = "#1186a8" if theme == "light" else "#35d6d0"
    ax.set_facecolor(bg)
    cmap_fn = plt.get_cmap(cmap)
    norm, hi = reinforcement_norm(net, vmax)
    order = np.argsort(net.k_scale)              # reinforced fibers drawn on top
    for idx in order:
        e = net.edges[idx]
        pa, pb = net.nodes[e[0]], net.nodes[e[1]]
        v = float(norm[idx])
        a = 0.45 + 0.5 * v if theme == "light" else 0.5 + 0.45 * v
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cmap_fn(v), lw=lw,
                alpha=a, solid_capstyle="round")
    if show_cells and cells is not None and len(cells):
        ax.scatter(cells[:, 0], cells[:, 1], s=cell_size, c=cell_color,
                   alpha=0.4, edgecolors="none", zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return hi


def add_colorbar(fig, ax=None, cmap="coolwarm", label="reinforcement (soft→strut)",
                 rect=(0.915, 0.30, 0.012, 0.40)):
    """Add a colour scale on a DEDICATED right-side axis (never steals panel space).

    Callers should leave room on the right, e.g.
    `fig.tight_layout(rect=[0, 0, 0.90, top])`. The `ax` argument is accepted for
    backward compatibility but ignored for placement.
    """
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cax = fig.add_axes(rect)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=9)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["soft", "strut"])
    return cb
