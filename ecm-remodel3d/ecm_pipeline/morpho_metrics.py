"""Quantitative ECM-architecture metrics for the 2D culture/spheroid studies.

All operate on a FiberNetwork (from ecm_mechanics / ecm_remodel) projected to the
xy-plane. They are the readouts that turn "the pictures look different" into
numbers you can plot over time or compare across conditions:

  pore_sizes              distribution of cell-free gap radii (coverage/openness)
  nematic_order           global fiber alignment (0 isotropic .. 1 aligned)
  radial_alignment        fiber alignment RELATIVE to a center (spheroid asters)
  reinforced_fraction     fraction of fibers matured into load-bearing struts
  strut_thickness         bundle thickness via local fiber-density on struts
  connectivity            largest-connected-component fraction of the strut graph
  density_profile         fiber density vs distance from a center (compaction)
"""

from __future__ import annotations

import numpy as np

from . import metrics as _m


def _edges_xy(net):
    p1 = net.nodes[net.edges[:, 0]][:, :2]
    p2 = net.nodes[net.edges[:, 1]][:, :2]
    mid = 0.5 * (p1 + p2)
    d = p2 - p1
    L = np.linalg.norm(d, axis=1)
    u = d / np.maximum(L[:, None], 1e-9)
    return p1, p2, mid, u, L


def pore_sizes(net, domain, n=4000, seed=1):
    p1, p2, *_ = _edges_xy(net)
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, domain, size=(n, 2))
    return _m.point_segment_distance(pts, p1, p2).min(axis=1)


def nematic_order(net, reinforced_only=True, k_thresh=1.4):
    _, _, _, u, L = _edges_xy(net)
    sel = net.k_scale > k_thresh if reinforced_only else np.ones(len(L), bool)
    if sel.sum() < 3:
        return 0.0, np.nan
    ang = np.arctan2(u[sel, 1], u[sel, 0])
    w = L[sel]
    c = np.average(np.cos(2 * ang), weights=w)
    s = np.average(np.sin(2 * ang), weights=w)
    order = float(np.hypot(c, s))
    director = 0.5 * np.arctan2(s, c)
    return order, director


def radial_alignment(net, center, reinforced_only=True, k_thresh=1.4):
    """Mean |fiber·radial|: 1 = fibers point at the center (aster), 0.5 isotropic."""
    _, _, mid, u, L = _edges_xy(net)
    sel = net.k_scale > k_thresh if reinforced_only else np.ones(len(L), bool)
    if sel.sum() < 3:
        return np.nan
    r = mid[sel] - np.asarray(center)[:2]
    r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
    return float(np.average(np.abs(np.einsum("ij,ij->i", u[sel], r)), weights=L[sel]))


def reinforced_fraction(net, k_thresh=1.4):
    return float(np.mean(net.k_scale > k_thresh)) if len(net.k_scale) else 0.0


def strut_thickness(net, domain, k_thresh=1.4, cell=6.0):
    """Median number of reinforced-fiber midpoints per occupied grid cell.

    A proxy for how thick the condensed bundles (struts) are: condensation packs
    more fibers into the same lateral spot, raising local density on struts.
    """
    _, _, mid, _, _ = _edges_xy(net)
    sel = net.k_scale > k_thresh
    if sel.sum() < 3:
        return 0.0
    keys = np.floor(mid[sel] / cell).astype(int)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return float(np.median(counts))


def connectivity(net, k_thresh=1.4):
    """Largest-connected-component fraction of the reinforced (strut) sub-network."""
    sel = np.where(net.k_scale > k_thresh)[0]
    if len(sel) < 2:
        return 0.0
    used_nodes = np.unique(net.edges[sel].ravel())
    remap = {n: i for i, n in enumerate(used_nodes)}
    parent = list(range(len(used_nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for e in net.edges[sel]:
        ra, rb = find(remap[e[0]]), find(remap[e[1]])
        if ra != rb:
            parent[ra] = rb
    sizes = {}
    for i in range(len(used_nodes)):
        r = find(i); sizes[r] = sizes.get(r, 0) + 1
    return float(max(sizes.values()) / len(used_nodes))


def branch_points(net, k_thresh=1.4):
    """Number of strut junctions (reinforced-fiber nodes of degree >= 3).

    A reticular / cotton-candy network is highly branched, so this counts the
    divergent junctions where struts split or meet.
    """
    sel = net.k_scale > k_thresh
    if sel.sum() < 3:
        return 0
    deg = {}
    for a, b in net.edges[sel]:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return int(sum(1 for d in deg.values() if d >= 3))


def coverage(net, domain, radius=12.0, n=3000, k_thresh=1.4, seed=2):
    """Fraction of the domain within `radius` of a reinforced strut.

    High when the strut network SPANS the area (divergent reach), low when ECM is
    confined to blobs — distinguishes a spanning reticular web from local clumps.
    """
    sel = net.k_scale > k_thresh
    if sel.sum() < 2:
        return 0.0
    p1 = net.nodes[net.edges[sel, 0]][:, :2]
    p2 = net.nodes[net.edges[sel, 1]][:, :2]
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, domain, size=(n, 2))
    d = _m.point_segment_distance(pts, p1, p2).min(axis=1)
    return float(np.mean(d < radius))


def density_profile(net, center, domain, k_thresh=1.4, nbins=12):
    """Reinforced-fiber length per unit area vs distance from `center`."""
    _, _, mid, _, L = _edges_xy(net)
    sel = net.k_scale > k_thresh
    if sel.sum() < 3:
        return np.array([]), np.array([])
    r = np.linalg.norm(mid[sel] - np.asarray(center)[:2], axis=1)
    rmax = domain * 0.5
    bins = np.linspace(0, rmax, nbins + 1)
    idx = np.clip(np.digitize(r, bins) - 1, 0, nbins - 1)
    length = np.zeros(nbins)
    for i, l in zip(idx, L[sel]):
        length[i] += l
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    centers = 0.5 * (bins[1:] + bins[:-1])
    return centers, length / area
