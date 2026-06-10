"""Cell seeding patterns — including DLP-printed gel regions.

The experiment: cells suspended in GelMA, then a DLP printer gels only chosen
regions, so cells are patterned in space. These helpers place contractile-cell
positions (M,3 arrays in the z=Z plane) by density and pattern, which the
mechanics/remodeling then turns into different ECM morphologies.

All return float arrays of shape (M,3).
"""

from __future__ import annotations

import numpy as np


def uniform(domain=220.0, n=30, z=110.0, margin=24.0, seed=0):
    """n cells, uniform random — density set by n."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(margin, domain - margin, size=(n, 2))
    return np.column_stack([xy, np.full(n, z)])


def jittered_grid(domain=220.0, spacing=34.0, z=110.0, jitter=0.28, margin=24.0, seed=0):
    """Evenly spread cells on a jittered grid — clean density control."""
    rng = np.random.default_rng(seed)
    xs = np.arange(margin, domain - margin + 1e-6, spacing)
    pts = []
    for x in xs:
        for y in xs:
            j = (rng.random(2) - 0.5) * jitter * spacing
            pts.append([x + j[0], y + j[1], z])
    return np.array(pts)


def line_cells(p0, p1, n=8, z=110.0, jitter=2.5, seed=0):
    """Cells along a line — e.g. a central arteriole."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)[:, None]
    pts = np.asarray(p0)[:2] * (1 - t) + np.asarray(p1)[:2] * t
    pts = pts + rng.normal(0, jitter, pts.shape)
    return np.column_stack([pts, np.full(n, z)])


def annulus(center, r_in, r_out, n=40, z=110.0, seed=0):
    """Cells in a ring — e.g. a periarteriolar sheath or marginal zone."""
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))
    th = rng.uniform(0, 2 * np.pi, n)
    xy = np.asarray(center)[:2] + np.column_stack([rr * np.cos(th), rr * np.sin(th)])
    return np.column_stack([xy, np.full(n, z)])


def spheroid(center, n_cells=20, radius=13.0, z=110.0, seed=0):
    """A spheroid = a dense disk-shaped cluster of cells (not a single point)."""
    rng = np.random.default_rng(seed)
    rr = radius * np.sqrt(rng.random(n_cells))
    th = rng.uniform(0, 2 * np.pi, n_cells)
    xy = np.asarray(center)[:2] + np.column_stack([rr * np.cos(th), rr * np.sin(th)])
    return np.column_stack([xy, np.full(n_cells, z)])


def spheroids(centers, n_cells=20, radius=13.0, z=110.0, seed=0):
    """Several spheroids; returns the concatenated cell positions."""
    parts = [spheroid(c, n_cells, radius, z, seed + i) for i, c in enumerate(centers)]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))


def in_mask(domain, n, mask_fn, z=110.0, margin=12.0, seed=0, max_tries=20000):
    """Seed n cells uniformly inside the region where mask_fn(x,y) is True.

    This is the DLP-printed gel: cells only survive where the gel was cured.
    """
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = rng.uniform(margin, domain - margin, size=2)
        tries += 1
        if mask_fn(p[0], p[1]):
            pts.append([p[0], p[1], z])
    return np.array(pts) if pts else np.zeros((0, 3))


# ---- DLP mask shapes (return a mask_fn(x,y) -> bool) ----------------------- #
def stripes_mask(domain, period=46.0, duty=0.5, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    def f(x, y):
        u = (x - domain / 2) * c + (y - domain / 2) * s
        return (u % period) < duty * period
    return f


def disk_mask(domain, cx=None, cy=None, r=60.0):
    cx = domain / 2 if cx is None else cx
    cy = domain / 2 if cy is None else cy
    def f(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 < r * r
    return f


def ring_mask(domain, r_in=40.0, r_out=70.0):
    c = domain / 2
    def f(x, y):
        d2 = (x - c) ** 2 + (y - c) ** 2
        return r_in * r_in < d2 < r_out * r_out
    return f


def two_blocks_mask(domain, gap=40.0, block=44.0):
    c = domain / 2
    def f(x, y):
        return (abs(y - c) < block) and (abs(abs(x - c) - (gap / 2 + block / 2)) < block / 2)
    return f
