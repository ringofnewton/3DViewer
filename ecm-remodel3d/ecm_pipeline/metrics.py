"""Quantitative ECM-architecture metrics.

Every function consumes the plain CSV schema written by `synthetic.write_csv`
(or by a real PhysiCell/PhysiMeSS parser). The eight metrics mirror the
"정량분석 metric" table in the decision map:

  1. ecm_volume_fraction       2. fiber_anisotropy
  3. local_alignment_index     4. pore_size_distribution
  5. cell_ecm_contact          6. migration_persistence
  7. remodeling_rate           8. network_connectivity
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_cells(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    out = dict(
        id=np.array([int(r["id"]) for r in rows]),
        pos=np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows]).reshape(n, 3),
        radius=np.array([float(r["radius"]) for r in rows]),
        phenotype=np.array([r["phenotype"] for r in rows]),
        vel=np.array([[float(r["vx"]), float(r["vy"]), float(r["vz"])] for r in rows]).reshape(n, 3),
    )
    return out


def load_fibers(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    p1 = np.array([[float(r["x1"]), float(r["y1"]), float(r["z1"])] for r in rows]).reshape(n, 3)
    p2 = np.array([[float(r["x2"]), float(r["y2"]), float(r["z2"])] for r in rows]).reshape(n, 3)
    seg = p2 - p1
    length = np.linalg.norm(seg, axis=1)
    safe = np.where(length[:, None] > 1e-9, seg, np.array([1.0, 0, 0]))
    direction = safe / np.where(length[:, None] > 1e-9, length[:, None], 1.0)
    return dict(
        id=np.array([int(r["id"]) for r in rows]),
        p1=p1, p2=p2, center=0.5 * (p1 + p2),
        direction=direction, length=length,
        radius=np.array([float(r["radius"]) for r in rows]),
        state=np.array([r["state"] for r in rows]),
    )


def frame_files(outdir: str):
    cells = sorted(glob.glob(os.path.join(outdir, "cells_t*.csv")))
    fibers = sorted(glob.glob(os.path.join(outdir, "fibers_t*.csv")))
    return list(zip(cells, fibers))


# --------------------------------------------------------------------------- #
# Geometry helper: distance from points to a set of segments
# --------------------------------------------------------------------------- #
def point_segment_distance(points: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return (n_points, n_segments) Euclidean distances."""
    seg = p2 - p1                                  # (S,3)
    seg_len2 = np.einsum("ij,ij->i", seg, seg)     # (S,)
    seg_len2 = np.where(seg_len2 < 1e-9, 1e-9, seg_len2)
    diff = points[:, None, :] - p1[None, :, :]     # (P,S,3)
    t = np.einsum("psj,sj->ps", diff, seg) / seg_len2[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = p1[None, :, :] + t[:, :, None] * seg[None, :, :]
    d = points[:, None, :] - proj
    return np.sqrt(np.einsum("psj,psj->ps", d, d))


# --------------------------------------------------------------------------- #
# 1. ECM volume fraction
# --------------------------------------------------------------------------- #
def ecm_volume_fraction(fibers: dict, domain: float) -> float:
    vol = np.sum(np.pi * fibers["radius"] ** 2 * fibers["length"])
    return float(vol / domain ** 3)


# --------------------------------------------------------------------------- #
# 2. Fiber anisotropy (orientation tensor)
# --------------------------------------------------------------------------- #
def orientation_tensor(directions: np.ndarray, weights: np.ndarray | None = None):
    if len(directions) == 0:
        return np.full(3, np.nan), np.eye(3)
    if weights is None:
        weights = np.ones(len(directions))
    w = weights / weights.sum()
    tensor = np.einsum("i,ij,ik->jk", w, directions, directions)
    vals, vecs = np.linalg.eigh(tensor)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def fiber_anisotropy(fibers: dict) -> dict:
    vals, vecs = orientation_tensor(fibers["direction"], fibers["length"])
    l1 = vals[0]
    # alignment index: 0 = isotropic (1/3,1/3,1/3), 1 = perfectly aligned
    alignment = float((3 * l1 - 1) / 2) if np.isfinite(l1) else float("nan")
    return dict(eigenvalues=vals, dominant_axis=vecs[:, 0], alignment_index=alignment)


# --------------------------------------------------------------------------- #
# 3. Local alignment index (per-cell neighborhood)
# --------------------------------------------------------------------------- #
def local_alignment_index(cells: dict, fibers: dict, radius: float = 26.0) -> float:
    if len(fibers["center"]) == 0 or len(cells["pos"]) == 0:
        return float("nan")
    vals = []
    for c in cells["pos"]:
        d = fibers["center"] - c
        near = np.einsum("ij,ij->i", d, d) < radius * radius
        if np.count_nonzero(near) >= 3:
            ev, _ = orientation_tensor(fibers["direction"][near], fibers["length"][near])
            vals.append((3 * ev[0] - 1) / 2)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# 4. Pore size distribution (Monte-Carlo nearest-fiber sampling)
# --------------------------------------------------------------------------- #
def pore_size_distribution(fibers: dict, domain: float, n_samples: int = 3000,
                           seed: int = 0) -> np.ndarray:
    if len(fibers["p1"]) == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, domain, size=(n_samples, 3))
    dist = point_segment_distance(pts, fibers["p1"], fibers["p2"])
    nearest = dist.min(axis=1) - fibers["radius"].min()
    return np.clip(nearest, 0, None)


# --------------------------------------------------------------------------- #
# 5. Cell-ECM contact frequency
# --------------------------------------------------------------------------- #
def cell_ecm_contact(cells: dict, fibers: dict, margin: float = 2.0) -> dict:
    if len(fibers["p1"]) == 0 or len(cells["pos"]) == 0:
        return dict(mean_contacts=float("nan"), fraction_in_contact=float("nan"))
    dist = point_segment_distance(cells["pos"], fibers["p1"], fibers["p2"])
    thresh = (cells["radius"][:, None] + fibers["radius"][None, :] + margin)
    contacts = (dist < thresh).sum(axis=1)
    return dict(mean_contacts=float(contacts.mean()),
                fraction_in_contact=float(np.mean(contacts > 0)))


# --------------------------------------------------------------------------- #
# 6. Migration persistence (from tracks.csv)
# --------------------------------------------------------------------------- #
def migration_persistence(tracks_csv: str) -> dict:
    rows = list(csv.DictReader(open(tracks_csv)))
    by_cell: dict[int, list] = {}
    for r in rows:
        by_cell.setdefault(int(r["cell_id"]), []).append(
            (int(r["frame"]), float(r["x"]), float(r["y"]), float(r["z"])))
    per_cell = []
    for cid, pts in by_cell.items():
        pts.sort()
        xyz = np.array([[x, y, z] for _, x, y, z in pts])
        if len(xyz) < 2:
            continue
        steps = np.diff(xyz, axis=0)
        path = np.sum(np.linalg.norm(steps, axis=1))
        net = np.linalg.norm(xyz[-1] - xyz[0])
        if path > 1e-9:
            per_cell.append(net / path)
    return dict(mean_persistence=float(np.mean(per_cell)) if per_cell else float("nan"),
                per_cell=np.array(per_cell))


# --------------------------------------------------------------------------- #
# 7. Remodeling rate (birth/death events from events.csv)
# --------------------------------------------------------------------------- #
def remodeling_rate(events_csv: str) -> dict:
    rows = list(csv.DictReader(open(events_csv)))
    births = np.array([int(r["births"]) for r in rows])
    deaths = np.array([int(r["deaths"]) for r in rows])
    frames = np.array([int(r["frame"]) for r in rows])
    return dict(frame=frames, births=births, deaths=deaths,
                turnover=births + deaths)


# --------------------------------------------------------------------------- #
# 8. Network connectivity (fiber endpoint graph)
# --------------------------------------------------------------------------- #
def network_connectivity(fibers: dict, crosslink_dist: float = 4.0) -> dict:
    n = len(fibers["p1"])
    if n == 0:
        return dict(n_components=0, largest_fraction=float("nan"))
    endpoints = np.concatenate([fibers["p1"], fibers["p2"]], axis=0)  # (2n,3)
    owner = np.concatenate([np.arange(n), np.arange(n)])

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # spatial hashing to avoid O(n^2): bucket endpoints into a grid
    cell = crosslink_dist
    buckets: dict[tuple, list[int]] = {}
    keys = np.floor(endpoints / cell).astype(int)
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    neigh = [(-1, 0, 1)] * 3
    import itertools
    for k, idxs in buckets.items():
        cand = []
        for dk in itertools.product(*neigh):
            cand += buckets.get((k[0] + dk[0], k[1] + dk[1], k[2] + dk[2]), [])
        for a in idxs:
            for b in cand:
                if b <= a:
                    continue
                if owner[a] == owner[b]:
                    continue
                if np.sum((endpoints[a] - endpoints[b]) ** 2) < cell * cell:
                    union(owner[a], owner[b])
    roots = {find(i) for i in range(n)}
    sizes: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    largest = max(sizes.values())
    return dict(n_components=len(roots), largest_fraction=float(largest / n))


# --------------------------------------------------------------------------- #
# Time-series driver
# --------------------------------------------------------------------------- #
def compute_time_series(outdir: str, domain: float) -> list[dict]:
    rows = []
    for fi, (cpath, fpath) in enumerate(frame_files(outdir)):
        cells = load_cells(cpath)
        fibers = load_fibers(fpath)
        aniso = fiber_anisotropy(fibers)
        contact = cell_ecm_contact(cells, fibers)
        conn = network_connectivity(fibers)
        rows.append(dict(
            frame=fi,
            n_cells=len(cells["pos"]),
            n_fibers=len(fibers["center"]),
            volume_fraction=ecm_volume_fraction(fibers, domain),
            alignment_index=aniso["alignment_index"],
            local_alignment=local_alignment_index(cells, fibers),
            mean_contacts=contact["mean_contacts"],
            fraction_in_contact=contact["fraction_in_contact"],
            largest_component=conn["largest_fraction"],
            n_components=conn["n_components"],
        ))
    return rows


def write_time_series(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
