"""Export a simulation result to a compact JSON scene for the three.js viewer.

Schema (viewer/scene.json):

  {
    "domain": 200.0,
    "scalar": "state" | "tension" | "reinforce",
    "fibers": [ {"path": [[x,y,z], ...], "v": <scalar 0..1>, "r": <radius> }, ... ],
    "cells":  [ {"p": [x,y,z], "r": <radius>, "pheno": "<name>"}, ... ]
  }

`fibers[].v` is a normalized 0..1 value the viewer maps to color (fiber state,
tension, or plastic reinforcement). Polylines keep filaments continuous so the
viewer tubes them as smooth curves.
"""

from __future__ import annotations

import json

import numpy as np


def _norm(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return v
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi - lo < 1e-9:
        return np.clip(v * 0 + 0.5, 0, 1)
    return np.clip((v - lo) / (hi - lo), 0, 1)


def from_polylines(path, nodes, filaments, cells, domain,
                   radius=0.5, scalar="state", values=None):
    """Export filament polylines (e.g. a grown network)."""
    nodes = np.asarray(nodes)
    norm = _norm(values) if values is not None else None
    fibers = []
    for k, fil in enumerate(filaments):
        if len(fil) < 2:
            continue
        v = float(norm[k]) if norm is not None else 0.5
        fibers.append(dict(path=[nodes[i].tolist() for i in fil], v=v, r=radius))
    _write(path, domain, scalar, fibers, cells)


def from_network(path, net, cells, domain, scalar="tension", values=None,
                 p_mech=None, radius=0.6):
    """Export a FiberNetwork (mechanics/remodel) — each edge is a short polyline.

    `values` per edge (else derived): scalar='tension' uses current tension,
    'reinforce' uses k_scale.
    """
    from . import ecm_mechanics as mech
    if values is None:
        if scalar == "reinforce":
            values = net.k_scale
        else:
            cell_arr = np.asarray([[c["p"][0], c["p"][1], c["p"][2]] for c in cells]) \
                if cells and isinstance(cells[0], dict) else np.zeros((0, 3))
            _, tension, _ = mech.compute_forces(net, cell_arr, p_mech or mech.MechParams())
            values = np.clip(tension, 0, None)
    norm = _norm(values)
    fibers = []
    for e, v in zip(net.edges, norm):
        pa, pb = net.nodes[e[0]].tolist(), net.nodes[e[1]].tolist()
        fibers.append(dict(path=[pa, pb], v=float(v), r=radius))
    _write(path, domain, scalar, fibers, cells)


def cells_payload(positions, radius, phenotypes):
    return [dict(p=[float(p[0]), float(p[1]), float(p[2])], r=float(r), pheno=str(s))
            for p, r, s in zip(positions, radius, phenotypes)]


# --------------------------------------------------------------------------- #
# Multi-metric export: every per-fiber scalar + scene-level stats in one scene.
# --------------------------------------------------------------------------- #
_STATE_VALUE = {"nascent": 0.12, "mature": 0.34, "aligned": 0.72,
                "reinforced": 0.85, "degraded": 0.0, "crosslink": 0.95}

# Per-fiber scalars the viewer can color by (key -> title, low label, high label).
_SCALAR_INFO = {
    "reinforce": ("Collagen — reinforcement", "soft", "reinforced"),
    "tension":   ("Collagen — tension", "slack", "high tension"),
    "maturity":  ("Collagen — fiber maturity", "nascent", "crosslinked"),
    "length":    ("Collagen — segment length", "short", "long"),
    "alignment": ("Collagen — alignment to main axis", "off-axis", "on-axis"),
}

# Scene-level (whole-matrix) metrics, with display labels for the stats panel.
_STAT_LABELS = {
    "n_fibers": "Fibers",
    "n_cells": "Cells",
    "volume_fraction": "ECM volume fraction",
    "alignment_index": "Global alignment index",
    "local_alignment": "Local alignment (near cells)",
    "mean_pore": "Mean pore size (µm)",
    "median_pore": "Median pore size (µm)",
    "mean_contacts": "Mean cell–ECM contacts",
    "fraction_in_contact": "Cells in contact with ECM",
    "n_components": "Network components",
    "largest_component": "Largest connected fraction",
}


def _fibers_dict_from_network(net, phys_radius):
    """Build the metrics.py 'fibers' schema straight from a FiberNetwork."""
    p1 = net.nodes[net.edges[:, 0]]
    p2 = net.nodes[net.edges[:, 1]]
    seg = p2 - p1
    length = np.linalg.norm(seg, axis=1)
    safe = np.where(length[:, None] > 1e-9, seg, np.array([1.0, 0, 0]))
    direction = safe / np.where(length[:, None] > 1e-9, length[:, None], 1.0)
    return dict(p1=p1, p2=p2, center=0.5 * (p1 + p2), direction=direction,
                length=length, radius=np.full(len(net.edges), phys_radius),
                state=np.asarray(net.state))


def from_network_metrics(path, net, cells, cells_xyz, domain, p_mech=None,
                         default="reinforce", radius=0.85, phys_radius=0.5):
    """Export a relaxed FiberNetwork with ALL per-fiber scalars + scene stats.

    `cells` is the cells_payload() list (for rendering); `cells_xyz` is the raw
    (N,3) cell-center array (for the contact / alignment metrics).
    """
    from . import ecm_mechanics as mech
    from . import metrics as M

    edges = net.edges
    p1 = net.nodes[edges[:, 0]]
    p2 = net.nodes[edges[:, 1]]
    seg = p2 - p1
    length = np.linalg.norm(seg, axis=1)
    direction = np.where(length[:, None] > 1e-9, seg / np.where(length[:, None] > 1e-9, length[:, None], 1.0), 0.0)

    # --- per-edge raw scalars ---
    reinforce = np.asarray(net.k_scale, float)
    cell_arr = np.asarray(cells_xyz, float).reshape(-1, 3)
    _, tension, _ = mech.compute_forces(net, cell_arr, p_mech or mech.MechParams())
    tension = np.clip(tension, 0, None)
    maturity = np.array([_STATE_VALUE.get(str(s), 0.34) for s in net.state], float)

    fdict = _fibers_dict_from_network(net, phys_radius)
    aniso = M.fiber_anisotropy(fdict)
    axis = np.asarray(aniso["dominant_axis"], float)
    alignment = np.abs(direction @ axis) if len(direction) else np.zeros(0)

    raw = dict(reinforce=reinforce, tension=tension, maturity=maturity,
               length=length, alignment=alignment)
    norm = {k: _norm(v) for k, v in raw.items()}

    # --- build fibers with every normalized scalar attached ---
    fibers = []
    for i in range(len(edges)):
        vals = {k: float(norm[k][i]) for k in raw}
        fibers.append(dict(path=[p1[i].tolist(), p2[i].tolist()],
                           v=vals[default], r=radius, vals=vals))

    # --- scene-level metrics (reuse metrics.py) ---
    cells_m = dict(pos=cell_arr, radius=np.full(len(cell_arr), 9.0),
                   phenotype=np.array([c.get("pheno", "") for c in cells]),
                   vel=np.zeros((len(cell_arr), 3)))
    pores = M.pore_size_distribution(fdict, domain)
    conn = M.network_connectivity(fdict)
    contact = M.cell_ecm_contact(cells_m, fdict)
    stats = dict(
        n_fibers=len(edges),
        n_cells=len(cell_arr),
        volume_fraction=M.ecm_volume_fraction(fdict, domain),
        alignment_index=aniso["alignment_index"],
        local_alignment=M.local_alignment_index(cells_m, fdict),
        mean_pore=float(np.mean(pores)) if len(pores) else float("nan"),
        median_pore=float(np.median(pores)) if len(pores) else float("nan"),
        mean_contacts=contact["mean_contacts"],
        fraction_in_contact=contact["fraction_in_contact"],
        n_components=conn["n_components"],
        largest_component=conn["largest_fraction"],
    )

    scalars = [dict(key=k, title=_SCALAR_INFO[k][0], lo=_SCALAR_INFO[k][1],
                    hi=_SCALAR_INFO[k][2]) for k in raw]
    scene = dict(domain=float(domain), scalar=default, scalars=scalars,
                 fibers=fibers, cells=list(cells), stats=stats,
                 stat_labels=_STAT_LABELS)
    with open(path, "w") as fh:
        json.dump(scene, fh)
    return scene


def _catmull_rom(points, subdiv=4):
    """Smooth a polyline with a Catmull-Rom spline.

    Turns the jointed growth filament into a continuous organic curve so the
    viewer/GLB renders flowing collagen, not faceted sticks.
    """
    pts = np.asarray(points, float)
    if len(pts) < 3 or subdiv <= 1:
        return pts
    ext = np.vstack([pts[0] + (pts[0] - pts[1]) * 0.5, pts,
                     pts[-1] + (pts[-1] - pts[-2]) * 0.5])
    out = []
    ts = np.linspace(0, 1, subdiv, endpoint=False)
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in ts:
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.asarray(out)


def from_polylines_metrics(path, net, filaments, cells, cells_xyz, domain,
                           p_mech=None, default="reinforce", radius=0.5,
                           phys_radius=0.4, smooth=4):
    """Export curved filament polylines with ALL per-fiber scalars + scene stats.

    Like `from_network_metrics`, but keeps each filament as a smooth curve
    (Catmull-Rom) so the matrix renders as flowing collagen instead of straight
    rods. Per-filament scalar = mean of that scalar over the filament's edges.
    """
    from . import ecm_mechanics as mech
    from . import metrics as M

    nodes = np.asarray(net.nodes, float)
    edges = np.asarray(net.edges)
    seg = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    length = np.linalg.norm(seg, axis=1)
    direction = np.where(length[:, None] > 1e-9,
                         seg / np.where(length[:, None] > 1e-9, length[:, None], 1.0), 0.0)

    reinforce = np.asarray(net.k_scale, float)
    cell_arr = np.asarray(cells_xyz, float).reshape(-1, 3)
    _, tension, _ = mech.compute_forces(net, cell_arr, p_mech or mech.MechParams())
    tension = np.clip(tension, 0, None)
    maturity = np.array([_STATE_VALUE.get(str(s), 0.34) for s in net.state], float)

    fdict = _fibers_dict_from_network(net, phys_radius)
    aniso = M.fiber_anisotropy(fdict)
    axis = np.asarray(aniso["dominant_axis"], float)
    alignment = np.abs(direction @ axis) if len(direction) else np.zeros(0)

    raw = dict(reinforce=reinforce, tension=tension, maturity=maturity,
               length=length, alignment=alignment)
    norm = {k: _norm(v) for k, v in raw.items()}

    # edge lookup so each filament can average the scalars of its own segments
    eidx = {}
    for i, (a, b) in enumerate(edges):
        eidx[(int(a), int(b))] = i
        eidx[(int(b), int(a))] = i

    fibers = []
    for fil in filaments:
        if len(fil) < 2:
            continue
        segs = [eidx.get((int(fil[i]), int(fil[i + 1]))) for i in range(len(fil) - 1)]
        segs = [s for s in segs if s is not None]
        if not segs:
            continue
        vals = {k: float(np.mean(norm[k][segs])) for k in raw}
        pts = _catmull_rom(nodes[np.asarray(fil)], subdiv=smooth)
        fibers.append(dict(path=pts.tolist(), v=vals[default], r=radius, vals=vals))

    cells_m = dict(pos=cell_arr, radius=np.full(len(cell_arr), 9.0),
                   phenotype=np.array([c.get("pheno", "") for c in cells]),
                   vel=np.zeros((len(cell_arr), 3)))
    pores = M.pore_size_distribution(fdict, domain)
    conn = M.network_connectivity(fdict)
    contact = M.cell_ecm_contact(cells_m, fdict)
    stats = dict(
        n_fibers=len(fibers),
        n_cells=len(cell_arr),
        volume_fraction=M.ecm_volume_fraction(fdict, domain),
        alignment_index=aniso["alignment_index"],
        local_alignment=M.local_alignment_index(cells_m, fdict),
        mean_pore=float(np.mean(pores)) if len(pores) else float("nan"),
        median_pore=float(np.median(pores)) if len(pores) else float("nan"),
        mean_contacts=contact["mean_contacts"],
        fraction_in_contact=contact["fraction_in_contact"],
        n_components=conn["n_components"],
        largest_component=conn["largest_fraction"],
    )
    scalars = [dict(key=k, title=_SCALAR_INFO[k][0], lo=_SCALAR_INFO[k][1],
                    hi=_SCALAR_INFO[k][2]) for k in raw]
    scene = dict(domain=float(domain), scalar=default, scalars=scalars,
                 fibers=fibers, cells=list(cells), stats=stats,
                 stat_labels=_STAT_LABELS)
    with open(path, "w") as fh:
        json.dump(scene, fh)
    return scene


def _write(path, domain, scalar, fibers, cells):
    scene = dict(domain=float(domain), scalar=scalar, fibers=fibers,
                 cells=list(cells))
    with open(path, "w") as fh:
        json.dump(scene, fh)
    return scene
