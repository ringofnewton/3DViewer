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


def _write(path, domain, scalar, fibers, cells):
    scene = dict(domain=float(domain), scalar=scalar, fibers=fibers,
                 cells=list(cells))
    with open(path, "w") as fh:
        json.dump(scene, fh)
    return scene
