#!/usr/bin/env python3
"""Convert a viewer scene (viewer/scene.json) into a GLB model.

The GLB can be opened directly in the main 3DViewer (index.html, which uses
Google <model-viewer>) and uploaded to the shared link. Fibers become tubes
colored by the same colormap the three.js viewer uses; cells become spheres
tinted by phenotype.

    python export_glb.py                         # viewer/scene.json -> viewer/model.glb
    python export_glb.py scene.json out.glb      # explicit paths
    python export_glb.py scene.json out.glb tension   # color by a chosen scalar
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))

# Same stops as viewer/index.html colormap() (inferno-like), as (t, r, g, b).
_CMAP_STOPS = np.array([
    [0.00, 0.02, 0.02, 0.23],
    [0.25, 0.42, 0.07, 0.43],
    [0.50, 0.74, 0.21, 0.33],
    [0.75, 0.95, 0.50, 0.16],
    [1.00, 0.99, 0.85, 0.55],
])

# Same phenotype tints as viewer/index.html PHENO_TINT (hex -> rgb 0..1).
_PHENO_TINT = {
    "producing":   (0x6f, 0xb0, 0xff),
    "contractile": (0xb9, 0x8c, 0xff),
    "degrading":   (0xff, 0x8d, 0x8d),
    "quiescent":   (0x9f, 0xe0, 0xd0),
}
_PHENO_DEFAULT = (0x8f, 0xb6, 0xff)


def colormap(t: float) -> np.ndarray:
    """Match the JS colormap: piecewise-linear over _CMAP_STOPS, returns rgb 0..1."""
    t = float(np.clip(t, 0.0, 1.0))
    ts = _CMAP_STOPS[:, 0]
    rgb = _CMAP_STOPS[:, 1:]
    i = int(np.searchsorted(ts, t, side="right") - 1)
    i = max(0, min(i, len(ts) - 2))
    u = (t - ts[i]) / (ts[i + 1] - ts[i]) if ts[i + 1] > ts[i] else 0.0
    return rgb[i] + (rgb[i + 1] - rgb[i]) * u


def _rgba(rgb01, alpha=255) -> np.ndarray:
    return np.array([*(np.clip(np.asarray(rgb01) * 255, 0, 255)), alpha], dtype=np.uint8)


def fiber_mesh(path, radius, rgb01):
    """A tube along a polyline as a chain of colored cylinders."""
    pts = np.asarray(path, dtype=float)
    color = _rgba(rgb01)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        if np.linalg.norm(b - a) < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, segment=[a, b], sections=8)
        cyl.visual.vertex_colors = np.tile(color, (len(cyl.vertices), 1))
        segs.append(cyl)
    return segs


def cell_mesh(center, radius, pheno):
    rgb = np.array(_PHENO_TINT.get(pheno, _PHENO_DEFAULT)) / 255.0
    sph = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sph.apply_translation(center)
    sph.visual.vertex_colors = np.tile(_rgba(rgb), (len(sph.vertices), 1))
    return sph


def fiber_value(f, key):
    vals = f.get("vals")
    if key and vals and key in vals:
        return vals[key]
    return f.get("v", 0.5)


def build(scene: dict, scalar=None) -> trimesh.Trimesh:
    key = scalar or scene.get("scalar")
    parts = []
    for f in scene["fibers"]:
        if len(f.get("path", [])) < 2:
            continue
        parts.extend(fiber_mesh(f["path"], float(f.get("r", 0.5)), colormap(fiber_value(f, key))))
    for c in scene.get("cells", []):
        parts.append(cell_mesh(np.asarray(c["p"], dtype=float), float(c["r"]), c.get("pheno", "")))

    if not parts:
        raise SystemExit("scene has no drawable geometry")

    mesh = trimesh.util.concatenate(parts)
    # Center on origin so <model-viewer> frames it nicely.
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "viewer", "scene.json")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "viewer", "model.glb")
    scalar = sys.argv[3] if len(sys.argv) > 3 else None

    with open(src) as fh:
        scene = json.load(fh)

    key = scalar or scene.get("scalar")
    print(f"[1/2] building mesh from {os.path.relpath(src, HERE)} "
          f"({len(scene['fibers'])} fibers, {len(scene.get('cells', []))} cells, color={key}) …")
    mesh = build(scene, scalar)
    print(f"      {len(mesh.vertices)} vertices · {len(mesh.faces)} faces")

    print(f"[2/2] writing {os.path.relpath(dst, HERE)}")
    mesh.export(dst)
    size_kb = os.path.getsize(dst) / 1024
    print(f"done. {size_kb:.0f} KB GLB. Open the main index.html and choose this file.")


if __name__ == "__main__":
    main()
