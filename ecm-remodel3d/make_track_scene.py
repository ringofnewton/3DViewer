#!/usr/bin/env python3
"""Export the cell-built collagen TRACK to the 3D viewer (viewer/scene.json).

Places contractile cells in a 3D collagen network, runs plastic remodeling, and
exports the result colored by plastic reinforcement — so the viewer shows the
load-bearing collagen tracks the cells built between themselves, in 3D.

    python make_track_scene.py
    cd viewer && python -m http.server 8000   →   http://localhost:8000
"""

from __future__ import annotations

import os

import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem, web_export

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "viewer", "scene.json")
DOMAIN = 200.0


def main():
    c = DOMAIN / 2
    # four contractile cells at the corners of a tetrahedron (6 pairwise tracks)
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cells_xyz = c + 34.0 * dirs

    print("[1/3] building 3D collagen network …")
    net0 = mech.build_random_network(DOMAIN, n_fibers=340, fiber_len=70,
                                     seg_len=7.0, crosslink_dist=4.5, seed=5)
    P = mech.MechParams(reach=34, reach_decay=20)

    print("[2/3] plastic remodeling under cell traction …")
    RP = rem.RemodelParams(n_steps=9, compaction=0.05, reinforce_rate=0.4,
                           slack_pct=30, slack_steps=2)
    frames, final = rem.remodel(net0, cells_xyz, P, RP)
    print(f"      fibers {frames[0]['n_edges']} → {len(final.edges)}  "
          f"({float(np.mean(final.k_scale > 1.4)):.0%} reinforced)")

    cells = web_export.cells_payload(
        cells_xyz, np.full(len(cells_xyz), 9.0),
        ["contractile"] * len(cells_xyz))

    print("[3/3] exporting", OUT, "(all per-fiber scalars + scene stats)")
    web_export.from_network_metrics(OUT, final, cells, cells_xyz, DOMAIN,
                                    p_mech=P, default="reinforce", radius=0.85)
    print("done. serve viewer/ and open index.html")


if __name__ == "__main__":
    main()
