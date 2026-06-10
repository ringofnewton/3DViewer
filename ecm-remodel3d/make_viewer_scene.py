#!/usr/bin/env python3
"""Generate viewer/scene.json for the three.js 3D viewer.

Grows a connected ECM network with cells and exports it as a web scene
(fibers as state-colored polylines, cells as organic translucent spheres).

    python make_viewer_scene.py
    # then serve and open the viewer:
    cd viewer && python -m http.server 8000   →   http://localhost:8000
"""

from __future__ import annotations

import os

import numpy as np

from ecm_pipeline import ecm_network, web_export

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "viewer", "scene.json")

# value (0..1) the colormap maps each fiber state to
STATE_VALUE = {"nascent": 0.12, "mature": 0.34, "aligned": 0.72,
               "degraded": 0.0, "crosslink": 0.95}


def main():
    print("[1/2] growing ECM network for the viewer …")
    params = ecm_network.NetworkParams(n_cells=28, steps=55, n_frames=2,
                                       max_nodes=6500, max_tips=260, seed=5)
    frames, _, p = ecm_network.simulate_network(params)
    fr = frames[-1]
    nodes = fr["nodes"]

    # per-filament colour value from the states of its edges
    edge_state = {}
    for a, b, st, _ in fr["edges"]:
        edge_state[(a, b)] = st
        edge_state[(b, a)] = st
    filaments, values = [], []
    for fil in fr["filaments"]:
        if len(fil) < 2:
            continue
        codes = [STATE_VALUE.get(edge_state.get((fil[i], fil[i + 1]), "mature"), 0.34)
                 for i in range(len(fil) - 1)]
        filaments.append(fil)
        values.append(float(np.mean(codes)))

    cells = web_export.cells_payload(
        np.array([[c["x"], c["y"], c["z"]] for c in fr["cells"]]),
        np.array([c["radius"] for c in fr["cells"]]),
        [c["phenotype"] for c in fr["cells"]])

    print(f"      {len(filaments)} filaments · {len(cells)} cells")
    print("[2/2] writing", OUT)
    web_export.from_polylines(OUT, nodes, filaments, cells, p.domain,
                              radius=0.7, scalar="state",
                              values=np.array(values) if values else None)
    print("done. serve viewer/ and open index.html")


if __name__ == "__main__":
    main()
