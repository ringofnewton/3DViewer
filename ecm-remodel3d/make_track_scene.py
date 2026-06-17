#!/usr/bin/env python3
"""Export an ORGANIC, cell-built collagen network to the 3D viewer.

Grows a dense, curved, branching, crosslinked ECM network (ecm_network), then
relaxes it under contractile-cell traction so the cells compact and align the
matrix into load-bearing tracks. Filaments are kept as smooth curves (not
straight rods) and exported with every per-fiber scalar + scene metrics.

    python make_track_scene.py
    cd viewer && python serve.py   →   http://localhost:8000
"""

from __future__ import annotations

import os

import numpy as np

from ecm_pipeline import ecm_network, ecm_mechanics as mech, ecm_remodel as rem, web_export

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "viewer", "scene.json")
DOMAIN = 200.0


def frame_to_network(frame, domain, anchor_margin=8.0):
    """Build a mechanics FiberNetwork from a grown ecm_network frame, keeping the
    curved filaments as backbone bends so the geometry stays organic."""
    nodes = np.asarray(frame["nodes"], float)
    edges = np.array([[int(a), int(b)] for a, b, *_ in frame["edges"]])
    states = np.array([str(s) for *_, s, _b in
                       [(a, b, st, bi) for a, b, st, bi in frame["edges"]]], dtype=object)
    bends = []
    for fil in frame["filaments"]:
        for i in range(1, len(fil) - 1):
            bends.append((int(fil[i - 1]), int(fil[i]), int(fil[i + 1])))
    bends = np.array(bends) if bends else np.zeros((0, 3), int)
    rest = np.maximum(np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1), 1e-3)
    fixed = ((nodes < anchor_margin) | (nodes > domain - anchor_margin)).any(axis=1)
    return mech.FiberNetwork(nodes=nodes, edges=edges, rest_length=rest,
                             bends=bends, fixed=fixed, state=states)


def main():
    print("[1/4] growing organic ECM network (curved, branching, crosslinked) …")
    p = ecm_network.NetworkParams(
        n_cells=40, steps=95, n_frames=2, seg_length=2.2, persistence=0.66,
        guidance=0.5, branch_prob=0.09, crosslink_dist=5.0, secretion_prob=0.75,
        max_tips=650, max_nodes=24000, tip_lifetime=40, seed=7)
    frames, _, pp = ecm_network.simulate_network(p)
    fr = frames[-1]
    print(f"      {len(fr['filaments'])} filaments · {len(fr['nodes'])} nodes · "
          f"{len(fr['edges'])} edges · {len(fr['cells'])} cells")

    net = frame_to_network(fr, DOMAIN)
    cells = fr["cells"]
    cells_xyz_all = np.array([[c["x"], c["y"], c["z"]] for c in cells])
    pull = np.array([p_ == "contractile" for p_ in (c["phenotype"] for c in cells)])
    traction_xyz = cells_xyz_all[pull] if pull.any() else cells_xyz_all

    print("[2/4] relaxing matrix under contractile-cell traction (gentle) …")
    # gentle + capped iterations: build tracks without straightening the curves
    P = mech.MechParams(reach=26, reach_decay=16, traction=0.9, max_iter=500)
    # degradation off (slack_steps huge) so filament topology stays intact
    RP = rem.RemodelParams(n_steps=3, compaction=0.03, reinforce_rate=0.35,
                           slack_pct=20, slack_steps=999)
    _, final = rem.remodel(net, traction_xyz, P, RP)
    print(f"      {float(np.mean(final.k_scale > 1.4)):.0%} of fibers reinforced")

    cells_payload = web_export.cells_payload(
        cells_xyz_all, np.array([c["radius"] for c in cells]),
        [c["phenotype"] for c in cells])

    print("[3/4] smoothing filaments + computing all metrics …")
    print("[4/4] exporting", OUT)
    web_export.from_polylines_metrics(
        OUT, final, fr["filaments"], cells_payload, traction_xyz, DOMAIN,
        p_mech=P, default="alignment", radius=0.45, smooth=4)
    print("done. serve viewer/ and open index.html")


if __name__ == "__main__":
    main()
