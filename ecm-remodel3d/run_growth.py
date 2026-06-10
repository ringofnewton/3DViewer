#!/usr/bin/env python3
"""Grow ECM as a connected network and visualize it forming over time.

    python run_growth.py

This is the network counterpart of run_demo.py. Instead of scattered straight
rods, it grows a branching, crosslinked, curved fiber network (see
ecm_pipeline/ecm_network.py), then runs the same metrics + export, and writes a
growth montage so you can SEE the matrix forming (web/mesh, not a buzz-cut).
"""

from __future__ import annotations

import os

from ecm_pipeline import ecm_network, metrics, export_vtk, figures

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output_network")
FIGDIR = os.path.join(OUTDIR, "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    print("[1/5] growing ECM network …")
    frames, tracks, params = ecm_network.simulate_network()
    ecm_network.write_csv(frames, tracks, OUTDIR)
    last = frames[-1]
    print(f"      {len(frames)} frames · {len(last['nodes'])} nodes · "
          f"{len(last['edges'])} segments · {len([f for f in last['filaments'] if len(f)>1])} filaments")

    print("[2/5] computing structural metrics …")
    ts = metrics.compute_time_series(OUTDIR, params.domain)
    metrics.write_time_series(ts, os.path.join(OUTDIR, "metrics_timeseries.csv"))
    m = ts[-1]
    print(f"      final: segments={m['n_fibers']}  alignment={m['alignment_index']:.3f}  "
          f"largest_connected_component={m['largest_component']:.2f}  "
          f"({m['n_components']} components)")

    print("[3/5] rendering growth visualization …")
    figures.fig_network_growth(frames, os.path.join(FIGDIR, "growth_montage.png"))
    figures.fig_network_final(last, os.path.join(FIGDIR, "network_final.png"))
    figures.fig_time_series(ts, os.path.join(FIGDIR, "metrics_time_series.png"))

    print("[4/5] exporting polyline VTP + graph JSON …")
    cells = metrics.load_cells(metrics.frame_files(OUTDIR)[-1][0])
    export_vtk.write_filaments_vtp(last["nodes"], last["filaments"],
                                   params.fiber_radius,
                                   os.path.join(OUTDIR, "network_final.vtp"))
    export_vtk.export_graph_scene_json(last["nodes"], last["filaments"], cells,
                                       os.path.join(OUTDIR, "network_final.json"))

    print("[5/5] done. Outputs in", OUTDIR)
    print("      figures/growth_montage.png  ← the matrix forming over time")
    print("      network_final.vtp           ← ParaView: Tube filter = continuous fibers")


if __name__ == "__main__":
    main()
