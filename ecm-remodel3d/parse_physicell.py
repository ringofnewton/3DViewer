#!/usr/bin/env python3
"""Convert real PhysiCell + PhysiMeSS output into metrics, figures, and 3D export.

    python parse_physicell.py <physicell_output_dir> [<out_dir>]

<physicell_output_dir>  folder containing outputNNNNNNNN.xml + _cells.mat snapshots
<out_dir>               where CSVs / figures / VTP are written (default: ./output)

This is the real-data twin of run_demo.py: instead of the synthetic generator,
it parses MultiCellDS snapshots into the shared CSV schema, then runs the exact
same metrics + figure + export steps.

If your PhysiMeSS custom-data labels differ, edit ParseConfig in
ecm_pipeline/physicell_parser.py (fiber_type_id, length_field, …).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from ecm_pipeline import physicell_parser, metrics, export_vtk, figures


def infer_domain(out_dir: str) -> float:
    frames = metrics.frame_files(out_dir)
    hi = 0.0
    for cpath, fpath in frames:
        cells = metrics.load_cells(cpath)
        fibers = metrics.load_fibers(fpath)
        for arr in (cells["pos"], fibers["p1"], fibers["p2"]):
            if len(arr):
                hi = max(hi, float(np.max(arr)))
    return float(np.ceil(hi / 10.0) * 10.0) or 1.0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    physicell_dir = argv[1]
    out_dir = argv[2] if len(argv) > 2 else os.path.join(os.path.dirname(__file__), "output")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[1/5] parsing PhysiCell/PhysiMeSS snapshots in {physicell_dir} …")
    n = physicell_parser.convert_directory(physicell_dir, out_dir)
    domain = infer_domain(out_dir)
    print(f"      {n} frames parsed, inferred domain ≈ {domain:.0f} µm")

    print("[2/5] computing structural metrics …")
    ts = metrics.compute_time_series(out_dir, domain)
    metrics.write_time_series(ts, os.path.join(out_dir, "metrics_timeseries.csv"))
    last = ts[-1]
    print(f"      final: fibers={last['n_fibers']}  volfrac={last['volume_fraction']:.2e}  "
          f"alignment={last['alignment_index']:.3f}  contacts/cell={last['mean_contacts']:.1f}")

    print("[3/5] exporting VTP + scene JSON …")
    lc, lf = metrics.frame_files(out_dir)[-1]
    cells, fibers = metrics.load_cells(lc), metrics.load_fibers(lf)
    export_vtk.write_fibers_vtp(fibers, os.path.join(out_dir, "fibers_final.vtp"))
    export_vtk.write_cells_vtp(cells, os.path.join(out_dir, "cells_final.vtp"))
    export_vtk.export_scene_json(cells, fibers, os.path.join(out_dir, "scene_final.json"))

    print("[4/5] rendering figures …")
    figures.fig_time_series(ts, os.path.join(fig_dir, "fig1_time_series.png"))
    figures.fig_orientation_rose(fibers, os.path.join(fig_dir, "fig2_orientation_rose.png"))
    figures.fig_pore_size(fibers, domain, os.path.join(fig_dir, "fig3_pore_size.png"))
    figures.fig_remodeling(os.path.join(out_dir, "events.csv"),
                           os.path.join(fig_dir, "fig4_remodeling.png"))
    figures.fig_persistence(os.path.join(out_dir, "tracks.csv"),
                            os.path.join(fig_dir, "fig5_persistence.png"))
    figures.fig_scene_3d(cells, fibers, os.path.join(fig_dir, "fig6_scene3d.png"))

    print("[5/5] done. Outputs in", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
