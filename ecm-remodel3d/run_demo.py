#!/usr/bin/env python3
"""End-to-end demo of the ECM remodeling analysis pipeline.

    python run_demo.py

Runs the synthetic generator, writes the CSV schema, computes the full metric
time series, exports ParaView VTP + a web-viewer JSON for the final frame, and
renders the publication figures into ./output and ./output/figures.

Swap `synthetic.simulate` for a real PhysiCell/PhysiMeSS parser that emits the
same cells_t*.csv / fibers_t*.csv / tracks.csv / events.csv files, and every
other step is unchanged.
"""

from __future__ import annotations

import os

from ecm_pipeline import synthetic, metrics, export_vtk, figures

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")
FIGDIR = os.path.join(OUTDIR, "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    # 1. Generate synthetic remodeling data ------------------------------- #
    print("[1/5] simulating synthetic ECM remodeling …")
    frames, tracks, params = synthetic.simulate()
    synthetic.write_csv(frames, tracks, OUTDIR)
    print(f"      {len(frames)} frames, domain={params.domain} µm, "
          f"{len(frames[-1]['fibers'])} fibers at final frame")

    # 2. Quantitative metric time series ---------------------------------- #
    print("[2/5] computing structural metrics …")
    ts = metrics.compute_time_series(OUTDIR, params.domain)
    metrics.write_time_series(ts, os.path.join(OUTDIR, "metrics_timeseries.csv"))
    last = ts[-1]
    print(f"      final: volfrac={last['volume_fraction']:.2e}  "
          f"alignment={last['alignment_index']:.3f}  "
          f"contacts/cell={last['mean_contacts']:.1f}  "
          f"largest_net={last['largest_component']:.2f}")

    pers = metrics.migration_persistence(os.path.join(OUTDIR, "tracks.csv"))
    print(f"      migration persistence (mean) = {pers['mean_persistence']:.3f}")

    # 3. Export final frame for ParaView + web viewer --------------------- #
    print("[3/5] exporting VTP + scene JSON …")
    last_cells_path, last_fibers_path = metrics.frame_files(OUTDIR)[-1]
    cells = metrics.load_cells(last_cells_path)
    fibers = metrics.load_fibers(last_fibers_path)
    export_vtk.write_fibers_vtp(fibers, os.path.join(OUTDIR, "fibers_final.vtp"))
    export_vtk.write_cells_vtp(cells, os.path.join(OUTDIR, "cells_final.vtp"))
    export_vtk.export_scene_json(cells, fibers, os.path.join(OUTDIR, "scene_final.json"))

    # 4. Publication figures ---------------------------------------------- #
    print("[4/5] rendering figures …")
    figures.fig_time_series(ts, os.path.join(FIGDIR, "fig1_time_series.png"))
    figures.fig_orientation_rose(fibers, os.path.join(FIGDIR, "fig2_orientation_rose.png"))
    figures.fig_pore_size(fibers, params.domain, os.path.join(FIGDIR, "fig3_pore_size.png"))
    figures.fig_remodeling(os.path.join(OUTDIR, "events.csv"),
                           os.path.join(FIGDIR, "fig4_remodeling.png"))
    figures.fig_persistence(os.path.join(OUTDIR, "tracks.csv"),
                            os.path.join(FIGDIR, "fig5_persistence.png"))
    figures.fig_scene_3d(cells, fibers, os.path.join(FIGDIR, "fig6_scene3d.png"))

    # 5. Done ------------------------------------------------------------- #
    print("[5/5] done. Outputs in", OUTDIR)
    print("      figures/  : 6 publication PNGs")
    print("      *.vtp     : open in ParaView (Tube filter on fibers, Glyph on cells)")
    print("      scene_final.json : feed to the web 3D viewer")


if __name__ == "__main__":
    main()
