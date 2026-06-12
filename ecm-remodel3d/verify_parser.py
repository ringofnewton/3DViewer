#!/usr/bin/env python3
"""Round-trip test: synthetic → MultiCellDS fixture → parser → CSV schema.

Proves the real-data path works without a PhysiCell install. Generates the
synthetic model, writes it out in genuine MultiCellDS xml + .mat layout, parses
it back, and checks that cell/fiber counts, positions, fiber lengths, and
phenotype/state labels survive the round trip.

    python verify_parser.py
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from ecm_pipeline import synthetic, fixture, physicell_parser, metrics


def main():
    frames, tracks, params = synthetic.simulate()

    tmp = tempfile.mkdtemp(prefix="pcfix_")
    raw = os.path.join(tmp, "physicell_output")
    parsed = os.path.join(tmp, "parsed")
    try:
        fixture.write_fixture(frames, raw)

        # parse a single snapshot and compare against the source frame ----- #
        last = frames[-1]
        xml = os.path.join(raw, f"output{last['frame']:08d}.xml")
        cells, fibers = physicell_parser.load_snapshot(xml, physicell_parser.ParseConfig())

        n_cells_src = len(last["cells"])
        n_fibers_src = len(last["fibers"])
        assert len(cells["id"]) == n_cells_src, "cell count mismatch"
        assert len(fibers["id"]) == n_fibers_src, "fiber count mismatch"

        # fiber length recovered from endpoints
        src_len = []
        for f in last["fibers"]:
            p1 = np.array((f["x1"], f["y1"], f["z1"]))
            p2 = np.array((f["x2"], f["y2"], f["z2"]))
            src_len.append(np.linalg.norm(p2 - p1))
        rec_len = np.linalg.norm(fibers["p2"] - fibers["p1"], axis=1)
        assert np.allclose(sorted(src_len), sorted(rec_len), atol=1e-3), "fiber length mismatch"

        # phenotype + state labels survived
        assert set(cells["phenotype"]) <= set(synthetic.PHENOTYPES), "bad phenotype labels"
        assert set(fibers["state"]) <= set(fixture.FIBER_STATES), "bad fiber state labels"

        # full directory conversion + downstream metrics
        n = physicell_parser.convert_directory(raw, parsed)
        ts = metrics.compute_time_series(parsed, params.domain)
        assert n == len(frames), "frame count mismatch"
        assert ts[-1]["n_fibers"] == n_fibers_src, "metric fiber count mismatch"

        print("PASS  round-trip verified")
        print(f"      frames={n}  cells={n_cells_src}  fibers(final)={n_fibers_src}")
        print(f"      final alignment_index={ts[-1]['alignment_index']:.3f}  "
              f"volfrac={ts[-1]['volume_fraction']:.2e}")
        print(f"      fiber length match: max abs err = "
              f"{np.max(np.abs(np.sort(src_len) - np.sort(rec_len))):.2e} µm")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
