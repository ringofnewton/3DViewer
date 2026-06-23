"""Write synthetic frames as REAL PhysiCell + PhysiMeSS MultiCellDS output.

This exists so the parser in `physicell_parser.py` can be verified end-to-end
without a PhysiCell install: we emit `outputNNNNNNNN.xml` + `_cells.mat` files in
the exact MultiCellDS layout PhysiCell/PhysiMeSS use (cells and ECM fibers in a
single agent matrix, separated by `cell_type`, fibers carrying `orientation` and
an `mLength` custom field), then parse them back.

Feature layout written into the matrix (cumulative MultiCellDS indices):

  index 0  size 1  ID
  index 1  size 3  position
  index 4  size 1  total_volume
  index 5  size 1  cell_type        (0 = cell, 1 = ECM fiber)
  index 6  size 3  orientation      (cell heading / fiber axis)
  index 9  size 1  mLength          (fiber length; 0 for cells)
  index 10 size 1  phenotype        (cell phenotype code; 0 for fibers)
  index 11 size 1  fiber_state      (fiber state code; 0 for cells)
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import savemat

from .synthetic import PHENOTYPES

FIBER_ID_OFFSET = 1_000_000  # keep cell and fiber IDs disjoint in the matrix
FIBER_STATES = ("nascent", "mature", "aligned", "degraded")

_LABELS = [
    ("ID", 0, 1), ("position", 1, 3), ("total_volume", 4, 1),
    ("cell_type", 5, 1), ("orientation", 6, 3), ("mLength", 9, 1),
    ("phenotype", 10, 1), ("fiber_state", 11, 1),
]
_N_FEATURES = 12


def _build_matrix(frame: dict) -> np.ndarray:
    cells = frame["cells"]
    fibers = frame["fibers"]
    cols = []

    for c in cells:
        r = c["radius"]
        vol = 4.0 / 3.0 * np.pi * r ** 3
        pheno_code = PHENOTYPES.index(c["phenotype"])
        col = np.zeros(_N_FEATURES)
        col[0] = c["id"]
        col[1:4] = (c["x"], c["y"], c["z"])
        col[4] = vol
        col[5] = 0                      # cell_type 0 = cell
        col[6:9] = (c["vx"], c["vy"], c["vz"])
        col[9] = 0.0                    # mLength
        col[10] = pheno_code
        col[11] = 0
        cols.append(col)

    for f in fibers:
        p1 = np.array((f["x1"], f["y1"], f["z1"]))
        p2 = np.array((f["x2"], f["y2"], f["z2"]))
        seg = p2 - p1
        length = float(np.linalg.norm(seg))
        direction = seg / length if length > 1e-9 else np.array([1.0, 0, 0])
        center = 0.5 * (p1 + p2)
        state_code = FIBER_STATES.index(f["state"]) if f["state"] in FIBER_STATES else 1
        col = np.zeros(_N_FEATURES)
        col[0] = f["id"] + FIBER_ID_OFFSET
        col[1:4] = center
        col[4] = np.pi * f["radius"] ** 2 * length   # rod volume
        col[5] = 1                      # cell_type 1 = ECM fiber
        col[6:9] = direction
        col[9] = length
        col[10] = 0
        col[11] = state_code
        cols.append(col)

    return np.array(cols).T if cols else np.zeros((_N_FEATURES, 0))


def _labels_xml() -> str:
    return "\n".join(
        f'            <label index="{i}" size="{s}">{name}</label>'
        for name, i, s in _LABELS
    )


def _snapshot_xml(mat_filename: str, time_min: float) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<MultiCellDS version="0.5" type="snapshot/simulation">
  <metadata>
    <current_time units="min">{time_min}</current_time>
  </metadata>
  <cellular_information>
    <cell_populations>
      <cell_population type="individual">
        <custom>
          <simplified_data type="matlab" source="PhysiCell">
            <labels>
{_labels_xml()}
            </labels>
            <filename>{mat_filename}</filename>
          </simplified_data>
        </custom>
      </cell_population>
    </cell_populations>
  </cellular_information>
</MultiCellDS>
"""


def write_fixture(frames, outdir: str, minutes_per_frame: float = 30.0):
    """Write all frames as MultiCellDS xml + _cells.mat into `outdir`."""
    os.makedirs(outdir, exist_ok=True)
    for fr in frames:
        idx = fr["frame"]
        mat_name = f"output{idx:08d}_cells.mat"
        savemat(os.path.join(outdir, mat_name), {"cells": _build_matrix(fr)})
        with open(os.path.join(outdir, f"output{idx:08d}.xml"), "w") as fh:
            fh.write(_snapshot_xml(mat_name, idx * minutes_per_frame))
    return len(frames)
