"""Parser for real PhysiCell + PhysiMeSS output → the pipeline's CSV schema.

PhysiCell writes one MultiCellDS snapshot per frame: an `outputNNNNNNNN.xml`
that names a MATLAB `_cells.mat` file plus a `<labels>` block mapping variable
names to row ranges in that matrix. PhysiMeSS stores ECM fibers as ordinary
PhysiCell agents in the SAME `.mat`, distinguished by `cell_type`, carrying a
fiber length in custom data and a direction in the standard `orientation`
field. A fiber's two endpoints are therefore  position ± 0.5·length·orientation.

This module turns a directory of those snapshots into:

  cells_t{frame:03d}.csv : id, x, y, z, radius, phenotype, vx, vy, vz
  fibers_t{frame:03d}.csv: id, x1, y1, z1, x2, y2, z2, radius, state, birth
  tracks.csv             : cell_id, frame, x, y, z   (cell IDs followed over time)
  events.csv             : frame, step, births, deaths (fiber ID set changes)

so every downstream module (metrics, figures, export) is unchanged.

Field names that vary between PhysiMeSS configurations are kept in `ParseConfig`
— adjust them there, not in the code, if your custom-data labels differ.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from scipy.io import loadmat


@dataclass
class ParseConfig:
    # cell_type id that PhysiMeSS uses for ECM fibers (everything else = cell)
    fiber_type_id: int = 1
    # custom-data label holding fiber length (PhysiMeSS: often "mLength")
    length_field: str = "mLength"
    length_fallbacks: tuple = ("length", "fiber_length", "mLength")
    # custom-data label holding a per-cell phenotype code (optional)
    phenotype_field: str = "phenotype"
    # custom-data label holding fiber state code (optional)
    fiber_state_field: str = "fiber_state"
    # default fiber radius (um) when total_volume is not meaningful for fibers
    default_fiber_radius: float = 0.10

    phenotype_names: tuple = ("producing", "contractile", "degrading", "quiescent")
    fiber_state_names: tuple = ("nascent", "mature", "aligned", "degraded")


# --------------------------------------------------------------------------- #
# MultiCellDS reading
# --------------------------------------------------------------------------- #
def _find_simplified_data(root: ET.Element):
    """Return the <simplified_data> element holding cell labels + .mat name."""
    for sd in root.iter("simplified_data"):
        if sd.find("labels") is not None and sd.find("filename") is not None:
            return sd
    raise ValueError("no cell simplified_data block found in MultiCellDS xml")


def parse_labels(sd: ET.Element) -> dict:
    """name -> (start_row, size) using MultiCellDS cumulative indices."""
    labels = {}
    for lab in sd.find("labels").findall("label"):
        name = (lab.text or "").strip()
        idx = int(lab.attrib.get("index", "0"))
        size = int(lab.attrib.get("size", "1"))
        labels[name] = (idx, size)
    return labels


def _load_matrix(mat_path: str) -> np.ndarray:
    data = loadmat(mat_path)
    arrays = [(k, v) for k, v in data.items()
              if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2]
    if not arrays:
        raise ValueError(f"no 2D matrix in {mat_path}")
    # prefer a variable literally named 'cells', else the largest matrix
    arrays.sort(key=lambda kv: (kv[0] != "cells", -kv[1].size))
    return np.asarray(arrays[0][1], dtype=float)


def _orient_matrix(mat: np.ndarray, n_features: int) -> np.ndarray:
    """PhysiCell stores features as rows; transpose if it arrived columns-first."""
    if mat.shape[0] == n_features:
        return mat
    if mat.shape[1] == n_features:
        return mat.T
    # be lenient: assume the smaller axis is features
    return mat if mat.shape[0] <= mat.shape[1] else mat.T


def _col(mat, labels, name, default=None):
    if name not in labels:
        return default
    start, size = labels[name]
    return mat[start:start + size, :]


# --------------------------------------------------------------------------- #
# Snapshot → internal cell/fiber dicts
# --------------------------------------------------------------------------- #
def load_snapshot(xml_path: str, cfg: ParseConfig):
    root = ET.parse(xml_path).getroot()
    sd = _find_simplified_data(root)
    labels = parse_labels(sd)
    mat_name = sd.find("filename").text.strip()
    mat_path = os.path.join(os.path.dirname(xml_path), os.path.basename(mat_name))
    n_features = max(s + sz for s, sz in labels.values())
    mat = _orient_matrix(_load_matrix(mat_path), n_features)

    ids = _col(mat, labels, "ID")[0].astype(int)
    pos = _col(mat, labels, "position")                       # (3, N)
    cell_type = _col(mat, labels, "cell_type")[0].astype(int)
    vol = _col(mat, labels, "total_volume")
    orient = _col(mat, labels, "orientation",
                  default=np.zeros((3, mat.shape[1])))         # (3, N)

    length = None
    for nm in (cfg.length_field, *cfg.length_fallbacks):
        length = _col(mat, labels, nm)
        if length is not None:
            length = length[0]
            break
    if length is None:
        length = np.zeros(mat.shape[1])

    pheno_code = _col(mat, labels, cfg.phenotype_field)
    pheno_code = pheno_code[0].astype(int) if pheno_code is not None else None
    fstate_code = _col(mat, labels, cfg.fiber_state_field)
    fstate_code = fstate_code[0].astype(int) if fstate_code is not None else None

    is_fiber = cell_type == cfg.fiber_type_id
    is_cell = ~is_fiber

    # cells -------------------------------------------------------------- #
    cpos = pos[:, is_cell].T
    cvol = vol[0, is_cell] if vol is not None else np.full(is_cell.sum(), 0.0)
    cradius = np.cbrt(np.maximum(cvol, 1e-9) * 3.0 / (4.0 * np.pi))
    if pheno_code is not None:
        names = cfg.phenotype_names
        cpheno = np.array([names[c] if 0 <= c < len(names) else "quiescent"
                           for c in pheno_code[is_cell]])
    else:
        cpheno = np.full(is_cell.sum(), "quiescent")
    cvel = orient[:, is_cell].T  # PhysiCell velocity is not in standard output;
                                 # orientation is the best available heading proxy
    cells = dict(id=ids[is_cell], pos=cpos, radius=cradius,
                 phenotype=cpheno, vel=cvel)

    # fibers ------------------------------------------------------------- #
    fcenter = pos[:, is_fiber].T
    fdir = orient[:, is_fiber].T
    norms = np.linalg.norm(fdir, axis=1, keepdims=True)
    fdir = np.divide(fdir, norms, out=np.tile([1.0, 0, 0], (len(fdir), 1)),
                     where=norms > 1e-9)
    flen = length[is_fiber]
    half = 0.5 * flen[:, None] * fdir
    if fstate_code is not None:
        snames = cfg.fiber_state_names
        fstate = np.array([snames[c] if 0 <= c < len(snames) else "mature"
                           for c in fstate_code[is_fiber]])
    else:
        fstate = np.full(is_fiber.sum(), "mature")
    fibers = dict(id=ids[is_fiber], p1=fcenter - half, p2=fcenter + half,
                  center=fcenter, direction=fdir, length=flen,
                  radius=np.full(is_fiber.sum(), cfg.default_fiber_radius),
                  state=fstate)
    return cells, fibers


# --------------------------------------------------------------------------- #
# Directory → CSV schema
# --------------------------------------------------------------------------- #
def _write_cells_csv(cells, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x", "y", "z", "radius", "phenotype", "vx", "vy", "vz"])
        for i in range(len(cells["id"])):
            p, v = cells["pos"][i], cells["vel"][i]
            w.writerow([cells["id"][i], p[0], p[1], p[2], cells["radius"][i],
                        cells["phenotype"][i], v[0], v[1], v[2]])


def _write_fibers_csv(fibers, path, birth_frame):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x1", "y1", "z1", "x2", "y2", "z2", "radius", "state", "birth"])
        for i in range(len(fibers["id"])):
            a, b = fibers["p1"][i], fibers["p2"][i]
            w.writerow([fibers["id"][i], a[0], a[1], a[2], b[0], b[1], b[2],
                        fibers["radius"][i], fibers["state"][i],
                        birth_frame.get(int(fibers["id"][i]), 0)])


def convert_directory(physicell_dir: str, out_dir: str,
                      cfg: ParseConfig | None = None,
                      pattern: str = "output*.xml") -> int:
    """Convert every MultiCellDS snapshot in `physicell_dir` to the CSV schema.

    Returns the number of frames converted.
    """
    import csv
    cfg = cfg or ParseConfig()
    os.makedirs(out_dir, exist_ok=True)
    xmls = sorted(glob.glob(os.path.join(physicell_dir, pattern)))
    xmls = [x for x in xmls if "initial" not in os.path.basename(x)]
    if not xmls:
        raise FileNotFoundError(f"no {pattern} snapshots in {physicell_dir}")

    tracks = []
    events = []
    birth_frame: dict[int, int] = {}
    prev_fiber_ids: set[int] = set()

    for frame, xml_path in enumerate(xmls):
        cells, fibers = load_snapshot(xml_path, cfg)

        fids = set(int(i) for i in fibers["id"])
        for fid in fids - prev_fiber_ids:
            birth_frame.setdefault(fid, frame)
        births = len(fids - prev_fiber_ids)
        deaths = len(prev_fiber_ids - fids)
        prev_fiber_ids = fids
        events.append((frame, frame, births, deaths))

        _write_cells_csv(cells, os.path.join(out_dir, f"cells_t{frame:03d}.csv"))
        _write_fibers_csv(fibers, os.path.join(out_dir, f"fibers_t{frame:03d}.csv"),
                          birth_frame)
        for i in range(len(cells["id"])):
            p = cells["pos"][i]
            tracks.append((int(cells["id"][i]), frame, p[0], p[1], p[2]))

    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "frame", "x", "y", "z"])
        w.writerows(tracks)
    with open(os.path.join(out_dir, "events.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "step", "births", "deaths"])
        w.writerows(events)

    return len(xmls)
