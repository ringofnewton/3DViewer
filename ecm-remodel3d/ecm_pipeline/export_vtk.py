"""Minimal, dependency-free VTK XML writers for ParaView.

We hand-write the XML so the pipeline needs no `vtk` install. Output:

  fibers_t{frame}.vtp : PolyData lines (one line cell per fiber) with
                        per-fiber scalars `radius` and `state_code`.
  cells_t{frame}.vtp  : PolyData verts (points) with `radius` and `pheno_code`.

In ParaView, apply a Tube filter to the fibers (radius from `radius`) and a
Glyph (sphere, scaled by `radius`) to the cells.
"""

from __future__ import annotations

import os

import numpy as np

_STATE_CODE = {"nascent": 0, "mature": 1, "aligned": 2, "degraded": 3}
_PHENO_CODE = {"producing": 0, "contractile": 1, "degrading": 2, "quiescent": 3}


def _data_array(name, values, fmt="%.5g", n_comp=1):
    body = " ".join(fmt % v for v in np.ravel(values))
    return (f'<DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="{n_comp}" format="ascii">{body}</DataArray>')


def write_fibers_vtp(fibers: dict, path: str):
    p1, p2 = fibers["p1"], fibers["p2"]
    n = len(p1)
    pts = np.empty((2 * n, 3))
    pts[0::2] = p1
    pts[1::2] = p2
    connectivity = " ".join(str(i) for i in range(2 * n))
    offsets = " ".join(str(2 * (i + 1)) for i in range(n))
    state_code = np.array([_STATE_CODE.get(s, 1) for s in fibers["state"]], dtype=float)

    coords = " ".join("%.4f" % v for v in pts.ravel())
    xml = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">
 <PolyData>
  <Piece NumberOfPoints="{2*n}" NumberOfVerts="0" NumberOfLines="{n}" NumberOfStrips="0" NumberOfPolys="0">
   <CellData Scalars="state_code">
    {_data_array("radius", fibers["radius"])}
    {_data_array("state_code", state_code)}
   </CellData>
   <Points>
    <DataArray type="Float32" NumberOfComponents="3" format="ascii">{coords}</DataArray>
   </Points>
   <Lines>
    <DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>
    <DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>
   </Lines>
  </Piece>
 </PolyData>
</VTKFile>
"""
    with open(path, "w") as fh:
        fh.write(xml)


def write_cells_vtp(cells: dict, path: str):
    pos = cells["pos"]
    n = len(pos)
    connectivity = " ".join(str(i) for i in range(n))
    offsets = " ".join(str(i + 1) for i in range(n))
    pheno_code = np.array([_PHENO_CODE.get(s, 3) for s in cells["phenotype"]], dtype=float)
    coords = " ".join("%.4f" % v for v in pos.ravel())
    xml = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">
 <PolyData>
  <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
   <PointData Scalars="pheno_code">
    {_data_array("radius", cells["radius"])}
    {_data_array("pheno_code", pheno_code)}
   </PointData>
   <Points>
    <DataArray type="Float32" NumberOfComponents="3" format="ascii">{coords}</DataArray>
   </Points>
   <Verts>
    <DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>
    <DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>
   </Verts>
  </Piece>
 </PolyData>
</VTKFile>
"""
    with open(path, "w") as fh:
        fh.write(xml)


def write_filaments_vtp(nodes: np.ndarray, filaments: list, radius: float, path: str):
    """Write filaments as continuous polylines (one PolyLine cell each).

    ParaView: apply a Tube filter -> continuous curved fibers, not loose sticks.
    """
    polylines = [f for f in filaments if len(f) > 1]
    coords = " ".join("%.4f" % v for v in np.asarray(nodes).ravel())
    conn, offs, run = [], [], 0
    for fil in polylines:
        conn += [str(i) for i in fil]
        run += len(fil)
        offs.append(str(run))
    n_pts = len(nodes)
    xml = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">
 <PolyData>
  <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="{len(polylines)}" NumberOfStrips="0" NumberOfPolys="0">
   <Points>
    <DataArray type="Float32" NumberOfComponents="3" format="ascii">{coords}</DataArray>
   </Points>
   <Lines>
    <DataArray type="Int64" Name="connectivity" format="ascii">{" ".join(conn)}</DataArray>
    <DataArray type="Int64" Name="offsets" format="ascii">{" ".join(offs)}</DataArray>
   </Lines>
  </Piece>
 </PolyData>
</VTKFile>
"""
    with open(path, "w") as fh:
        fh.write(xml)


def export_graph_scene_json(nodes, filaments, cells, path: str):
    """Per-frame network scene for the web / cinematic renderer (polylines)."""
    import json
    nodes = np.asarray(nodes)
    data = dict(
        nodes=nodes.tolist(),
        filaments=[[int(i) for i in f] for f in filaments if len(f) > 1],
        cells=[dict(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                    r=float(r), pheno=str(s))
               for p, r, s in zip(cells["pos"], cells["radius"], cells["phenotype"])],
    )
    with open(path, "w") as fh:
        json.dump(data, fh)


def export_scene_json(cells: dict, fibers: dict, path: str):
    """Compact JSON for the web (three.js / vtk.js) viewer."""
    import json
    data = dict(
        cells=[dict(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                    r=float(r), pheno=str(s))
               for p, r, s in zip(cells["pos"], cells["radius"], cells["phenotype"])],
        fibers=[dict(p1=p1.tolist(), p2=p2.tolist(), r=float(r), state=str(s))
                for p1, p2, r, s in zip(fibers["p1"], fibers["p2"],
                                        fibers["radius"], fibers["state"])],
    )
    with open(path, "w") as fh:
        json.dump(data, fh)
