"""Growing ECM as a connected fiber NETWORK (not scattered rigid rods).

The earlier `synthetic.py` represents ECM the way PhysiMeSS does for fast
cell-fiber mechanics: each fiber is an independent straight segment. Real
collagen ECM is not like that — it is a *percolating network* of curved,
branching, crosslinked filaments. This module models that growth process so the
emergent architecture looks like a web / mesh, not a buzz-cut.

Mechanism (mirrors collagen self-assembly):

  nucleation   producing cells seed growing tips at their surface
  elongation   each tip extends one short segment per step, with directional
               persistence -> filaments are curved polylines, not straight rods
  guidance     tips steer toward the local mean fiber orientation (contact
               guidance) -> bundling and aligned tracks
  branching    a tip occasionally splits, creating Y junctions
  crosslinking a tip that grows near an existing node SNAPS to it, closing a
               loop -> this is what turns loose filaments into a connected mesh
  traction     contractile cells pull/bend nearby nodes toward themselves,
               compacting and aligning the matrix locally
  degradation  degrading cells / MMP delete nearby edges

Internal state is a graph: `nodes` (3D points) + `edges` (pairs of node indices).
On export each edge becomes one row of the existing `fibers_t*.csv` schema, so
every downstream metric / figure / VTK exporter works unchanged — but now the
segments share endpoints, so the network-connectivity metric actually rises and
ParaView/Blender tubes render as continuous curved fibers. A richer per-frame
graph JSON (nodes + polyline filaments) is also written for cinematic rendering.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

PHENOTYPES = ("producing", "contractile", "degrading", "quiescent")
EDGE_STATES = ("nascent", "mature", "aligned", "degraded", "crosslink")


@dataclass
class NetworkParams:
    domain: float = 200.0
    n_cells: int = 30
    cell_radius: float = 9.0
    steps: int = 90
    n_frames: int = 16
    seg_length: float = 3.2          # length of one growth segment (um)
    fiber_radius: float = 0.12
    persistence: float = 0.82        # 0=random walk, 1=straight  -> curvature
    guidance: float = 0.45           # steer toward local fiber orientation
    branch_prob: float = 0.045       # per tip per step
    crosslink_dist: float = 4.0      # snap distance -> forms loops / mesh (um)
    secretion_prob: float = 0.5      # producing cell spawns a tip / step
    seeds_per_event: int = 1
    max_tips: int = 320
    max_nodes: int = 9000
    tip_lifetime: int = 26           # steps before a tip stops growing
    align_radius: float = 22.0
    traction_pull: float = 0.06      # contractile compaction strength
    degrade_radius: float = 20.0
    degrade_prob: float = 0.05
    speed: float = 1.8               # cell motility (um/step)
    cell_guidance: float = 0.6       # contact-guided migration strength
    seed: int = 11


@dataclass
class _Tip:
    node: int                 # index of the growing end node
    direction: np.ndarray     # unit heading
    filament: int             # filament id (for polyline grouping)
    age: int = 0
    alive: bool = True


@dataclass
class _Cell:
    cid: int
    pos: np.ndarray
    phenotype: str
    traction_axis: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _rng_dir(rng):
    return _unit(rng.normal(size=3))


class _Grid:
    """Spatial hash of node indices for fast proximity / crosslink queries."""

    def __init__(self, cell):
        self.cell = cell
        self.buckets: dict[tuple, list[int]] = {}

    def _key(self, p):
        return (int(np.floor(p[0] / self.cell)),
                int(np.floor(p[1] / self.cell)),
                int(np.floor(p[2] / self.cell)))

    def add(self, idx, p):
        self.buckets.setdefault(self._key(p), []).append(idx)

    def near(self, p):
        k = self._key(p)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out += self.buckets.get((k[0] + dx, k[1] + dy, k[2] + dz), [])
        return out


def simulate_network(params: NetworkParams | None = None):
    """Grow the ECM network. Returns (frames, tracks, params).

    frames[i] = dict(frame, step, cells, nodes, edges, filaments, events)
      cells: list of cell row dicts (same schema as synthetic)
      nodes: (M,3) array
      edges: list of (a, b, state, birth)  node-index pairs
      filaments: list of node-index lists (ordered polylines, for rendering)
    """
    p = params or NetworkParams()
    rng = np.random.default_rng(p.seed)

    cells = []
    for i in range(p.n_cells):
        cells.append(_Cell(
            cid=i,
            pos=rng.uniform(0.2, 0.8, size=3) * p.domain,
            phenotype=PHENOTYPES[i % len(PHENOTYPES)],
            traction_axis=_rng_dir(rng),
        ))

    nodes: list[np.ndarray] = []
    edges: list[list] = []            # [a, b, state, birth]
    filaments: list[list[int]] = []   # ordered node indices per filament
    tips: list[_Tip] = []
    grid = _Grid(p.crosslink_dist)

    def add_node(pos):
        idx = len(nodes)
        nodes.append(np.asarray(pos, dtype=float))
        grid.add(idx, nodes[idx])
        return idx

    def seed_filament(pos, direction):
        if len(tips) >= p.max_tips or len(nodes) >= p.max_nodes:
            return
        n0 = add_node(pos)
        fil = len(filaments)
        filaments.append([n0])
        tips.append(_Tip(node=n0, direction=_unit(direction), filament=fil))

    # sparse background network so cells start in a real (if thin) matrix
    for _ in range(14):
        c = rng.uniform(0.15, 0.85, size=3) * p.domain
        seed_filament(c, _rng_dir(rng))

    def local_orientation(pos, radius):
        idxs = grid.near(pos)
        dirs = []
        for a, b, st, _ in edges:
            pass  # edges scan is below; we approximate via node neighborhood
        # approximate local orientation from edges touching nearby nodes
        near = [i for i in idxs if np.sum((nodes[i] - pos) ** 2) < radius * radius]
        if len(near) < 2:
            return None
        nearset = set(near)
        for a, b, st, _ in edges:
            if a in nearset or b in nearset:
                d = nodes[b] - nodes[a]
                if np.dot(d, d) > 1e-9:
                    dirs.append(_unit(d))
        if len(dirs) < 2:
            return None
        D = np.array(dirs)
        tensor = D.T @ D / len(D)
        w, V = np.linalg.eigh(tensor)
        return _unit(V[:, -1])

    save_steps = set(np.linspace(0, p.steps, p.n_frames, dtype=int).tolist())
    frames = []
    tracks = []

    def snapshot(step, births, deaths):
        fi = len(frames)
        cell_rows = [dict(id=c.cid, x=c.pos[0], y=c.pos[1], z=c.pos[2],
                          radius=p.cell_radius, phenotype=c.phenotype,
                          vx=c.vel[0], vy=c.vel[1], vz=c.vel[2]) for c in cells]
        frames.append(dict(
            frame=fi, step=step,
            cells=cell_rows,
            nodes=np.array(nodes) if nodes else np.zeros((0, 3)),
            edges=[list(e) for e in edges],
            filaments=[list(f) for f in filaments],
            events=dict(births=births, deaths=deaths),
        ))
        for c in cells:
            tracks.append(dict(cell_id=c.cid, frame=fi,
                               x=c.pos[0], y=c.pos[1], z=c.pos[2]))

    if 0 in save_steps:
        snapshot(0, births=len(edges), deaths=0)

    for step in range(1, p.steps + 1):
        births = 0
        deaths = 0

        # --- nucleation: producing cells seed new tips at their surface ---
        for c in cells:
            if c.phenotype == "producing" and rng.random() < p.secretion_prob:
                for _ in range(p.seeds_per_event):
                    off = _rng_dir(rng) * (p.cell_radius + rng.uniform(0.5, 3))
                    direction = local_orientation(c.pos, p.align_radius)
                    if direction is None or rng.random() < 0.4:
                        direction = _rng_dir(rng)
                    seed_filament(c.pos + off, direction)

        # --- elongation + branching + crosslinking -----------------------
        new_tips = []
        for tip in tips:
            if not tip.alive:
                continue
            tip.age += 1
            if tip.age > p.tip_lifetime or len(nodes) >= p.max_nodes:
                tip.alive = False
                continue

            base = nodes[tip.node]
            local = local_orientation(base, p.align_radius)
            rand = _rng_dir(rng)
            heading = p.persistence * tip.direction + (1 - p.persistence) * rand
            if local is not None:
                if np.dot(local, tip.direction) < 0:
                    local = -local
                heading = (1 - p.guidance) * heading + p.guidance * local
            heading = _unit(heading)
            new_pos = base + heading * p.seg_length

            # boundary: reflect direction, don't leave the box
            for ax in range(3):
                if new_pos[ax] < 2 or new_pos[ax] > p.domain - 2:
                    heading[ax] *= -1
            heading = _unit(heading)
            new_pos = np.clip(base + heading * p.seg_length, 2, p.domain - 2)

            # crosslink: snap to an existing nearby node (not our own filament tail)
            crosslinked = False
            own_tail = set(filaments[tip.filament][-3:])
            for j in grid.near(new_pos):
                if j == tip.node or j in own_tail:
                    continue
                if np.sum((nodes[j] - new_pos) ** 2) < p.crosslink_dist ** 2:
                    edges.append([tip.node, j, "crosslink", step])
                    births += 1
                    tip.alive = False   # filament terminates into the network
                    crosslinked = True
                    break
            if crosslinked:
                continue

            nj = add_node(new_pos)
            edges.append([tip.node, nj, "nascent", step])
            births += 1
            filaments[tip.filament].append(nj)
            tip.node = nj
            tip.direction = heading

            # branch -> Y junction (the thing that makes it a web)
            if rng.random() < p.branch_prob and len(tips) + len(new_tips) < p.max_tips:
                rot = _unit(heading + _rng_dir(rng) * 0.9)
                fil = len(filaments)
                filaments.append([nj])
                new_tips.append(_Tip(node=nj, direction=rot, filament=fil))

        tips = [t for t in tips if t.alive] + new_tips

        # --- maturation of nascent edges ---------------------------------
        for e in edges:
            if e[2] == "nascent" and step - e[3] > 5:
                e[2] = "mature"

        # --- contractile traction: pull nearby nodes toward the cell -----
        if nodes:
            node_arr = np.array(nodes)
            for c in cells:
                if c.phenotype != "contractile":
                    continue
                d = node_arr - c.pos
                dist2 = np.einsum("ij,ij->i", d, d)
                near = np.where(dist2 < p.align_radius ** 2)[0]
                for i in near:
                    pull = (c.pos - nodes[i]) * p.traction_pull
                    nodes[i] = nodes[i] + pull
                    node_arr[i] = nodes[i]
            # mark edges near contractile cells as aligned/bundled
            for e in edges:
                if e[2] in ("mature", "nascent"):
                    mid = 0.5 * (nodes[e[0]] + nodes[e[1]])
                    for c in cells:
                        if c.phenotype == "contractile" and \
                                np.sum((mid - c.pos) ** 2) < p.align_radius ** 2:
                            e[2] = "aligned"
                            break

        # --- degradation: degrading cells / MMP remove nearby edges ------
        if edges:
            kept = []
            for e in edges:
                mid = 0.5 * (nodes[e[0]] + nodes[e[1]])
                drop = False
                for c in cells:
                    if c.phenotype == "degrading" and \
                            np.sum((mid - c.pos) ** 2) < p.degrade_radius ** 2 and \
                            rng.random() < p.degrade_prob:
                        drop = True
                        break
                if drop:
                    deaths += 1
                else:
                    kept.append(e)
            edges = kept

        # --- contact-guided cell migration -------------------------------
        for c in cells:
            local = local_orientation(c.pos, p.align_radius)
            rand = _rng_dir(rng)
            if local is not None:
                if np.dot(local, c.vel) < 0:
                    local = -local
                heading = _unit((1 - p.cell_guidance) * rand + p.cell_guidance * local)
            else:
                heading = rand
            c.vel = heading * p.speed
            c.pos = np.clip(c.pos + c.vel, 2, p.domain - 2)

        if step in save_steps:
            snapshot(step, births, deaths)

    return frames, tracks, p


# --------------------------------------------------------------------------- #
# Export: edges -> existing fibers CSV schema; plus rich graph JSON
# --------------------------------------------------------------------------- #
def write_csv(frames, tracks, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    cell_fields = ["id", "x", "y", "z", "radius", "phenotype", "vx", "vy", "vz"]
    fiber_fields = ["id", "x1", "y1", "z1", "x2", "y2", "z2", "radius", "state", "birth"]
    radius = NetworkParams().fiber_radius

    for fr in frames:
        with open(os.path.join(outdir, f"cells_t{fr['frame']:03d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cell_fields)
            w.writeheader()
            w.writerows(fr["cells"])

        nodes = fr["nodes"]
        with open(os.path.join(outdir, f"fibers_t{fr['frame']:03d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fiber_fields)
            w.writeheader()
            for eid, (a, b, state, birth) in enumerate(fr["edges"]):
                pa, pb = nodes[a], nodes[b]
                w.writerow(dict(id=eid, x1=pa[0], y1=pa[1], z1=pa[2],
                                x2=pb[0], y2=pb[1], z2=pb[2],
                                radius=radius, state=state, birth=birth))

        # rich graph for rendering (polylines preserved)
        with open(os.path.join(outdir, f"graph_t{fr['frame']:03d}.json"), "w") as fh:
            json.dump(dict(
                nodes=nodes.tolist(),
                edges=[[int(a), int(b), st] for a, b, st, _ in fr["edges"]],
                filaments=[[int(i) for i in fil] for fil in fr["filaments"] if len(fil) > 1],
            ), fh)

    with open(os.path.join(outdir, "tracks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell_id", "frame", "x", "y", "z"])
        w.writeheader()
        w.writerows(tracks)
    with open(os.path.join(outdir, "events.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "step", "births", "deaths"])
        w.writeheader()
        for fr in frames:
            w.writerow(dict(frame=fr["frame"], step=fr["step"],
                            births=fr["events"]["births"], deaths=fr["events"]["deaths"]))
