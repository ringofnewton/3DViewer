"""Synthetic cell + ECM-fiber remodeling generator.

This is a lightweight agent model whose ONLY purpose is to produce realistic,
trend-bearing input for the analysis pipeline so it can run without a real
PhysiCell + PhysiMeSS simulation. It deliberately reproduces the four MVP
behaviors from the decision map:

  MVP1  matrix-producing cells nucleate short fibers near their surface
  MVP2  matrix-degrading cells / MMP remove nearby fibers
  MVP3  contractile cells rotate nearby fibers toward a traction axis (alignment)
  MVP4  cells migrate with a velocity biased along the local fiber orientation
        (contact guidance), and we record their tracks

Units are micrometers (um) and arbitrary "steps" of time.

Output schema (the contract the rest of the pipeline depends on):

  cells_t{frame:03d}.csv : id, x, y, z, radius, phenotype, vx, vy, vz
  fibers_t{frame:03d}.csv: id, x1, y1, z1, x2, y2, z2, radius, state, birth
  tracks.csv             : cell_id, frame, x, y, z
  events.csv             : frame, births, deaths
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

PHENOTYPES = ("producing", "contractile", "degrading", "quiescent")


@dataclass
class Params:
    domain: float = 200.0          # cube side length (um)
    n_cells: int = 36
    cell_radius: float = 9.0       # um
    steps: int = 60                # total simulated steps
    n_frames: int = 13             # snapshots written (incl. t0)
    fiber_length: float = 14.0     # um, nominal new-fiber length
    fiber_radius: float = 0.10     # um (collagen-like)
    secretion_prob: float = 0.55   # per producing cell per step
    degrade_radius: float = 22.0   # um, MMP action radius
    degrade_prob: float = 0.30     # per fiber in range per degrading cell
    align_radius: float = 26.0     # um, traction reach of contractile cells
    align_rate: float = 0.22       # fraction rotated toward traction axis / step
    guidance: float = 0.6          # 0..1 contact-guidance strength on motility
    speed: float = 2.2             # um/step base motility
    seed: int = 7


@dataclass
class _Fiber:
    fid: int
    center: np.ndarray
    direction: np.ndarray   # unit vector
    length: float
    radius: float
    state: str              # nascent | mature | aligned | degraded
    birth: int


@dataclass
class _Cell:
    cid: int
    pos: np.ndarray
    phenotype: str
    traction_axis: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _rng_dir(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return _unit(v)


def _local_orientation(center, fibers_dir, fibers_pos, radius):
    """Mean orientation tensor of fibers near `center`; returns dominant axis."""
    if len(fibers_pos) == 0:
        return None
    d = fibers_pos - center
    near = np.einsum("ij,ij->i", d, d) < radius * radius
    if not np.any(near):
        return None
    dirs = fibers_dir[near]
    # orientation is axial (d and -d equivalent): use the structure tensor
    tensor = dirs.T @ dirs / len(dirs)
    w, V = np.linalg.eigh(tensor)
    return _unit(V[:, -1])


def simulate(params: Params | None = None):
    """Run the synthetic model. Returns (frames, params).

    `frames` is a list of dicts: {frame, step, cells, fibers, events}.
    """
    p = params or Params()
    rng = np.random.default_rng(p.seed)

    cells: list[_Cell] = []
    for i in range(p.n_cells):
        pheno = PHENOTYPES[i % len(PHENOTYPES)]
        cells.append(
            _Cell(
                cid=i,
                pos=rng.uniform(0.15, 0.85, size=3) * p.domain,
                phenotype=pheno,
                traction_axis=_rng_dir(rng),
            )
        )

    fibers: list[_Fiber] = []
    next_fid = 0
    # seed a sparse isotropic background matrix
    for _ in range(120):
        c = rng.uniform(0.05, 0.95, size=3) * p.domain
        fibers.append(_Fiber(next_fid, c, _rng_dir(rng), p.fiber_length,
                             p.fiber_radius, "mature", 0))
        next_fid += 1

    save_steps = set(np.linspace(0, p.steps, p.n_frames, dtype=int).tolist())
    frames = []
    tracks = []

    def snapshot(step, births, deaths):
        frame_idx = len(frames)
        cell_rows = [
            dict(id=c.cid, x=c.pos[0], y=c.pos[1], z=c.pos[2],
                 radius=p.cell_radius, phenotype=c.phenotype,
                 vx=c.vel[0], vy=c.vel[1], vz=c.vel[2])
            for c in cells
        ]
        fiber_rows = []
        for f in fibers:
            half = 0.5 * f.length * f.direction
            fiber_rows.append(dict(
                id=f.fid,
                x1=f.center[0] - half[0], y1=f.center[1] - half[1], z1=f.center[2] - half[2],
                x2=f.center[0] + half[0], y2=f.center[1] + half[1], z2=f.center[2] + half[2],
                radius=f.radius, state=f.state, birth=f.birth,
            ))
        frames.append(dict(frame=frame_idx, step=step, cells=cell_rows,
                           fibers=fiber_rows, events=dict(births=births, deaths=deaths)))
        for c in cells:
            tracks.append(dict(cell_id=c.cid, frame=frame_idx,
                               x=c.pos[0], y=c.pos[1], z=c.pos[2]))

    if 0 in save_steps:
        snapshot(0, births=len(fibers), deaths=0)

    for step in range(1, p.steps + 1):
        births = 0
        deaths = 0

        fib_pos = np.array([f.center for f in fibers]) if fibers else np.zeros((0, 3))
        fib_dir = np.array([f.direction for f in fibers]) if fibers else np.zeros((0, 3))

        # --- per-cell behaviors ------------------------------------------
        for c in cells:
            # MVP1: secretion. New fibers tend to align with local matrix.
            if c.phenotype == "producing" and rng.random() < p.secretion_prob:
                local = _local_orientation(c.pos, fib_dir, fib_pos, p.align_radius)
                if local is not None and rng.random() < 0.6:
                    jitter = _rng_dir(rng) * 0.35
                    direction = _unit(local + jitter)
                else:
                    direction = _rng_dir(rng)
                offset = _rng_dir(rng) * (p.cell_radius + rng.uniform(1, 6))
                fibers.append(_Fiber(next_fid, c.pos + offset, direction,
                                     p.fiber_length * rng.uniform(0.6, 1.1),
                                     p.fiber_radius, "nascent", step))
                next_fid += 1
                births += 1

            # MVP3: contractile cells align nearby fibers toward traction axis.
            if c.phenotype == "contractile" and len(fib_pos):
                d = fib_pos - c.pos
                near = np.where(np.einsum("ij,ij->i", d, d) < p.align_radius ** 2)[0]
                axis = c.traction_axis
                for idx in near:
                    f = fibers[idx]
                    # rotate direction a fraction toward the (axial) traction axis
                    target = axis if np.dot(f.direction, axis) >= 0 else -axis
                    f.direction = _unit((1 - p.align_rate) * f.direction + p.align_rate * target)
                    if f.state in ("nascent", "mature"):
                        f.state = "aligned"

            # MVP2: degrading cells / MMP remove nearby fibers.
            if c.phenotype == "degrading" and len(fib_pos):
                d = fib_pos - c.pos
                near = np.where(np.einsum("ij,ij->i", d, d) < p.degrade_radius ** 2)[0]
                for idx in near:
                    if rng.random() < p.degrade_prob:
                        fibers[idx].state = "degraded"

        # apply degradation removals
        before = len(fibers)
        fibers = [f for f in fibers if f.state != "degraded"]
        deaths += before - len(fibers)

        # maturation of nascent fibers
        for f in fibers:
            if f.state == "nascent" and step - f.birth > 4:
                f.state = "mature"

        # --- MVP4: contact-guided migration ------------------------------
        fib_pos = np.array([f.center for f in fibers]) if fibers else np.zeros((0, 3))
        fib_dir = np.array([f.direction for f in fibers]) if fibers else np.zeros((0, 3))
        for c in cells:
            rand_dir = _rng_dir(rng)
            local = _local_orientation(c.pos, fib_dir, fib_pos, p.align_radius)
            if local is not None:
                # bias along the fiber axis, sign chosen to keep momentum
                if np.dot(local, c.vel) < 0:
                    local = -local
                heading = _unit((1 - p.guidance) * rand_dir + p.guidance * local)
            else:
                heading = rand_dir
            c.vel = heading * p.speed
            c.pos = np.clip(c.pos + c.vel, 2.0, p.domain - 2.0)

        if step in save_steps:
            snapshot(step, births, deaths)

    return frames, tracks, p


def write_csv(frames, tracks, outdir: str):
    """Persist all frames to the on-disk schema the pipeline consumes."""
    os.makedirs(outdir, exist_ok=True)
    cell_fields = ["id", "x", "y", "z", "radius", "phenotype", "vx", "vy", "vz"]
    fiber_fields = ["id", "x1", "y1", "z1", "x2", "y2", "z2", "radius", "state", "birth"]

    for fr in frames:
        with open(os.path.join(outdir, f"cells_t{fr['frame']:03d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cell_fields)
            w.writeheader()
            w.writerows(fr["cells"])
        with open(os.path.join(outdir, f"fibers_t{fr['frame']:03d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fiber_fields)
            w.writeheader()
            w.writerows(fr["fibers"])

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
