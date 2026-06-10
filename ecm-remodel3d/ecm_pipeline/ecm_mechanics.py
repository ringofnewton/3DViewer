"""Mechanical fiber-network model of ECM under cell traction.

This is the physics layer that makes the simulation *predictive* for
morphogenesis: cells do not paint collagen, they pull on it, and the matrix
deforms to mechanical equilibrium. The emergent pattern of aligned, tensed
collagen between cells is what guides tissue shape.

Constitutive model (chosen to match measured collagen behavior):

  stretch     each segment is an axial spring about a rest length L0.
  buckling    collagen carries tension but BUCKLES under compression, so the
              axial stiffness is strongly asymmetric: k_tension >> k_compression.
              This nonlinearity is what focuses force into long-range, aligned
              tracks between cells (a linear material diffuses force and forms
              no tracks) — see Notbohm et al., Wang et al. on nonlinear ECM.
  stiffening  under tension the stiffness grows with strain (strain-stiffening),
              the other hallmark of collagen/fibrin networks.
  bending     semiflexible filaments resist bending along their backbone
              (discrete worm-like-chain / Laplacian curvature penalty); free
              hinges at crosslink junctions.
  anchoring   nodes near the domain boundary are fixed: the matrix is tethered
              to its surroundings, giving cell traction something to pull against.

Solver: overdamped (quasi-static) relaxation. ECM remodeling is slow, so inertia
is negligible; we descend the elastic energy under the external traction field
until the network reaches force balance.

The network is plain arrays, so it accepts ANY source — a grown network
(`ecm_network`), a random Mikado-style network (`build_random_network`), or a
parsed PhysiMeSS network — and every metric/exporter downstream still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MechParams:
    k_stretch: float = 1.0          # axial stiffness in tension (ref)
    buckle_ratio: float = 0.02      # compression stiffness / tension stiffness
    buckle_width: float = 0.03      # smoothing width of the tension/compression switch
    stiffen_alpha: float = 6.0      # strain-stiffening: k_eff = k*(1+alpha*strain+)
    k_bend: float = 0.12            # filament bending stiffness (stabilizes floppy modes)
    traction: float = 1.5           # contractile force magnitude per node (ref)
    reach: float = 28.0             # traction reach radius around a cell (um)
    reach_decay: float = 14.0       # exp decay length of traction with distance
    cell_radius: float = 9.0        # traction vanishes inside the cell body (um)
    anchor_margin: float = 8.0      # nodes within this of a wall are fixed (um)
    # FIRE relaxation (robust force-only minimization of the stiff network)
    fire_dt: float = 0.15
    fire_dtmax: float = 1.0
    fire_a0: float = 0.1
    fire_nmin: int = 5
    max_iter: int = 4000
    ftol: float = 1.5e-2            # converged when max nodal force < ftol
    max_step: float = 0.6           # clamp per-iteration displacement (um)


@dataclass
class FiberNetwork:
    nodes: np.ndarray                       # (N,3)
    edges: np.ndarray                       # (E,2) int
    rest_length: np.ndarray                 # (E,)
    bends: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), int))  # (B,3) prev,center,next
    fixed: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))      # (N,)
    state: np.ndarray = field(default_factory=lambda: np.zeros(0, object))    # (E,) labels
    k_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))          # (E,) plastic stiffness gain

    def __post_init__(self):
        if len(self.k_scale) != len(self.edges):
            self.k_scale = np.ones(len(self.edges))

    def copy(self):
        return FiberNetwork(self.nodes.copy(), self.edges.copy(),
                            self.rest_length.copy(), self.bends.copy(),
                            self.fixed.copy(), self.state.copy(),
                            self.k_scale.copy())


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _segment_filament(p0, direction, seg_len, n_seg):
    pts = [p0 + direction * seg_len * i for i in range(n_seg + 1)]
    return np.array(pts)


def build_random_network(domain=200.0, n_fibers=140, fiber_len=70.0,
                         seg_len=7.0, crosslink_dist=4.5, seed=0,
                         slab_thickness=None):
    """3D Mikado-style network: random straight fibers, crosslinked where close.

    A standard, well-characterized model network for collagen mechanics.
    Returns a FiberNetwork with backbone bends and boundary anchoring.

    `slab_thickness`: if set, fibers are confined to a thin z-slab centered in
    the domain with near-in-plane orientation — the quasi-2D geometry of the
    canonical explant/gel traction experiments (cleaner statistics).
    """
    rng = np.random.default_rng(seed)
    nodes = []
    edges = []
    bends = []
    states = []
    zc = domain / 2.0

    for _ in range(n_fibers):
        if slab_thickness is not None:
            c = np.array([rng.uniform(0.1, 0.9) * domain,
                          rng.uniform(0.1, 0.9) * domain,
                          zc + rng.uniform(-0.5, 0.5) * slab_thickness])
            d = np.array([rng.normal(), rng.normal(), rng.normal() * 0.12])
        else:
            c = rng.uniform(0.1, 0.9, size=3) * domain
            d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        n_seg = max(2, int(fiber_len / seg_len))
        start = c - 0.5 * fiber_len * d
        pts = _segment_filament(start, d, seg_len, n_seg)
        pts = np.clip(pts, 1.0, domain - 1.0)
        if slab_thickness is not None:
            pts[:, 2] = np.clip(pts[:, 2], zc - 0.5 * slab_thickness,
                                zc + 0.5 * slab_thickness)
        base = len(nodes)
        nodes.extend(pts.tolist())
        for i in range(len(pts) - 1):
            edges.append((base + i, base + i + 1))
            states.append("mature")
        for i in range(1, len(pts) - 1):
            bends.append((base + i - 1, base + i, base + i + 1))

    nodes = np.array(nodes)
    edges = list(edges)
    states = list(states)

    # crosslink: connect close nodes from different fibers with a stiff short bond
    cell = crosslink_dist
    buckets: dict[tuple, list[int]] = {}
    keys = np.floor(nodes / cell).astype(int)
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    import itertools
    added = set()
    for k, idxs in buckets.items():
        cand = []
        for dk in itertools.product((-1, 0, 1), repeat=3):
            cand += buckets.get((k[0] + dk[0], k[1] + dk[1], k[2] + dk[2]), [])
        for a in idxs:
            for b in cand:
                if b <= a:
                    continue
                pair = (a, b)
                if pair in added:
                    continue
                dd = nodes[a] - nodes[b]
                if dd @ dd < cell * cell:
                    edges.append((a, b))
                    states.append("crosslink")
                    added.add(pair)

    edges = np.array(edges)
    rest = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    rest = np.maximum(rest, 1e-3)
    bends = np.array(bends) if bends else np.zeros((0, 3), int)

    fixed = ((nodes < MechParams().anchor_margin) |
             (nodes > domain - MechParams().anchor_margin)).any(axis=1)

    return FiberNetwork(nodes=nodes, edges=edges, rest_length=rest, bends=bends,
                        fixed=fixed, state=np.array(states, dtype=object))


def from_growth_frame(frame, domain, anchor_margin=8.0):
    """Build a FiberNetwork from an ecm_network growth frame."""
    nodes = np.asarray(frame["nodes"], dtype=float)
    edges = np.array([[a, b] for a, b, *_ in frame["edges"]])
    states = np.array([st for *_, st, _ in [(e[0], e[1], e[2], e[3]) for e in frame["edges"]]],
                      dtype=object) if len(frame["edges"]) else np.zeros(0, object)
    rest = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1) if len(edges) else np.zeros(0)
    bends = []
    for fil in frame["filaments"]:
        for i in range(1, len(fil) - 1):
            bends.append((fil[i - 1], fil[i], fil[i + 1]))
    bends = np.array(bends) if bends else np.zeros((0, 3), int)
    fixed = ((nodes < anchor_margin) | (nodes > domain - anchor_margin)).any(axis=1)
    return FiberNetwork(nodes, edges, np.maximum(rest, 1e-3), bends, fixed, states)


# --------------------------------------------------------------------------- #
# Forces
# --------------------------------------------------------------------------- #
def _axial_stiffness(strain, p: MechParams):
    """Asymmetric, strain-stiffening axial stiffness (vectorized).

    Smooth tension/compression switch (tanh) avoids the stiffness discontinuity
    at strain=0 that otherwise makes the relaxation oscillate.
    """
    switch = 0.5 * (1.0 + np.tanh(strain / p.buckle_width))   # 0 compression → 1 tension
    base = p.k_stretch * (p.buckle_ratio + (1.0 - p.buckle_ratio) * switch)
    return base * (1.0 + p.stiffen_alpha * np.clip(strain, 0, None))


def compute_forces(net: FiberNetwork, cells: np.ndarray, p: MechParams):
    """Return (force per node, per-edge tension, per-node diagonal stiffness).

    The diagonal stiffness preconditions the relaxation so a single step size is
    stable across the (highly varying) edge stiffnesses.
    """
    nodes = net.nodes
    N = len(nodes)
    F = np.zeros((N, 3))
    Kdiag = np.full(N, 1e-6)

    # --- axial (stretch + buckling) ---------------------------------------
    i, j = net.edges[:, 0], net.edges[:, 1]
    vec = nodes[j] - nodes[i]
    L = np.linalg.norm(vec, axis=1)
    L = np.maximum(L, 1e-9)
    dirv = vec / L[:, None]
    strain = (L - net.rest_length) / net.rest_length
    k = _axial_stiffness(strain, p) * net.k_scale  # plastic reinforcement gain
    tension = k * (L - net.rest_length)           # >0 tension, <0 compression
    fvec = tension[:, None] * dirv
    np.add.at(F, i, fvec)
    np.add.at(F, j, -fvec)
    np.add.at(Kdiag, i, k)
    np.add.at(Kdiag, j, k)

    # --- bending (Laplacian curvature penalty along filament backbone) ----
    if len(net.bends):
        a, c, b = net.bends[:, 0], net.bends[:, 1], net.bends[:, 2]
        lap = nodes[a] + nodes[b] - 2.0 * nodes[c]
        np.add.at(F, c, p.k_bend * lap)
        np.add.at(F, a, -0.5 * p.k_bend * lap)
        np.add.at(F, b, -0.5 * p.k_bend * lap)
        np.add.at(Kdiag, c, 2.0 * p.k_bend)
        np.add.at(Kdiag, a, 0.5 * p.k_bend)
        np.add.at(Kdiag, b, 0.5 * p.k_bend)

    # --- cell traction (contractile cells pull nearby nodes inward) -------
    if len(cells):
        for cpos in cells:
            d = cpos - nodes                       # toward cell
            dist = np.linalg.norm(d, axis=1)
            within = (dist < p.reach) & (dist > 1e-6)
            if not np.any(within):
                continue
            dd = dist[within]
            # taper to zero inside the cell body so nodes settle on a stable
            # shell instead of overshooting through the cell (the runaway).
            ramp = np.clip((dd - p.cell_radius) / max(p.cell_radius, 1e-6), 0.0, 1.0)
            mag = p.traction * np.exp(-dd / p.reach_decay) * ramp
            F[within] += mag[:, None] * d[within] / dd[:, None]
            Kdiag[within] += mag / p.reach_decay + p.traction / max(p.cell_radius, 1e-6)

    return F, tension, Kdiag


def relax(net: FiberNetwork, cells: np.ndarray, p: MechParams | None = None,
          record=False):
    """Overdamped relaxation to mechanical equilibrium. Mutates a copy.

    Returns (relaxed_net, info) where info has convergence history + final
    per-edge tension.
    """
    p = p or MechParams()
    net = net.copy()
    free = ~net.fixed
    history = []
    tension = None

    # FIRE (Fast Inertial Relaxation Engine) on Jacobi-preconditioned forces.
    # Robust force-only minimization for stiff, partly-floppy fiber networks.
    v = np.zeros_like(net.nodes)
    dt = p.fire_dt
    alpha = p.fire_a0
    n_pos = 0
    fmax = np.inf
    it = 0
    for it in range(p.max_iter):
        F, tension, Kdiag = compute_forces(net, cells, p)
        acc = F / Kdiag[:, None]
        acc[~free] = 0.0

        power = float(np.sum(F[free] * v[free]))
        vnorm = np.linalg.norm(v)
        anorm = np.linalg.norm(acc)
        if anorm > 1e-12:
            v = (1.0 - alpha) * v + alpha * vnorm * (acc / anorm)
        if power > 0:
            n_pos += 1
            if n_pos > p.fire_nmin:
                dt = min(dt * 1.1, p.fire_dtmax)
                alpha *= 0.99
        else:
            n_pos = 0
            v[:] = 0.0
            dt = max(dt * 0.5, 1e-3)
            alpha = p.fire_a0

        v = v + dt * acc
        v[~free] = 0.0
        disp = dt * v
        mag = np.linalg.norm(disp, axis=1)
        toobig = mag > p.max_step
        if np.any(toobig):
            disp[toobig] *= (p.max_step / mag[toobig])[:, None]
        disp[~free] = 0.0
        net.nodes = net.nodes + disp

        fmax = float(np.max(np.linalg.norm(F[free], axis=1))) if free.any() else 0.0
        if record:
            history.append(fmax)
        if fmax < p.ftol:
            break

    info = dict(iters=it + 1, converged=fmax < p.ftol,
                final_max_force=fmax, tension=tension, history=history)
    return net, info


# --------------------------------------------------------------------------- #
# Morphogenesis-relevant metrics
# --------------------------------------------------------------------------- #
def edge_dirs_lengths(net: FiberNetwork):
    vec = net.nodes[net.edges[:, 1]] - net.nodes[net.edges[:, 0]]
    L = np.linalg.norm(vec, axis=1)
    L = np.maximum(L, 1e-9)
    return vec / L[:, None], L


def axis_alignment(net: FiberNetwork, axis: np.ndarray, region_pts=None,
                   band=None):
    """Mean (dir·axis)^2 of fibers, optionally restricted to a band region.

    Returns the nematic order along `axis`: 1/3 isotropic, 1 fully aligned.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    dirs, L = edge_dirs_lengths(net)
    mid = 0.5 * (net.nodes[net.edges[:, 0]] + net.nodes[net.edges[:, 1]])
    mask = np.ones(len(dirs), bool)
    if region_pts is not None and band is not None:
        a, b = region_pts
        ab = b - a
        t = np.clip(((mid - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
        proj = a + t[:, None] * ab
        dperp = np.linalg.norm(mid - proj, axis=1)
        mask = dperp < band
    if not np.any(mask):
        return float("nan"), 0
    val = (dirs[mask] @ axis) ** 2
    w = L[mask]
    return float(np.sum(w * val) / np.sum(w)), int(mask.sum())


def tension_along_axis(net, cells_pair, p, band):
    """Fraction of total tension carried by the band between two cells."""
    _, tension, _ = compute_forces(net, cells_pair, p)
    tension = np.clip(tension, 0, None)            # tensile only
    mid = 0.5 * (net.nodes[net.edges[:, 0]] + net.nodes[net.edges[:, 1]])
    a, b = cells_pair
    ab = b - a
    t = np.clip(((mid - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    dperp = np.linalg.norm(mid - proj, axis=1)
    band_mask = dperp < band
    total = np.sum(tension) + 1e-12
    return float(np.sum(tension[band_mask]) / total)
