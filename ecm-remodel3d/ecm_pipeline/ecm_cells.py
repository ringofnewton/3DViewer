"""Active cell agents: migration + mechanosensing — Step 4 of the roadmap.

So far cells were fixed contractile foci. Real cells are active agents that read
the matrix mechanics and move:

  rigidity sensing   a cell pulls harder on a stiffer neighbourhood (traction
                     scales with local matrix stiffness) — Lo 2000, Engler 2006.
  durotaxis          cells migrate up a stiffness gradient.
  contact guidance   cells migrate preferentially along the local fiber
                     alignment axis.

This module adds a thin agent layer on top of the `ecm_mechanics` network. Each
update: relax the matrix under the cells' traction, let every cell sense its local
mechanical environment (a proximity-weighted stiffness proxy and the local fiber
orientation), then step the cell up the stiffness gradient with a contact-guidance
bias. The matrix and the cells thus co-evolve — the heart of morphogenesis.

The stiffness proxy is the load-bearing material nearby:
    S(x) = sum_edges  k_base * k_scale_e * w(|mid_e - x|)
which rises with both fiber density and plastic reinforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ecm_mechanics as mech


@dataclass
class CellParams:
    sense_radius: float = 22.0      # neighbourhood the cell integrates over (um)
    probe: float = 6.0              # finite-difference step for the gradient (um)
    speed: float = 4.0              # max migration step per update (um)
    durotaxis: float = 1.0          # weight on the stiffness-gradient direction
    guidance: float = 0.6           # weight on the local fiber-alignment direction
    base_traction: float = 1.2      # traction at reference stiffness
    rigidity_gain: float = 0.8      # extra traction per unit local stiffness (sensing)
    planar: bool = True


def _stiffness_proxy(net: mech.FiberNetwork, pts, p: CellParams):
    """Proximity-weighted load-bearing material near each query point (..,3)->(..)."""
    mid = 0.5 * (net.nodes[net.edges[:, 0]] + net.nodes[net.edges[:, 1]])
    w_edge = net.k_scale                                   # plastic stiffness gain
    pts = np.atleast_2d(pts)
    out = np.zeros(len(pts))
    r2 = p.sense_radius ** 2
    for i, x in enumerate(pts):
        d2 = np.sum((mid - x) ** 2, axis=1)
        m = d2 < r2
        if np.any(m):
            wt = np.exp(-d2[m] / (0.5 * r2))
            out[i] = float(np.sum(w_edge[m] * wt))
    return out


def _local_orientation(net: mech.FiberNetwork, x, p: CellParams):
    """Dominant fiber direction near x (principal axis of the orientation tensor)."""
    mid = 0.5 * (net.nodes[net.edges[:, 0]] + net.nodes[net.edges[:, 1]])
    d2 = np.sum((mid - x) ** 2, axis=1)
    m = d2 < p.sense_radius ** 2
    if m.sum() < 3:
        return np.zeros(3)
    dirs, L = mech.edge_dirs_lengths(net)
    dirs = dirs[m]; w = L[m]
    Q = (dirs * w[:, None]).T @ dirs                      # weighted orientation tensor
    vals, vecs = np.linalg.eigh(Q)
    return vecs[:, -1]                                     # principal axis


def sense_traction(net: mech.FiberNetwork, cells, p: CellParams):
    """Per-cell traction magnitude from local rigidity sensing (normalized)."""
    S = _stiffness_proxy(net, cells, p)
    S = S / (np.mean(S) + 1e-9)                            # relative stiffness
    return p.base_traction * (1.0 + p.rigidity_gain * (S - 1.0))


def migrate(net: mech.FiberNetwork, cells0, p_cell: CellParams | None = None,
            p_mech: mech.MechParams | None = None, n_steps=12, record=True):
    """Co-evolve matrix + migrating cells. Returns (trajectory, final_net).

    trajectory: (n_steps+1, M, 3) cell positions over time.
    """
    p_cell = p_cell or CellParams()
    p_mech = p_mech or mech.MechParams(planar=p_cell.planar)
    cells = np.array(cells0, dtype=float)
    traj = [cells.copy()]
    net = net.copy()

    for step in range(n_steps):
        # rigidity sensing sets each cell's traction this step
        p_mech.traction = float(np.mean(sense_traction(net, cells, p_cell)))
        net, _ = mech.relax(net, cells, p_mech)

        new = cells.copy()
        for ci, x in enumerate(cells):
            # stiffness gradient by central finite differences (durotaxis)
            grad = np.zeros(3)
            axes = (0, 1) if p_cell.planar else (0, 1, 2)
            for ax in axes:
                xp = x.copy(); xm = x.copy()
                xp[ax] += p_cell.probe; xm[ax] -= p_cell.probe
                sp, sm = _stiffness_proxy(net, [xp, xm], p_cell)
                grad[ax] = (sp - sm) / (2 * p_cell.probe)
            gnorm = np.linalg.norm(grad)
            dvec = grad / gnorm if gnorm > 1e-9 else np.zeros(3)

            # contact guidance along local fiber axis (sign toward stiffer side)
            ax_dir = _local_orientation(net, x, p_cell)
            if ax_dir @ dvec < 0:
                ax_dir = -ax_dir

            move = p_cell.durotaxis * dvec + p_cell.guidance * ax_dir
            mnorm = np.linalg.norm(move)
            if mnorm > 1e-9:
                new[ci] = x + p_cell.speed * move / mnorm
            if p_cell.planar:
                new[ci, 2] = x[2]
        cells = new
        if record:
            traj.append(cells.copy())

    return np.array(traj), net


def net_displacement(traj):
    """Per-cell net displacement vector and its mean x-component (durotaxis axis)."""
    disp = traj[-1] - traj[0]
    return disp, float(np.mean(disp[:, 0]))
