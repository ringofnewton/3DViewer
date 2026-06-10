"""Plastic (time-evolving) ECM remodeling driven by mechanical state.

The mechanics in `ecm_mechanics` give a single elastic equilibrium: remove the
cells and the matrix springs back. Real morphogenesis is *plastic* — the matrix
records the cells' pulling history and keeps the structure. This module adds that
time evolution by alternating:

    relax to equilibrium  →  read each fiber's tension  →  remodel  →  repeat

Remodel rules (mechanobiology, "use it or lose it"):

  reinforcement   fibers under sustained tension recruit collagen and crosslink
                  (lysyl-oxidase-like): their stiffness gain k_scale grows.
  compaction      tensed fibers shorten their rest length — the matrix contracts
                  and densifies along the tension direction.
  protected turnover  slack / buckled fibers that carry no load are degraded by
                  MMP-like turnover; loaded fibers are protected. This is what
                  prunes the network down to the load-bearing tracks.

Consequence: the aligned collagen track between cells becomes a PERMANENT
structure. We prove the plasticity by removing the cells afterward and showing
the track's alignment is largely retained (an elastic-only model would relax
back to isotropic).

Thresholds are adaptive (percentiles of the current tension distribution), so the
rules are robust to the absolute force scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ecm_mechanics as mech


@dataclass
class RemodelParams:
    n_steps: int = 10
    reinforce_pct: float = 60.0     # tension percentile above which fibers reinforce
    reinforce_rate: float = 0.35    # k_scale *= (1 + rate) per step when loaded
    max_k_scale: float = 8.0
    compaction: float = 0.03        # rest-length shortening of loaded fibers / step
    min_rest_frac: float = 0.55     # don't shorten a fiber below this of its original
    slack_pct: float = 25.0         # below this tension percentile counts as "slack"
    slack_steps: int = 3            # consecutive slack steps before degradation
    protect_k: float = 1.4          # fibers reinforced past this are never degraded
    seed: int = 0


def _edge_tension(net, cells, p_mech):
    _, tension, _ = mech.compute_forces(net, cells, p_mech)
    return tension


def remodel(net0: mech.FiberNetwork, cells: np.ndarray,
            p_mech: mech.MechParams | None = None,
            p_rem: RemodelParams | None = None, record=True):
    """Run plastic remodeling. Returns (frames, final_net).

    frames[i] = dict(step, net (copy), tension, n_edges, frac_reinforced, alignment)
    """
    p_mech = p_mech or mech.MechParams()
    p_rem = p_rem or RemodelParams()
    net = net0.copy()
    orig_rest = net.rest_length.copy()
    slack_age = np.zeros(len(net.edges), dtype=int)
    frames = []

    axis = cells[1] - cells[0] if len(cells) >= 2 else np.array([1.0, 0, 0])
    region = (cells[0], cells[1]) if len(cells) >= 2 else None

    for step in range(p_rem.n_steps):
        net, _ = mech.relax(net, cells, p_mech)
        tension = _edge_tension(net, cells, p_mech)
        pos_t = tension[tension > 0]
        hi = np.percentile(pos_t, p_rem.reinforce_pct) if len(pos_t) else np.inf
        lo = np.percentile(np.abs(tension), p_rem.slack_pct) if len(tension) else 0.0

        loaded = tension >= hi
        slack = np.abs(tension) <= lo

        # reinforcement + compaction of loaded fibers
        net.k_scale[loaded] = np.minimum(net.k_scale[loaded] * (1 + p_rem.reinforce_rate),
                                         p_rem.max_k_scale)
        net.rest_length[loaded] = np.maximum(
            net.rest_length[loaded] * (1 - p_rem.compaction),
            orig_rest[loaded] * p_rem.min_rest_frac)
        net.state[loaded] = "reinforced"

        # protected turnover of persistently slack fibers
        slack_age = np.where(slack, slack_age + 1, 0)
        degrade = slack & (slack_age >= p_rem.slack_steps) & (net.k_scale < p_rem.protect_k)

        if record:
            al = mech.axis_alignment(net, axis, region, 22.0)[0] if region else np.nan
            frames.append(dict(step=step, net=net.copy(), tension=tension.copy(),
                               n_edges=len(net.edges),
                               frac_reinforced=float(np.mean(net.k_scale > p_rem.protect_k)),
                               alignment=al))

        if np.any(degrade):
            keep = ~degrade
            net = _filter_edges(net, keep)
            orig_rest = orig_rest[keep]
            slack_age = slack_age[keep]

    return frames, net


def _filter_edges(net: mech.FiberNetwork, keep: np.ndarray) -> mech.FiberNetwork:
    """Return a network with only the kept edges (bends referencing dropped
    edges are conservatively retained — they reference nodes, not edges)."""
    return mech.FiberNetwork(
        nodes=net.nodes, edges=net.edges[keep], rest_length=net.rest_length[keep],
        bends=net.bends, fixed=net.fixed, state=net.state[keep],
        k_scale=net.k_scale[keep])


def retained_after_cell_removal(net: mech.FiberNetwork, cells: np.ndarray,
                                p_mech: mech.MechParams):
    """Relax with NO cell traction and report how much track alignment remains.

    Plastic matrix keeps its track; a purely elastic one springs back.
    """
    axis = cells[1] - cells[0]
    region = (cells[0], cells[1])
    al_before = mech.axis_alignment(net, axis, region, 22.0)[0]
    relaxed, _ = mech.relax(net, np.zeros((0, 3)), p_mech)
    al_after = mech.axis_alignment(relaxed, axis, region, 22.0)[0]
    return al_before, al_after, relaxed


def residual_self_stress(net: mech.FiberNetwork, p_mech: mech.MechParams):
    """Total tensile force the matrix holds at equilibrium with NO cells.

    The cleanest signature of plasticity: compaction shortens rest lengths so the
    reinforced track stays pre-stressed (self-equilibrated tension) after the
    cells are gone. An elastic network relaxes to ~zero. Returns (total_tension,
    relaxed_net).
    """
    relaxed, _ = mech.relax(net, np.zeros((0, 3)), p_mech)
    _, tension, _ = mech.compute_forces(relaxed, np.zeros((0, 3)), p_mech)
    return float(np.sum(np.clip(tension, 0, None))), relaxed
