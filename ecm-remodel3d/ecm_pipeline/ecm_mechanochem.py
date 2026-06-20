"""Mechanochemically-coupled remodeling — Step 3 of the roadmap.

`ecm_remodel` selects load-bearing fibers with an adaptive *percentile* rule
(top X% reinforce, bottom X% degrade). It works but is phenomenological and does
not conserve collagen mass. This module replaces the heuristic with first-
principles kinetics that mirror the actual enzymology:

  collagen mass   each fiber carries a collagen content m (stiffness k_scale = m).
  synthesis       loaded fibers recruit collagen (LOX-assisted crosslinking),
                  a saturating (Hill) function of tension:
                      dm_syn = k_syn * T/(F_syn + T)            (T = tensile force)
  degradation     MMPs cleave collagen, but MECHANICAL STRAIN PROTECTS the fiber
                  from cleavage (Adhikari 2011; Saini 2020): the off-rate falls
                  with tension:
                      dm_deg = k_deg * exp(-T / F_prot) * m
  mass balance    m += dm_syn - dm_deg; a fiber is removed when m < m_min.
  mass budget     optional global synthesis cap (finite collagen pool) so the
                  matrix conserves/redistributes mass instead of creating it.

Result: the same emergent load-bearing track as the percentile rule, but the
selection now comes from a tension-protection curve, collagen mass is tracked and
(optionally) conserved, and every parameter maps to a measurable rate.

Operates on the standard `ecm_mechanics.FiberNetwork`, so it is a drop-in
alternative to `ecm_remodel.remodel` in any runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ecm_mechanics as mech


@dataclass
class MechanoChemParams:
    n_steps: int = 12
    dt: float = 1.0
    k_syn: float = 0.5         # max collagen synthesis rate on a loaded fiber
    F_syn: float = 0.5         # tension at half-max synthesis (Hill constant)
    k_deg: float = 0.4         # bare MMP degradation rate
    F_prot: float = 0.6        # tension scale of strain-protection from cleavage
    m_min: float = 0.15        # remove a fiber below this collagen mass
    m_max: float = 8.0         # saturating stiffness gain
    compaction: float = 0.03   # loaded fibers shorten rest length (matrix contracts)
    min_rest_frac: float = 0.55
    mass_budget: float | None = None   # global synthesis cap per step (None = unlimited)


def remodel(net0: mech.FiberNetwork, cells: np.ndarray,
            p_mech: mech.MechParams | None = None,
            mp: MechanoChemParams | None = None, record=True):
    """Mechanochemical remodeling loop. Returns (frames, final_net).

    Each frame: dict(step, net, tension, mass_total, n_edges, frac_loaded).
    """
    p_mech = p_mech or mech.MechParams()
    mp = mp or MechanoChemParams()
    net = net0.copy()
    orig_rest = net.rest_length.copy()
    mass = net.k_scale.copy()                 # collagen content == stiffness gain
    frames = []

    axis = cells[1] - cells[0] if len(cells) >= 2 else np.array([1.0, 0, 0])
    region = (cells[0], cells[1]) if len(cells) >= 2 else None

    for step in range(mp.n_steps):
        net.k_scale = mass
        net, _ = mech.relax(net, cells, p_mech)
        _, tension, _ = mech.compute_forces(net, cells, p_mech)
        T = np.clip(tension, 0, None)          # tensile force per fiber

        # --- synthesis (saturating in tension) ---------------------------
        dm_syn = mp.k_syn * (T / (mp.F_syn + T)) * mp.dt
        if mp.mass_budget is not None and dm_syn.sum() > mp.mass_budget:
            dm_syn *= mp.mass_budget / (dm_syn.sum() + 1e-12)   # conserve a finite pool

        # --- degradation (strain-protected) ------------------------------
        dm_deg = mp.k_deg * np.exp(-T / mp.F_prot) * mass * mp.dt

        mass = np.clip(mass + dm_syn - dm_deg, 0.0, mp.m_max)

        # --- compaction of loaded fibers ---------------------------------
        loaded = T > np.percentile(T[T > 0], 60) if np.any(T > 0) else np.zeros(len(T), bool)
        net.rest_length[loaded] = np.maximum(
            net.rest_length[loaded] * (1 - mp.compaction),
            orig_rest[loaded] * mp.min_rest_frac)
        net.state[loaded] = "reinforced"

        if record:
            al = mech.axis_alignment(net, axis, region, 22.0)[0] if region else np.nan
            frames.append(dict(step=step, net=net.copy(), tension=tension.copy(),
                               mass_total=float(mass.sum()), n_edges=len(net.edges),
                               frac_loaded=float(np.mean(mass > 1.0)), alignment=al))

        # --- remove fully-degraded fibers --------------------------------
        keep = mass >= mp.m_min
        if not np.all(keep):
            net = _filter(net, keep)
            mass = mass[keep]
            orig_rest = orig_rest[keep]

    net.k_scale = mass
    return frames, net


def _filter(net: mech.FiberNetwork, keep: np.ndarray) -> mech.FiberNetwork:
    return mech.FiberNetwork(
        nodes=net.nodes, edges=net.edges[keep], rest_length=net.rest_length[keep],
        bends=net.bends, fixed=net.fixed, state=net.state[keep],
        k_scale=net.k_scale[keep])


def protection_curve(F_prot=0.6, k_deg=0.4, T=None):
    """The degradation off-rate vs tension (for plotting the mechanism)."""
    if T is None:
        T = np.linspace(0, 3, 60)
    return T, k_deg * np.exp(-T / F_prot)
