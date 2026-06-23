"""Viscoelasticity from transient crosslinks — Step 2 of the roadmap.

The pipeline so far relaxes to a single elastic equilibrium: hold a strain and the
stress stays forever. Real ECM is *viscoelastic* — held at fixed strain, stress
relaxes over time as crosslinks unbind, fibers slide, and bonds turn over. Cell
behavior (spreading, fate, migration) depends strongly on this relaxation time,
not just on the modulus (Chaudhuri et al., 2016).

Mechanism implemented: crosslinks are transient bonds with force-accelerated
unbinding (Bell, 1978):

    k_off(F) = k0 * exp(|F| / F_star)

A loaded crosslink is more likely to rupture; when it does, the constraint it
provided is removed and the held stress partially relaxes. The fiber backbone
(covalent, not modeled as transient) sets the long-time elastic plateau, so the
material behaves as a viscoelastic *solid*. Optional stress-free rebinding
(`k_on`) lets ruptured crosslinks reform unstressed, deepening the relaxation
toward a fluid-like response.

This is built on the periodic (Lees-Edwards) network so the stress is the
boundary-free virial stress and the step strain is felt uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ecm_mechanics as mech
from . import ecm_periodic as per


@dataclass
class ViscoParams:
    k0: float = 0.05          # bare crosslink off-rate (per unit time)
    F_star: float = 0.04      # Bell force scale (reduced force); smaller = more force-sensitive
    k_on: float = 0.0         # stress-free rebinding rate of ruptured crosslinks
    dt: float = 1.0           # time step (reduced time units)
    n_steps: int = 40
    seed: int = 0


def _drop_crosslinks(net: per.PeriodicNetwork, keep_xlink: np.ndarray) -> per.PeriodicNetwork:
    """Return a copy keeping all backbone edges and only the flagged crosslinks.
    `keep_xlink` is a boolean mask over the crosslink edges (edges[n_backbone:])."""
    nb = net.n_backbone
    keep = np.ones(len(net.edges), bool)
    keep[nb:] = keep_xlink
    return per.PeriodicNetwork(
        net.nodes, net.edges[keep], net.edge_off[keep], net.rest_length[keep],
        net.bends, net.k_scale[keep], net.L, nb)


def stress_relaxation(net0: per.PeriodicNetwork, gamma0=0.03,
                      vp: ViscoParams | None = None,
                      p_mech: mech.MechParams | None = None,
                      bell=True):
    """Step-strain stress relaxation (mean-field transient-network form).

    Apply an affine step shear gamma0, relax, then hold the strain while each
    crosslink loses load-bearing capacity at its force-accelerated unbinding rate
    (Bell). Rather than removing bonds stochastically (noisy), we evolve a
    continuous survival fraction s_i in [0,1] and scale that crosslink's stiffness
    by s_i — the ensemble-averaged transient network, which gives a smooth,
    deterministic relaxation curve.

        s_i(t+dt) = s_i * exp(-k_off_i dt),   k_off_i = k0 exp(|F_i|/F_star)
        (+ optional stress-free rebinding toward s=1 at rate k_on)

    Returns dict(t, sigma, surv) where surv is the mean crosslink survival.
    If bell=False, survival is frozen (elastic reference: sigma stays flat).
    """
    vp = vp or ViscoParams()
    p_mech = p_mech or mech.MechParams(traction=0.0, max_iter=4000, ftol=2.5e-2,
                                       k_bend=0.15)
    nb = net0.n_backbone
    n_x = len(net0.edges) - nb

    net = net0.copy()
    net.nodes[:, 0] = np.mod(net.nodes[:, 0] + gamma0 * net.nodes[:, 1], net.L)
    net, _ = per.relax(net, gamma0, p_mech)

    surv = np.ones(n_x)                      # crosslink survival fraction
    ts, sig, sv = [], [], []
    for step in range(vp.n_steps + 1):
        net.k_scale[nb:] = surv              # crosslink stiffness scaled by survival
        net, _ = per.relax(net, gamma0, p_mech)
        sigma = per.virial_shear_stress(net, gamma0, p_mech)
        ts.append(step * vp.dt); sig.append(sigma); sv.append(float(surv.mean()))

        if not bell:
            continue
        _, tension, _, _, _ = per._forces(net, gamma0, p_mech)
        xt = np.abs(tension[nb:])            # per crosslink (effective, force already ~ s*base)
        k_off = vp.k0 * np.exp(np.clip(xt / vp.F_star, 0, 30))
        surv = surv * np.exp(-k_off * vp.dt)
        if vp.k_on > 0:                       # stress-free rebinding toward full
            surv = surv + (1.0 - surv) * (1.0 - np.exp(-vp.k_on * vp.dt))

    return dict(t=np.array(ts), sigma=np.array(sig), surv=np.array(sv),
                sigma0=sig[0])


def relaxation_time(t, sigma):
    """Time for stress to fall to 1/e of the way from sigma0 to its plateau.

    Returns tau (interpolated) and the plateau fraction sigma_inf/sigma0.
    """
    sigma = np.asarray(sigma, float)
    s0 = sigma[0]; s_inf = sigma[-1]
    if s0 <= s_inf + 1e-12:
        return float("inf"), s_inf / s0 if s0 else float("nan")
    target = s_inf + (s0 - s_inf) / np.e
    below = np.where(sigma <= target)[0]
    if not len(below):
        return float("inf"), s_inf / s0
    i = below[0]
    if i == 0:
        return 0.0, s_inf / s0
    # linear interpolation in time
    t0, t1 = t[i - 1], t[i]
    a0, a1 = sigma[i - 1], sigma[i]
    frac = (a0 - target) / (a0 - a1 + 1e-12)
    return float(t0 + frac * (t1 - t0)), float(s_inf / s0)
