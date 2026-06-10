#!/usr/bin/env python3
"""Regression guard for the ECM mechanics — asserts the physics predicts right.

Checks the three behaviors that make the model usable for morphogenesis:

  1. Relaxation reaches mechanical equilibrium (force balance).
  2. A single contractile cell builds RADIAL tension around itself (fibers
     pointing at the cell carry more tension than tangential ones).
  3. The buckling nonlinearity focuses force: between two cells, the nonlinear
     network carries a larger fraction of its tension in the connecting band
     than a linear control — this is what creates long-range collagen tracks.

    python verify_mechanics.py
"""

from __future__ import annotations

import numpy as np

from ecm_pipeline import ecm_mechanics as mech

DOMAIN = 200.0
ZC = DOMAIN / 2.0


def main():
    P = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0, buckle_ratio=0.02)
    P_LIN = mech.MechParams(traction=1.5, reach=40.0, reach_decay=22.0,
                            buckle_ratio=1.0, stiffen_alpha=0.0)

    # 1. convergence -------------------------------------------------------
    net0 = mech.build_random_network(DOMAIN, n_fibers=220, seed=3, slab_thickness=26)
    c = DOMAIN / 2
    cells = np.array([[c - 28, c, ZC], [c + 28, c, ZC]])
    net_nl, info = mech.relax(net0, cells, P, record=True)
    assert info["converged"], f"relaxation did not reach force balance (fmax={info['final_max_force']:.3f})"

    # 2. single-cell radial tension ---------------------------------------
    single = np.array([[c, c, ZC]])
    net_s, _ = mech.relax(net0, single, P)
    _, tension, _ = mech.compute_forces(net_s, single, P)
    dirs, _ = mech.edge_dirs_lengths(net_s)
    mid = 0.5 * (net_s.nodes[net_s.edges[:, 0]] + net_s.nodes[net_s.edges[:, 1]])
    radial = mid - single[0]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    radiality = np.abs(np.einsum("ij,ij->i", dirs, radial))   # 1 = points at cell
    near = np.linalg.norm(mid - single[0], axis=1) < P.reach
    ten = np.clip(tension, 0, None)
    rad_w = np.average(radiality[near], weights=ten[near] + 1e-9)
    rad_u = np.mean(radiality[near])
    assert rad_w > rad_u, "tension is not preferentially radial around a single cell"

    # 3. the buckling nonlinearity is REQUIRED to align a track between cells:
    #    the nonlinear matrix gains axis alignment, a linear control does not.
    axis = cells[1] - cells[0]
    al_init, _ = mech.axis_alignment(net0, axis, (cells[0], cells[1]), 22.0)
    al_nl, _ = mech.axis_alignment(net_nl, axis, (cells[0], cells[1]), 22.0)
    net_lin, _ = mech.relax(net0, cells, P_LIN)
    al_lin, _ = mech.axis_alignment(net_lin, axis, (cells[0], cells[1]), 22.0)
    assert al_nl > al_init + 0.03, "nonlinear matrix should align into a track"
    assert al_nl > al_lin + 0.03, "nonlinear should out-align the linear control"

    print("PASS  mechanics predictions hold")
    print(f"      converged in {info['iters']} iters (max nodal force {info['final_max_force']:.1e})")
    print(f"      single-cell radiality: tension-weighted {rad_w:.2f} > uniform {rad_u:.2f}")
    print(f"      inter-cell track alignment: init {al_init:.2f} → nonlinear {al_nl:.2f} "
          f"(linear control {al_lin:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
