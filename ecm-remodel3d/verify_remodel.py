#!/usr/bin/env python3
"""Regression guard for plastic remodeling.

Asserts the three behaviors that make the remodeling plastic (not just elastic):

  1. protected turnover prunes the network (fiber count drops).
  2. loaded fibers reinforce (reinforced fraction rises).
  3. the track is RETAINED after the cells are removed, and more so than an
     elastic-only equilibrium (plastic memory).

    python verify_remodel.py
"""

from __future__ import annotations

import numpy as np

from ecm_pipeline import ecm_mechanics as mech, ecm_remodel as rem

DOMAIN, ZC = 200.0, 100.0


def main():
    c = 100
    cells = np.array([[c - 28, c, ZC], [c + 28, c, ZC]])
    P = mech.MechParams(reach=40, reach_decay=22)
    RP = rem.RemodelParams(n_steps=8, compaction=0.05, reinforce_rate=0.4,
                           slack_pct=30, slack_steps=2)

    net0 = mech.build_random_network(DOMAIN, n_fibers=180, seed=3, slab_thickness=26)
    frames, final = rem.remodel(net0, cells, P, RP)

    n_start, n_end = frames[0]["n_edges"], frames[-1]["n_edges"]
    reinf = frames[-1]["frac_reinforced"]
    assert n_end < 0.8 * n_start, f"network was not pruned ({n_start} -> {n_end})"
    assert reinf > 0.2, f"too few fibers reinforced ({reinf:.2f})"

    elastic, _ = mech.relax(net0, cells, P)
    res_plastic, _ = rem.residual_self_stress(final, P)
    res_elastic, _ = rem.residual_self_stress(elastic, P)
    assert res_plastic > 5 * res_elastic + 1, \
        "plastic matrix should hold residual self-stress after cells are removed"

    print("PASS  plastic remodeling behaves")
    print(f"      fibers pruned {n_start} → {n_end}  ({reinf:.0%} reinforced)")
    print(f"      residual self-stress after cell removal: "
          f"plastic {res_plastic:.0f}  vs  elastic {res_elastic:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
