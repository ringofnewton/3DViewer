#!/usr/bin/env python3
"""Mechanochemical remodeling vs the percentile heuristic — Step 3 verify.

Claim: replacing the adaptive-percentile remodel rule with first-principles
kinetics (tension-saturating synthesis + strain-PROTECTED MMP degradation, with
tracked collagen mass) reproduces the same emergent load-bearing track between
two contractile cells — but now the selection comes from a measurable
tension-protection curve and collagen mass is conserved/redistributed, not
invented.

We run BOTH remodelers from the identical initial network + cell pair and compare
the inter-cell track alignment (vs the isotropic baseline 1/3). We also report the
collagen-mass trajectory (mechanochem) and the protection curve that drives it.

    python verify_mechanochem.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech
from ecm_pipeline import ecm_remodel as rem
from ecm_pipeline import ecm_mechanochem as mc

DOMAIN, SLAB = 160.0, 26.0
SEEDS = [1, 2, 3]
T_CRIT = {3: 4.303, 4: 3.182}


def stat(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), T_CRIT.get(n, 2.776) * x.std(ddof=1) / np.sqrt(n)


def setup(seed):
    net = mech.build_random_network(DOMAIN, n_fibers=260, fiber_len=70.0, seg_len=7.0,
                                    crosslink_dist=4.5, seed=seed, slab_thickness=SLAB)
    c = DOMAIN / 2.0
    cells = np.array([[c - 30, c, c], [c + 30, c, c]])     # pair along x
    return net, cells


def track(net, cells):
    axis = cells[1] - cells[0]
    return mech.axis_alignment(net, axis, (cells[0], cells[1]), 22.0)[0]


def main():
    pmech = mech.MechParams(planar=True, traction=1.5, max_iter=4000, ftol=2e-2)
    print(f"Mechanochemical vs percentile remodeling  (n={len(SEEDS)} seeds, mean +/- 95% CI)")
    print("inter-cell track alignment along the cell-cell axis (isotropic = 0.333)\n")

    pct_al, mc_al, mc_mass0, mc_mass1, mc_n0, mc_n1 = [], [], [], [], [], []
    for s in SEEDS:
        net, cells = setup(s)

        # percentile remodeler (existing)
        _, net_pct = rem.remodel(net, cells, pmech,
                                 rem.RemodelParams(n_steps=10), record=False)
        pct_al.append(track(net_pct, cells))

        # mechanochemical remodeler (new)
        frames, net_mc = mc.remodel(net, cells, pmech,
                                    mc.MechanoChemParams(n_steps=10), record=True)
        mc_al.append(track(net_mc, cells))
        mc_mass0.append(frames[0]["mass_total"]); mc_mass1.append(frames[-1]["mass_total"])
        mc_n0.append(frames[0]["n_edges"]); mc_n1.append(len(net_mc.edges))

    mp, cp = stat(pct_al); mm, cm = stat(mc_al)
    print(f"  percentile rule      track align = {mp:.3f} +/- {cp:.3f}")
    print(f"  mechanochemical      track align = {mm:.3f} +/- {cm:.3f}")
    print(f"  (both >> 0.333 isotropic: {'YES' if mp > 0.4 and mm > 0.4 else 'check'})")

    consistent = abs(mp - mm) < 0.12
    print(f"\n  agreement |pct - mechanochem| = {abs(mp-mm):.3f}  "
          f"({'consistent — same emergent track from enzyme kinetics' if consistent else 'differ'})")

    print(f"\nCollagen mass (mechanochem, tracked):")
    print(f"  total mass start -> end : {np.mean(mc_mass0):.0f} -> {np.mean(mc_mass1):.0f}"
          f"  (synthesis on loaded fibers, degradation of slack)")
    print(f"  fibers      start -> end : {np.mean(mc_n0):.0f} -> {np.mean(mc_n1):.0f}"
          f"  (strain-protected pruning)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        T, koff = mc.protection_curve()
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].bar([0, 1], [mp, mm], yerr=[cp, cm], capsize=5,
                  color=["#7f8c8d", "#27ae60"])
        ax[0].axhline(1/3, color="#c0392b", ls="--", label="isotropic 0.333")
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["percentile", "mechanochem"])
        ax[0].set_ylabel("inter-cell track alignment"); ax[0].legend(fontsize=8)
        ax[0].set_title("Same emergent track,\ntwo remodeling laws")
        ax[1].plot(T, koff, color="#c0392b", lw=2)
        ax[1].set_xlabel("fiber tension"); ax[1].set_ylabel("MMP degradation rate")
        ax[1].set_title("Strain-protection curve\n(loaded fibers resist cleavage)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "mechanochem.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'mechanochem.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
