#!/usr/bin/env python3
"""3D reproduction of the headline results (VALIDATION.md §2.5).

The constitutive force law is fully 3D; the reported studies use a planar (2D)
constraint for cleaner statistics. This checks the two central conclusions survive
in a **full 3D** network (no planar constraint):

  A. Single-cell radial aster — tension-weighted vs uniform radiality.
  B. Inter-cell bridge        — alignment along the focus axis vs the two
                                transverse (perpendicular) axes.

In 3D the isotropic nematic order along any axis is 1/3. Reports mean ± 95% CI
over seeds and writes output_validation/figures/validate_3d.png.

    python validate_3d.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech

D = 140.0
C = D / 2.0
N = 10
SEEDS = list(range(1, N + 1))
T_CRIT = {6: 2.571, 8: 2.365, 10: 2.262, 12: 2.201}
ISO = 1.0 / 3.0


def _net(seed):
    # full 3D: no slab_thickness -> isotropic 3D Mikado network
    return mech.build_random_network(D, n_fibers=220, fiber_len=60.0, seg_len=9.0,
                                     crosslink_dist=5.5, seed=seed)


def aster3d(seed):
    net0 = _net(seed)
    P = mech.MechParams(traction=1.5, reach=34.0, reach_decay=20.0, max_iter=2500,
                        planar=False)
    single = np.array([[C, C, C]])
    net, _ = mech.relax(net0, single, P)
    _, tension, _ = mech.compute_forces(net, single, P)
    dirs, _ = mech.edge_dirs_lengths(net)
    mid = 0.5 * (net.nodes[net.edges[:, 0]] + net.nodes[net.edges[:, 1]])
    radial = mid - single[0]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    radiality = np.abs(np.einsum("ij,ij->i", dirs, radial))
    near = np.linalg.norm(mid - single[0], axis=1) < P.reach
    ten = np.clip(tension, 0, None)
    return (float(np.average(radiality[near], weights=ten[near] + 1e-9)),
            float(np.mean(radiality[near])))


def bridge3d(seed, sep=44.0):
    net0 = _net(seed)
    a = np.array([C - sep / 2, C, C]); b = np.array([C + sep / 2, C, C])
    P = mech.MechParams(traction=1.5, reach=34.0, reach_decay=20.0, buckle_ratio=0.02,
                        max_iter=2500, planar=False)
    net, _ = mech.relax(net0, np.array([a, b]), P)
    al, _ = mech.axis_alignment(net, b - a, (a, b), band=18.0)
    py, _ = mech.axis_alignment(net, np.array([0.0, 1.0, 0.0]), (a, b), band=18.0)
    pz, _ = mech.axis_alignment(net, np.array([0.0, 0.0, 1.0]), (a, b), band=18.0)
    return float(al), float(0.5 * (py + pz))


def stat(x):
    x = np.asarray(x); return x.mean(), T_CRIT[N] * x.std(ddof=1) / np.sqrt(len(x))


def main():
    print(f"3D reproduction of the headline results (full 3D, n={N} seeds, 95% CI)")
    print(f"(isotropic nematic order along any axis = {ISO:.3f})\n")
    aw, au = zip(*[aster3d(s) for s in SEEDS])
    bx, bp = zip(*[bridge3d(s) for s in SEEDS])
    maw, caw = stat(aw); mau, cau = stat(au)
    mbx, cbx = stat(bx); mbp, cbp = stat(bp)

    # 2D references (from stats_power.py, n=16)
    ref = dict(aw=0.969, au=0.689, bx=0.532, bp=0.371)
    print("A. Single-cell radial aster")
    print(f"   tension-weighted radiality  3D {maw:.3f} ± {caw:.3f}   (2D {ref['aw']:.3f})")
    print(f"   uniform radiality           3D {mau:.3f} ± {cau:.3f}   (2D {ref['au']:.3f})")
    print(f"   → aster present in 3D: {'YES' if maw - mau > 0.05 else 'NO'} "
          f"(Δ {maw-mau:.3f})")
    print("\nB. Inter-cell bridge (sep=44 µm)")
    print(f"   axis alignment              3D {mbx:.3f} ± {cbx:.3f}   (2D {ref['bx']:.3f})")
    print(f"   transverse control          3D {mbp:.3f} ± {cbp:.3f}   (2D {ref['bp']:.3f})")
    print(f"   → bridge present in 3D: {'YES' if (mbx>ISO+0.05 and mbx-mbp>0.05) else 'NO'} "
          f"(axis−perp {mbx-mbp:.3f})")
    ok = (maw - mau > 0.05) and (mbx > ISO + 0.05) and (mbx - mbp > 0.05)
    print("\nPASS  the 2D headline conclusions reproduce in full 3D"
          if ok else "\nCHECK  see numbers above")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].bar([0, 1], [maw, mau], yerr=[caw, cau], capsize=5, color=["#c0392b", "#95a5a6"])
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["tension-wtd", "uniform"])
        ax[0].set_ylim(0, 1); ax[0].set_title("A. Single-cell aster (3D)")
        ax[1].bar([0, 1], [mbx, mbp], yerr=[cbx, cbp], capsize=5, color=["#c0392b", "#2c6fbb"])
        ax[1].axhline(ISO, color="0.5", ls=":", label="isotropic (1/3)")
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["bridge axis", "transverse"])
        ax[1].set_ylim(0, 0.7); ax[1].set_title("B. Inter-cell bridge (3D)"); ax[1].legend(fontsize=8)
        for a in ax: a.set_ylabel("nematic alignment"); a.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Full-3D reproduction: mean ± 95% CI (n={N} seeds)")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "validate_3d.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'validate_3d.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
