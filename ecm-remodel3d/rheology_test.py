#!/usr/bin/env python3
"""Best-effort network shear-modulus measurement (closes CALIBRATION.md §2.1).

Clamped simple shear: fix top & bottom boundary layers, impose an affine shear
gamma, relax the interior to mechanical equilibrium, and read the reaction shear
stress on the top plate. G_reduced = stress/gamma. Converted to SI via the
calibrated stress unit (~156 Pa) and compared to the collagen-gel range 1-100 Pa.

A sub-isostatic collagen network has soft shear modes, so this is reported as an
order-of-magnitude estimate with the residual force balance (fmax) shown.

    python rheology_test.py
"""
from __future__ import annotations
import numpy as np
from ecm_pipeline import ecm_mechanics as mech
from ecm_pipeline.calibration import calibrate

DOMAIN, SLAB = 200.0, 26.0

def shear_modulus(n_fibers, seg_len, seed, gamma, xdist=6.0, margin=15.0):
    net = mech.build_random_network(DOMAIN, n_fibers=n_fibers, fiber_len=70.0,
        seg_len=seg_len, crosslink_dist=xdist, seed=seed, slab_thickness=SLAB)
    ys = net.nodes[:,1].copy()
    bottom, top = ys < margin, ys > DOMAIN-margin
    net.fixed = bottom | top
    net.nodes[:,0] = net.nodes[:,0] + gamma*ys          # affine simple-shear guess, held at plates
    p = mech.MechParams(planar=True, max_iter=12000, ftol=2e-2, traction=0.0)
    relaxed, info = mech.relax(net, np.zeros((0,3)), p)
    F,_,_ = mech.compute_forces(relaxed, np.zeros((0,3)), p)
    stress = abs(float(np.sum(F[top,0]))) / (DOMAIN*SLAB)
    return stress/gamma, info["final_max_force"], info["converged"]

def main():
    cal = calibrate(); su = cal["stress_unit_Pa"]
    print(f"stress unit = {su:.1f} Pa   (collagen gel target ~1-100 Pa)\n")
    print(f"{'gamma':>6} {'seed':>5} {'G_reduced':>10} {'fmax':>8} {'conv':>6} {'G_SI(Pa)':>9}")
    Gs=[]
    for gamma in (0.01, 0.02):
        for seed in (1,2):
            G,fmax,conv = shear_modulus(360, 6.0, seed, gamma)
            Gs.append(G)
            print(f"{gamma:6.2f} {seed:5d} {G:10.5f} {fmax:8.3f} {str(conv):>6} {G*su:9.3f}")
    Gm=float(np.mean(Gs)); gsi=Gm*su
    print(f"\nmean G_reduced = {Gm:.5f}  ->  G_SI = {gsi:.3f} Pa")
    if gsi < 0.5:
        print(f"VERDICT: G_SI = {gsi:.2f} Pa — softer than the 1-100 Pa gel range")
        print("         (very sparse / sub-isostatic; raise fiber density + crosslinking)")
    elif gsi <= 200.0:
        print(f"VERDICT: G_SI = {gsi:.2f} Pa — order-of-magnitude consistent with a")
        print("         soft / dilute collagen gel (~1-100 Pa); calibration self-consistent.")
    else:
        print(f"VERDICT: G_SI = {gsi:.2f} Pa — stiffer than gel range; revisit anchors")
    print("NOTE: residual fmax above ftol (clamped/open-side BCs) -> order-of-magnitude")
    print("      estimate; periodic (Lees-Edwards) shear would tighten it.")

if __name__=="__main__":
    main()
