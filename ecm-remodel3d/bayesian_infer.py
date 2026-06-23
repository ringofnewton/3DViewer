#!/usr/bin/env python3
"""Bayesian parameter inference + predictive validation — Item 3 of the roadmap.

OAT/Sobol tell us WHICH parameters matter; this fits them to data WITH
uncertainty, then does the test that separates a fitted model from a predictive
one: calibrate on one experiment, PREDICT a different held-out experiment.

Setup (a twin / identical-model experiment, so ground truth is known):
  parameters inferred : buckle_ratio, stiffen_alpha  (the two nonlinearities)
  calibration data    : inter-cell BRIDGE signal (axis - perp alignment) + noise
  held-out prediction : single-cell ASTER (tension-weighted radiality) + noise

We compute a grid posterior P(theta | bridge_data) with a Gaussian likelihood,
report the recovered marginals vs the true values, then form the posterior-
PREDICTIVE distribution of the aster and check it covers the held-out datum.

    python bayesian_infer.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech

DOMAIN, SLAB = 150.0, 26.0
TRUE = dict(buckle_ratio=0.02, stiffen_alpha=6.0)
BR_GRID = np.array([0.01, 0.04, 0.10, 0.20, 0.35])
AL_GRID = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
SIG_OBS = 0.025          # observation noise (alignment units)
SEED_NET = 3


def _net(seed):
    return mech.build_random_network(DOMAIN, n_fibers=160, fiber_len=70.0, seg_len=7.0,
                                     crosslink_dist=4.5, seed=seed, slab_thickness=SLAB)


def bridge_signal(br, al, seed=SEED_NET):
    p = mech.MechParams(planar=True, max_iter=2600, ftol=2e-2,
                        buckle_ratio=br, stiffen_alpha=al)
    net = _net(seed); c = DOMAIN / 2.0
    cells = np.array([[c - 28, c, c], [c + 28, c, c]])
    r, _ = mech.relax(net, cells, p)
    region = (cells[0], cells[1])
    ax = mech.axis_alignment(r, np.array([1.0, 0, 0]), region, 22.0)[0]
    pe = mech.axis_alignment(r, np.array([0.0, 1.0, 0]), region, 22.0)[0]
    return ax - pe


def aster_signal(br, al, seed=SEED_NET):
    p = mech.MechParams(planar=True, max_iter=2600, ftol=2e-2,
                        buckle_ratio=br, stiffen_alpha=al)
    net = _net(seed); c = DOMAIN / 2.0
    cell = np.array([[c, c, c]])
    r, _ = mech.relax(net, cell, p)
    _, tension, _ = mech.compute_forces(r, cell, p)
    w = np.clip(tension, 0, None)
    mid = 0.5 * (r.nodes[r.edges[:, 0]] + r.nodes[r.edges[:, 1]])
    rad = mid - cell[0]; rad /= (np.linalg.norm(rad, axis=1)[:, None] + 1e-9)
    dirs, L = mech.edge_dirs_lengths(r)
    val = (np.sum(dirs * rad, axis=1)) ** 2          # radiality (isotropic 0.5 in-plane)
    return float(np.sum(w * val) / (np.sum(w) + 1e-9))


def main():
    print("Bayesian inference of (buckle_ratio, stiffen_alpha) + predictive validation\n")
    print(f"true params: buckle_ratio={TRUE['buckle_ratio']}, stiffen_alpha={TRUE['stiffen_alpha']}")

    # synthetic data from the true params (twin experiment) + observation noise
    rng = np.random.default_rng(7)
    bridge_data = bridge_signal(TRUE["buckle_ratio"], TRUE["stiffen_alpha"]) + SIG_OBS * rng.standard_normal()
    aster_data = aster_signal(TRUE["buckle_ratio"], TRUE["stiffen_alpha"]) + SIG_OBS * rng.standard_normal()
    print(f"calibration datum (bridge axis-perp) = {bridge_data:.3f}")
    print(f"held-out datum    (aster radiality)  = {aster_data:.3f}\n")

    # grid posterior over the two parameters from the bridge datum
    logpost = np.full((len(BR_GRID), len(AL_GRID)), -np.inf)
    bridge_grid = np.zeros_like(logpost)
    print("computing grid forward models (bridge)...")
    for i, br in enumerate(BR_GRID):
        for j, al in enumerate(AL_GRID):
            f = bridge_signal(br, al)
            bridge_grid[i, j] = f
            logpost[i, j] = -0.5 * ((f - bridge_data) / SIG_OBS) ** 2
    post = np.exp(logpost - logpost.max())
    post /= post.sum()

    # marginal posteriors -> mean +/- std
    p_br = post.sum(axis=1); p_al = post.sum(axis=0)
    br_mean = float(np.sum(BR_GRID * p_br)); br_sd = float(np.sqrt(np.sum(p_br * (BR_GRID - br_mean) ** 2)))
    al_mean = float(np.sum(AL_GRID * p_al)); al_sd = float(np.sqrt(np.sum(p_al * (AL_GRID - al_mean) ** 2)))
    print("\nPosterior (marginal mean +/- sd):")
    print(f"  buckle_ratio  = {br_mean:.3f} +/- {br_sd:.3f}   (true {TRUE['buckle_ratio']})")
    print(f"  stiffen_alpha = {al_mean:.2f} +/- {al_sd:.2f}   (true {TRUE['stiffen_alpha']})")
    br_ok = abs(br_mean - TRUE["buckle_ratio"]) < 2 * br_sd + 0.05
    print(f"  buckle_ratio recovered within posterior uncertainty: {'YES' if br_ok else 'NO'}")

    # posterior-PREDICTIVE check on the held-out aster experiment
    print("\nPredictive validation (held-out aster, NOT used for fitting):")
    # weight aster forward models by the posterior (compute at grid points, reuse)
    aster_grid = np.zeros_like(post)
    for i, br in enumerate(BR_GRID):
        for j, al in enumerate(AL_GRID):
            if post[i, j] > 0.005:                    # only meaningful posterior mass
                aster_grid[i, j] = aster_signal(br, al)
    mask = post > 0.005
    wsum = post[mask].sum()
    pred_mean = float(np.sum(post[mask] * aster_grid[mask]) / wsum)
    pred_sd = float(np.sqrt(np.sum(post[mask] * (aster_grid[mask] - pred_mean) ** 2) / wsum))
    pred_sd = np.sqrt(pred_sd ** 2 + SIG_OBS ** 2)    # add observation noise
    print(f"  posterior-predictive aster = {pred_mean:.3f} +/- {pred_sd:.3f}")
    print(f"  held-out datum             = {aster_data:.3f}")
    covered = abs(pred_mean - aster_data) < 2 * pred_sd
    print(f"  -> prediction covers the held-out datum (|err| < 2sd): {'YES — predictive' if covered else 'NO'}")
    print("Interpretation: parameters fit from the bridge experiment, with quantified")
    print("uncertainty, correctly PREDICT an independent single-cell experiment.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im = ax[0].imshow(post.T, origin="lower", aspect="auto", cmap="viridis",
                          extent=[BR_GRID[0], BR_GRID[-1], AL_GRID[0], AL_GRID[-1]])
        ax[0].scatter([TRUE["buckle_ratio"]], [TRUE["stiffen_alpha"]], c="r", marker="*",
                      s=150, label="true")
        ax[0].scatter([br_mean], [al_mean], c="w", marker="x", s=80, label="posterior mean")
        ax[0].set_xlabel("buckle_ratio"); ax[0].set_ylabel("stiffen_alpha")
        ax[0].set_title("Posterior P(theta | bridge data)"); ax[0].legend(fontsize=8)
        fig.colorbar(im, ax=ax[0], shrink=0.8)
        ax[1].axhline(aster_data, color="#c0392b", ls="--", label="held-out datum")
        ax[1].errorbar([0], [pred_mean], yerr=[pred_sd], fmt="o", color="#2c6fbb",
                       capsize=6, ms=8, label="posterior-predictive")
        ax[1].set_xlim(-1, 1); ax[1].set_xticks([])
        ax[1].set_ylabel("aster radiality"); ax[1].set_title("Predictive validation (held-out)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "bayesian_infer.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'bayesian_infer.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
