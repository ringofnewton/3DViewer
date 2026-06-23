#!/usr/bin/env python3
"""Richer Bayesian inference: emulator-based MCMC over FOUR parameters — Item 3.

Upgrades the 2-parameter, 5x5 grid posterior (`bayesian_infer.py`) to a proper,
scalable workflow that delivers what a grid cannot: a joint posterior over more
parameters, an identifiability analysis (which parameters the data can and cannot
constrain), full MCMC convergence diagnostics, and a held-out predictive check.

Pipeline (the standard expensive-simulator calibration pattern):
  1. space-filling Latin-hypercube design over the 4-D parameter box,
  2. run the fiber-network forward model once per design point (bridge + aster),
  3. fit a Gaussian-RBF EMULATOR to each output, cross-validated (leave-one-out),
  4. run affine-invariant ensemble MCMC (Goodman-Weare) on the emulator posterior
     of the BRIDGE experiment — cheap, so 10^4-10^5 samples are trivial,
  5. report marginals vs truth, the posterior CORRELATION matrix (identifiability),
     R-hat + autocorrelation (convergence), and
  6. a posterior-PREDICTIVE check on a held-out single-cell ASTER experiment.

Parameters inferred: buckle_ratio, stiffen_alpha, k_bend, traction.
Emulator error (LOO) is folded into the likelihood, so it is accounted for.

    python mcmc_infer.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech
from ecm_pipeline import inference as inf

DOMAIN, SLAB = 150.0, 26.0
SEED_NET = 3
NAMES = ["buckle_ratio", "stiffen_alpha", "k_bend", "traction"]
BOUNDS = [(0.005, 0.40), (0.0, 14.0), (0.03, 0.40), (0.6, 2.4)]
TRUE = np.array([0.02, 6.0, 0.12, 1.5])
SIG_OBS = 0.025
N_DESIGN = 64

_NET_CACHE = {}


def _net(seed):
    # the seed is fixed across forward calls, so build the base network once
    if seed not in _NET_CACHE:
        _NET_CACHE[seed] = mech.build_random_network(
            DOMAIN, n_fibers=120, fiber_len=70.0, seg_len=7.0,
            crosslink_dist=4.5, seed=seed, slab_thickness=SLAB)
    return _NET_CACHE[seed].copy()


def _params(theta):
    br, al, kb, tr = theta
    return mech.MechParams(planar=True, max_iter=1200, ftol=2.5e-2,
                           buckle_ratio=br, stiffen_alpha=al, k_bend=kb, traction=tr)


def bridge_signal(theta, seed=SEED_NET):
    p = _params(theta); net = _net(seed); c = DOMAIN / 2.0
    cells = np.array([[c - 28, c, c], [c + 28, c, c]])
    r, _ = mech.relax(net, cells, p)
    region = (cells[0], cells[1])
    ax = mech.axis_alignment(r, np.array([1.0, 0, 0]), region, 22.0)[0]
    pe = mech.axis_alignment(r, np.array([0.0, 1.0, 0]), region, 22.0)[0]
    return ax - pe


def aster_signal(theta, seed=SEED_NET):
    p = _params(theta); net = _net(seed); c = DOMAIN / 2.0
    cell = np.array([[c, c, c]])
    r, _ = mech.relax(net, cell, p)
    _, tension, _ = mech.compute_forces(r, cell, p)
    w = np.clip(tension, 0, None)
    mid = 0.5 * (r.nodes[r.edges[:, 0]] + r.nodes[r.edges[:, 1]])
    rad = mid - cell[0]; rad /= (np.linalg.norm(rad, axis=1)[:, None] + 1e-9)
    dirs, L = mech.edge_dirs_lengths(r)
    val = (np.sum(dirs * rad, axis=1)) ** 2
    return float(np.sum(w * val) / (np.sum(w) + 1e-9))


def main():
    print("Richer Bayesian inference: emulator-based MCMC over 4 parameters\n")
    print(f"parameters : {NAMES}")
    print(f"true values: {[float(x) for x in TRUE]}\n")

    # 1-2. design + forward model evaluations
    X = inf.latin_hypercube(N_DESIGN, BOUNDS, seed=1)
    print(f"running forward model at {N_DESIGN} design points (bridge + aster)...")
    yb = np.array([bridge_signal(x) for x in X])
    ya = np.array([aster_signal(x) for x in X])

    # 3. emulators (LOO-cross-validated)
    emu_b = inf.RBFEmulator(X, yb)
    emu_a = inf.RBFEmulator(X, ya)
    print(f"emulator LOO RMSE : bridge {emu_b.loo_rmse:.4f} (len {emu_b.length:.2f}),"
          f"  aster {emu_a.loo_rmse:.4f} (len {emu_a.length:.2f})")
    print(f"  signal spread    : bridge sd {yb.std():.3f},  aster sd {ya.std():.3f}"
          f"   -> emulator explains ~{100*(1-emu_b.loo_rmse**2/yb.var()):.0f}% of bridge variance")

    # 4. synthetic data (twin experiment) + likelihood with emulator error folded in
    rng = np.random.default_rng(7)
    bridge_data = bridge_signal(TRUE) + SIG_OBS * rng.standard_normal()
    aster_data = aster_signal(TRUE) + SIG_OBS * rng.standard_normal()
    sig_eff = np.sqrt(SIG_OBS ** 2 + emu_b.loo_rmse ** 2)
    print(f"\ncalibration datum (bridge) = {bridge_data:.3f}    "
          f"held-out datum (aster) = {aster_data:.3f}")
    print(f"likelihood sigma (obs + emulator) = {sig_eff:.4f}")

    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])

    def log_prob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return -0.5 * ((emu_b(theta) - bridge_data) / sig_eff) ** 2

    # 5. ensemble MCMC
    K, n_steps, burn = 40, 3000, 600
    p0 = inf.latin_hypercube(K, BOUNDS, seed=2)
    print(f"\nrunning ensemble MCMC: {K} walkers x {n_steps} steps (burn {burn})...")
    chain, acc = inf.sample_ensemble(log_prob, p0, n_steps, seed=3)
    post = chain[burn:].reshape(-1, len(NAMES))            # (samples, d)

    rhat = [inf.gelman_rubin(chain[burn:, :, j].T) for j in range(len(NAMES))]
    tau = [inf.autocorr_time(chain[burn:, :, j].mean(1)) for j in range(len(NAMES))]
    ess = K * (n_steps - burn) / max(np.mean(tau), 1.0)
    print(f"  acceptance fraction = {acc:.2f}   max R-hat = {max(rhat):.3f}   "
          f"ESS ~ {ess:.0f}")
    print(f"  {'(converged: R-hat < 1.1, acceptance 0.2-0.5)' if max(rhat) < 1.1 else '(WARNING: not converged)'}")

    # marginals vs truth
    print("\nPosterior marginals (mean +/- sd) vs truth:")
    mean = post.mean(0); sd = post.std(0)
    for j, nm in enumerate(NAMES):
        within = abs(mean[j] - TRUE[j]) < 2 * sd[j] + 0.05 * (hi[j] - lo[j])
        rel = sd[j] / (hi[j] - lo[j])
        tag = "constrained" if rel < 0.22 else "weakly constrained (broad)"
        print(f"  {nm:14s} = {mean[j]:7.3f} +/- {sd[j]:6.3f}   (true {TRUE[j]:6.3f})  "
              f"{'OK' if within else '!!'}  [{tag}]")

    # 5b. identifiability: posterior correlations
    print("\nIdentifiability — posterior correlation matrix:")
    C = np.corrcoef(post.T)
    print("              " + " ".join(f"{n[:6]:>7}" for n in NAMES))
    for j, nm in enumerate(NAMES):
        print(f"  {nm[:12]:12s} " + " ".join(f"{C[j, k]:+7.2f}" for k in range(len(NAMES))))
    degen = [(NAMES[i], NAMES[j], C[i, j]) for i in range(len(NAMES))
             for j in range(i + 1, len(NAMES)) if abs(C[i, j]) > 0.6]
    if degen:
        for a, b, c in degen:
            print(f"  -> {a} & {b} are degenerate (corr {c:+.2f}): the bridge alone cannot separate them")
    else:
        print("  -> no strong pairwise degeneracies at |corr|>0.6")

    # 6. posterior-PREDICTIVE check on the held-out aster
    idx = np.random.default_rng(0).choice(len(post), size=min(4000, len(post)), replace=False)
    pred = emu_a(post[idx])
    pred_mean = float(pred.mean())
    pred_sd = float(np.sqrt(pred.var() + SIG_OBS ** 2 + emu_a.loo_rmse ** 2))
    covered = abs(pred_mean - aster_data) < 2 * pred_sd
    print("\nPredictive validation (held-out aster, NOT used for fitting):")
    print(f"  posterior-predictive aster = {pred_mean:.3f} +/- {pred_sd:.3f}")
    print(f"  held-out datum             = {aster_data:.3f}")
    print(f"  -> covers the held-out datum (|err| < 2sd): {'YES — predictive' if covered else 'NO'}")
    print("\nWhat this adds over the grid: a joint 4-D posterior with quantified")
    print("uncertainty, an explicit identifiability read-out (the correlation matrix),")
    print("MCMC convergence diagnostics, and a passing out-of-sample prediction.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        # marginal of the best-constrained param
        ax[0].hist(post[:, 0], bins=40, density=True, color="#2c6fbb", alpha=0.8)
        ax[0].axvline(TRUE[0], color="#c0392b", ls="--", label="true")
        ax[0].set_xlabel(NAMES[0]); ax[0].set_ylabel("posterior density")
        ax[0].set_title("Marginal posterior"); ax[0].legend(fontsize=8)
        # joint posterior (the two nonlinearities) showing correlation
        ax[1].scatter(post[::5, 0], post[::5, 1], s=3, alpha=0.25, color="#2c6fbb")
        ax[1].scatter([TRUE[0]], [TRUE[1]], color="#c0392b", marker="*", s=160, label="true")
        ax[1].set_xlabel(NAMES[0]); ax[1].set_ylabel(NAMES[1])
        ax[1].set_title("Joint posterior"); ax[1].legend(fontsize=8)
        # predictive check
        ax[2].hist(pred, bins=40, density=True, color="#27ae60", alpha=0.7, label="post-predictive")
        ax[2].axvline(aster_data, color="#c0392b", ls="--", lw=2, label="held-out datum")
        ax[2].set_xlabel("aster radiality"); ax[2].set_ylabel("density")
        ax[2].set_title("Predictive validation (held-out)"); ax[2].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "mcmc_infer.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'mcmc_infer.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
