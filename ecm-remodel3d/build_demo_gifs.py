#!/usr/bin/env python3
"""Mobile-viewable GIFs for demos A (rheology/solver), B (KS/EMD), C (MCMC).

Each animates the REAL result/data so it plays inline on mobile (no HTML to open):
  A  FIRE-vs-Newton force-balance residual + force-balanced stress sigma(gamma)
     across the shear sweep (rheology_periodic.py data).
  B  model pore distribution sweeping its median against the digitized reference,
     with the binned KS D / p / EMD (ecm_pipeline.dist_stats) updating live.
  C  affine-invariant ensemble MCMC (ecm_pipeline.inference) converging on the
     buckle_ratio posterior from the real bridge response curve.

    python build_demo_gifs.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ecm_pipeline import dist_stats as ds
from ecm_pipeline import inference as inf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "output_validation", "figures")

# --- real data ---
RHEO_G = np.array([0.02, 0.05, 0.08, 0.11, 0.13, 0.15, 0.16, 0.17, 0.18, 0.20])
RHEO_FIRE = np.array([0.095, 11.951, 57.71, 218.327, 203.542, 409.568, 296.691, 359.744, 593.028, 14115.4])
RHEO_NEWT = np.array([0.024, 0.011, 0.019, 0.024, 0.023, 0.024, 225.691, 915.404, 323.439, 574.261])
RHEO_SIG = np.array([0.00001, 0.00003, 0.00004, 0.00007, 0.00009, 0.00012, 0.2524, 0.39461, 0.31883, 0.26126])
SU, FTOL, GC = 155.8, 0.025, 0.15
BR = np.array([0.005, 0.0354, 0.0658, 0.0962, 0.1265, 0.1569, 0.1873, 0.2177, 0.2481, 0.2785, 0.3088, 0.3392, 0.3696, 0.4])
BV = np.array([0.2495, 0.2175, 0.1938, 0.1783, 0.1713, 0.1645, 0.1605, 0.1575, 0.1527, 0.1483, 0.1466, 0.1495, 0.1482, 0.1332])


def gif_rheology():
    out = os.path.join(FIG, "rheology_demo.gif")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8)); fig.patch.set_facecolor("white")
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.26)
    frames = list(range(len(RHEO_G))) + [len(RHEO_G) - 1] * 4

    def draw(fk):
        i = min(fk, len(RHEO_G) - 1)
        a0, a1 = ax; a0.clear(); a1.clear()
        a0.set_yscale("log"); a0.set_xlim(0, 0.21); a0.set_ylim(5e-3, 3e4)
        a0.axhline(FTOL, color="#cbd5e1", ls="--", lw=1); a0.text(0.005, FTOL*1.3, "ftol", color="#94a3b8", fontsize=9)
        a0.plot(RHEO_G[:i+1], np.maximum(RHEO_FIRE[:i+1], 5e-3), "s-", color="#94a3b8", label="FIRE")
        a0.plot(RHEO_G[:i+1], np.maximum(RHEO_NEWT[:i+1], 5e-3), "o-", color="#2563eb", label="Newton")
        a0.axvline(RHEO_G[i], color="#db2777", lw=1.2)
        a0.set_xlabel("shear strain γ"); a0.set_ylabel("residual max force (force balance)")
        a0.set_title("FIRE stalls · Newton reaches force balance"); a0.legend(loc="upper left", fontsize=9)
        a1.set_xlim(0, 0.21); a1.set_ylim(1e-3, 1e2); a1.set_yscale("log")
        a1.axvline(GC, color="#c0392b", ls=":", lw=1.4); a1.text(GC+0.003, 2e-3, "rigidity\ntransition", color="#c0392b", fontsize=8)
        a1.plot(RHEO_G[:i+1], np.maximum(RHEO_SIG[:i+1]*SU, 1e-3), "o-", color="#059669")
        a1.set_xlabel("shear strain γ"); a1.set_ylabel("shear stress σ (Pa)")
        a1.set_title("force-balanced stress σ(γ)")
        fb = RHEO_FIRE[i] < FTOL; nb = RHEO_NEWT[i] < FTOL
        fig.suptitle(f"γ={RHEO_G[i]:.2f}   FIRE balanced: {'yes' if fb else 'NO'}   "
                     f"Newton balanced: {'yes' if nb else 'NO'}", fontsize=12, y=0.05)
        return []
    FuncAnimation(fig, draw, frames=len(frames), interval=600, blit=False).save(out, writer=PillowWriter(fps=2))
    plt.close(fig); print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")


def gif_ks():
    out = os.path.join(FIG, "ks_demo.gif")
    centers, counts, lo, hi = ds.load_histogram_csv(os.path.join(HERE, "datasets", "collagen_pore_um.csv"))
    meds = np.concatenate([np.linspace(12, 40, 22), np.linspace(40, 12, 14)])
    sig = 0.55
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); fig.patch.set_facecolor("white")
    xs = np.linspace(0, 85, 300)

    def draw(k):
        med = meds[k]; ax.clear()
        rng = np.random.default_rng(0)
        samp = med * np.exp(sig * rng.standard_normal(4000))
        D, p = ds.ks_sample_vs_binned(samp, lo, hi, counts)
        W = ds.emd_sample_vs_binned(samp, lo, hi, counts)
        tot = counts.sum()
        for k2 in range(len(counts)):
            d = counts[k2]/(tot*(hi[k2]-lo[k2]))
            ax.bar(lo[k2], d, width=hi[k2]-lo[k2], align="edge", color="#cbd5e1", edgecolor="white", linewidth=0.5)
        mden = np.exp(-(np.log(xs/med))**2/(2*sig*sig))/(xs*sig*np.sqrt(2*np.pi))
        mden[~np.isfinite(mden)] = 0
        ax.plot(xs, mden, color="#059669", lw=2.5, label=f"model (median {med:.0f} µm)")
        ax.set_xlim(0, 85); ax.set_ylim(0, 0.06); ax.set_xlabel("pore size (µm)"); ax.set_ylabel("density")
        verdict = "consistent" if p > 0.05 else "DISTINGUISHABLE"
        ax.set_title(f"GOF vs digitized reference (453 pores)\n"
                     f"KS D={D:.3f}  ·  p={'<1e-4' if p<1e-4 else f'{p:.3f}'}  ·  EMD={W:.1f} µm  →  {verdict}",
                     fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.text(2, 0.056, "grey bars = real digitized histogram", color="#64748b", fontsize=9)
        return []
    FuncAnimation(fig, draw, frames=len(meds), interval=150, blit=False).save(out, writer=PillowWriter(fps=7))
    plt.close(fig); print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")


def gif_mcmc():
    out = os.path.join(FIG, "mcmc_demo.gif")
    data, sigobs = 0.239, 0.025          # real calibration datum
    lo_b, hi_b = 0.005, 0.40

    def logp(th):
        br = th[0]
        if br < lo_b or br > hi_b:
            return -np.inf
        return -0.5 * ((np.interp(br, BR, BV) - data) / sigobs) ** 2

    K, steps = 30, 260
    p0 = np.random.default_rng(1).uniform(lo_b, hi_b, size=(K, 1))
    chain, acc = inf.sample_ensemble(logp, p0, steps, seed=2)
    burn = 80
    postsamp = chain[burn:, :, 0].ravel()
    mean, sd = postsamp.mean(), postsamp.std()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6)); fig.patch.set_facecolor("white")
    fig.subplots_adjust(bottom=0.2, top=0.88, wspace=0.26)
    show = list(range(0, steps, 6))

    def draw(fk):
        t = show[fk]; a0, a1 = ax; a0.clear(); a1.clear()
        # walkers exploring (x=buckle_ratio, y=walker)
        a0.scatter(chain[t, :, 0], np.arange(K), s=24, c="#7c3aed", alpha=0.8)
        a0.axvline(0.02, color="#c0392b", ls="--", lw=1.4); a0.text(0.025, K*0.9, "true 0.02", color="#c0392b", fontsize=9)
        a0.set_xlim(0, 0.4); a0.set_ylim(-1, K); a0.set_xlabel("buckle_ratio"); a0.set_ylabel("walker")
        a0.set_title(f"ensemble MCMC walkers · step {t}")
        # posterior so far (post-burn up to t)
        if t > burn:
            s = chain[burn:t+1, :, 0].ravel()
            a1.hist(s, bins=np.linspace(0, 0.4, 40), density=True, color="#7c3aed", alpha=0.75)
        a1.axvline(0.02, color="#c0392b", ls="--", lw=1.4)
        a1.set_xlim(0, 0.4); a1.set_xlabel("buckle_ratio"); a1.set_ylabel("posterior density")
        a1.set_title(f"posterior  ({mean:.3f} ± {sd:.3f})")
        fig.suptitle(f"Bayesian inference: walkers converge → posterior (acceptance {acc:.2f})",
                     fontsize=12, y=0.04)
        return []
    FuncAnimation(fig, draw, frames=len(show), interval=120, blit=False).save(out, writer=PillowWriter(fps=8))
    plt.close(fig); print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    gif_rheology(); gif_ks(); gif_mcmc()
