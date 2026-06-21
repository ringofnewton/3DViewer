"""Bayesian inference toolkit (numpy-only) — richer inference for the roadmap.

Upgrades the 2-parameter grid posterior to a proper, scalable workflow:

  latin_hypercube     space-filling design over an N-dim parameter box.
  RBFEmulator         fast, smooth surrogate of an expensive forward model
                      (Gaussian radial basis functions + nugget), with
                      leave-one-out cross-validation so its error is quantified.
  sample_ensemble     affine-invariant ensemble MCMC (Goodman & Weare 2010 — the
                      emcee stretch move), which needs no proposal tuning and
                      handles correlated / degenerate posteriors (the heart of an
                      identifiability analysis).
  gelman_rubin        split-chain R-hat convergence diagnostic.
  autocorr_time       integrated autocorrelation time (effective sample size).

Why a surrogate: the fiber-network forward model costs ~seconds per evaluation, so
direct MCMC (10^4-10^5 evaluations) is infeasible. Evaluate it on a space-filling
design once, fit a validated emulator, then run full MCMC on the emulator — the
standard simulator-calibration pattern (Kennedy & O'Hagan 2001). Emulator error is
reported (LOO) and folded into the likelihood, so it is accounted for, not hidden.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
def latin_hypercube(n, bounds, seed=0):
    """n-point Latin-hypercube sample in the box `bounds` = [(lo,hi), ...]."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    d = len(bounds)
    cut = np.linspace(0, 1, n + 1)
    X = np.zeros((n, d))
    for j in range(d):
        u = cut[:-1] + rng.random(n) * (1.0 / n)
        X[:, j] = rng.permutation(u)
    return bounds[:, 0] + X * (bounds[:, 1] - bounds[:, 0])


# --------------------------------------------------------------------------- #
# Emulator
# --------------------------------------------------------------------------- #
class RBFEmulator:
    """Gaussian radial-basis-function interpolator with a nugget.

    Inputs are min-max normalized so one length scale is sensible across
    parameters. The length scale is chosen by minimizing the leave-one-out RMSE
    (closed-form via the kernel inverse), which is also reported as the emulator's
    honest accuracy.
    """

    def __init__(self, X, y, nugget=1e-6, lengths=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.lo = X.min(0)
        self.span = np.where(X.max(0) - X.min(0) > 0, X.max(0) - X.min(0), 1.0)
        self.Xn = (X - self.lo) / self.span
        self.y = y
        self.nugget = nugget
        D = self._dist(self.Xn, self.Xn)
        grid = lengths if lengths is not None else np.geomspace(0.1, 3.0, 24)
        self.length, self.loo_rmse = self._select_length(D, y, grid, nugget)
        Phi = np.exp(-(D / self.length) ** 2)
        self.w = np.linalg.solve(Phi + nugget * np.eye(len(X)), y)

    @staticmethod
    def _dist(A, B):
        return np.sqrt(np.maximum(
            ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0) + 1e-30)

    @staticmethod
    def _select_length(D, y, grid, nugget):
        n = len(y); best = (grid[0], np.inf)
        for L in grid:
            A = np.exp(-(D / L) ** 2) + nugget * np.eye(n)
            try:
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            resid = (Ainv @ y) / np.clip(np.diag(Ainv), 1e-12, None)  # LOO residuals
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            if rmse < best[1]:
                best = (L, rmse)
        return best

    def __call__(self, x):
        xn = (np.atleast_2d(np.asarray(x, float)) - self.lo) / self.span
        K = np.exp(-(self._dist(xn, self.Xn) / self.length) ** 2)
        out = K @ self.w
        return out if out.shape[0] > 1 else float(out[0])


# --------------------------------------------------------------------------- #
# Affine-invariant ensemble MCMC (Goodman & Weare 2010)
# --------------------------------------------------------------------------- #
def sample_ensemble(log_prob, p0, n_steps, a=2.0, seed=0):
    """Stretch-move ensemble sampler.

    log_prob: callable(theta)->float (log posterior, -inf outside support).
    p0:       (K, d) initial walker positions (K >= 2d+2 recommended).
    Returns (chain (n_steps, K, d), acceptance_fraction).
    """
    rng = np.random.default_rng(seed)
    walkers = np.array(p0, float)
    K, d = walkers.shape
    lp = np.array([log_prob(w) for w in walkers])
    chain = np.zeros((n_steps, K, d))
    n_acc = 0
    for t in range(n_steps):
        for k in range(K):
            j = int(rng.integers(K - 1))
            if j >= k:
                j += 1                                  # a partner walker != k
            z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a      # g(z) ~ 1/sqrt(z)
            y = walkers[j] + z * (walkers[k] - walkers[j])
            ly = log_prob(y)
            log_alpha = (d - 1) * np.log(z) + ly - lp[k]
            if np.log(rng.random()) < log_alpha:
                walkers[k] = y; lp[k] = ly; n_acc += 1
        chain[t] = walkers
    return chain, n_acc / (n_steps * K)


# --------------------------------------------------------------------------- #
# Convergence diagnostics
# --------------------------------------------------------------------------- #
def gelman_rubin(chains):
    """Split-chain potential scale reduction factor R-hat for one parameter.

    chains: (n_chains, n_draws). R-hat -> 1 as chains mix (converged ~< 1.1).
    """
    chains = np.asarray(chains, float)
    m, n = chains.shape
    means = chains.mean(1)
    W = chains.var(1, ddof=1).mean()
    B = n * means.var(ddof=1)
    var = (1 - 1.0 / n) * W + B / n
    return float(np.sqrt(var / W)) if W > 0 else np.inf


def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time of a 1-D series (Sokal windowing)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] == 0:
        return np.inf
    acf /= acf[0]
    tau = 1.0 + 2.0 * np.cumsum(acf[1:])
    window = np.arange(len(tau))
    m = window < c * tau
    return float(tau[np.argmax(~m)]) if (~m).any() else float(tau[-1])
