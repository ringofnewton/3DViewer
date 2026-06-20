"""Distribution-level comparison statistics (numpy-only) — Step "Item 2".

Tools to compare a model distribution to a reference at the DISTRIBUTION level,
beyond matching a mean or a range:

  ks_2samp        two-sample Kolmogorov-Smirnov statistic D and asymptotic p-value
  wasserstein1    1-D Earth Mover's Distance (Wasserstein-1), in the data's units
  lognormal_sample  draw a reference sample from a log-normal given median & sigma

No SciPy: the KS p-value uses the standard asymptotic Kolmogorov series
    Q_KS(lambda) = 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 lambda^2)
with the Stephens small-sample correction on the effective n.
"""

from __future__ import annotations

import numpy as np


def _ks_q(lam):
    """Kolmogorov survival function Q_KS(lambda) = P(D > lambda)."""
    if lam < 1e-3:
        return 1.0
    k = np.arange(1, 101)
    terms = 2.0 * ((-1) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * lam ** 2)
    return float(min(max(np.sum(terms), 0.0), 1.0))


def ks_2samp(a, b):
    """Two-sample KS: max |F_a - F_b| over the pooled support, with p-value."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    n1, n2 = len(a), len(b)
    allx = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allx, side="right") / n1
    cdf_b = np.searchsorted(b, allx, side="right") / n2
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    ne = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D
    return D, _ks_q(lam)


def wasserstein1(a, b):
    """1-D Wasserstein-1 (EMD) = integral |F_a - F_b| dx (same units as data)."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    allx = np.sort(np.concatenate([a, b]))
    dx = np.diff(allx)
    cdf_a = np.searchsorted(a, allx[:-1], side="right") / len(a)
    cdf_b = np.searchsorted(b, allx[:-1], side="right") / len(b)
    return float(np.sum(np.abs(cdf_a - cdf_b) * dx))


def lognormal_sample(median, sigma, n, seed=0):
    """Sample a log-normal with the given MEDIAN and log-space sigma."""
    rng = np.random.default_rng(seed)
    return median * np.exp(sigma * rng.standard_normal(n))
