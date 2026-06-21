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


# --------------------------------------------------------------------------- #
# Digitized-histogram support
#
# A digitized experimental distribution (e.g. WebPlotDigitizer output of a
# published histogram) is a list of bins with a COUNT each — there is no raw
# per-object sample. These compare a model SAMPLE directly against such a binned
# reference (weighted ECDF), so a real digitized CSV is a drop-in replacement for
# a synthetic reference. The reference's total count sets the KS power.
# --------------------------------------------------------------------------- #
def load_histogram_csv(path):
    """Load a digitized histogram: columns bin_low,bin_high,count.

    Skips '#' comments and a non-numeric header row. Returns (centers, counts,
    lo, hi).
    """
    data = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data.append([float(t) for t in line.split(",")[:3]])
            except ValueError:
                continue                       # header / non-numeric row
    rows = np.atleast_2d(np.array(data, float))
    lo, hi, cnt = rows[:, 0], rows[:, 1], rows[:, 2]
    return 0.5 * (lo + hi), cnt, lo, hi


def sample_from_histogram(lo, hi, count, n, seed=0):
    """Draw n samples uniformly within bins with probability ~ count (for plots)."""
    rng = np.random.default_rng(seed)
    k = rng.choice(len(count), size=n, p=count / count.sum())
    return lo[k] + rng.random(n) * (hi[k] - lo[k])


def _binned_cdf(x_eval, lo, hi, counts):
    """Piecewise-linear reference CDF of a binned histogram.

    A digitized histogram gives a COUNT per interval [lo, hi); the maximum-entropy
    reconstruction is uniform within each bin, so the CDF ramps linearly across the
    bin (not a step at the centre). Using this removes a spurious ~half-bin KS
    discrepancy that point-at-centre would introduce.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    counts = np.asarray(counts, float); tot = counts.sum()
    cum_before = np.concatenate([[0.0], np.cumsum(counts)])[:-1]
    x = np.asarray(x_eval, float)
    F = np.zeros_like(x)
    for k in range(len(counts)):
        m = (x >= lo[k]) & (x < hi[k])
        F[m] = (cum_before[k] + counts[k] * (x[m] - lo[k]) / (hi[k] - lo[k])) / tot
    F[x >= hi[-1]] = 1.0
    return F


def ks_sample_vs_binned(a, lo, hi, counts):
    """KS between a raw sample `a` and a digitized histogram (lo, hi, counts).

    Compares the sample ECDF to the bin's piecewise-linear CDF. The reference's
    total count is its effective n, so a higher-count digitized histogram yields a
    more powerful (smaller-p) test — as it should.
    """
    a = np.sort(np.asarray(a, float))
    n1 = len(a)
    n2 = float(np.sum(counts))
    pts = np.concatenate([a, np.asarray(lo, float), np.asarray(hi, float)])
    Fn_r = np.searchsorted(a, pts, side="right") / n1
    Fn_l = np.searchsorted(a, pts, side="left") / n1
    G = _binned_cdf(pts, lo, hi, counts)
    D = float(max(np.max(np.abs(Fn_r - G)), np.max(np.abs(Fn_l - G))))
    ne = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D
    return D, _ks_q(lam)


def emd_sample_vs_binned(a, lo, hi, counts, ngrid=3000):
    """Wasserstein-1 (EMD) between a raw sample and a digitized histogram."""
    a = np.sort(np.asarray(a, float))
    xs = np.linspace(min(a[0], lo[0]), max(a[-1], hi[-1]), ngrid)
    Fn = np.searchsorted(a, xs, side="right") / len(a)
    G = _binned_cdf(xs, lo, hi, counts)
    return float(np.trapezoid(np.abs(Fn - G), xs))
