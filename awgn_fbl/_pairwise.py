"""
Mathematical primitives shared by the converse and achievability modules.

This file groups the low-level building blocks that are reused throughout
the library:

* ``pairwise_error_prob`` — Lemma 1 integral `P(ρ̂ ≥ t)`, linear evaluation
  via ``scipy.integrate.quad``.
* ``log_pairwise_error_prob`` — the same quantity in log domain, evaluated
  via ``logsumexp`` on a power-transformed θ-grid.  Stays accurate for large
  n and t close to 1 where the linear form underflows to zero.
* ``log_pairwise_error_prob_vec`` — vectorised log-domain Lemma 1 over an
  array of t's.  Useful for Monte-Carlo workloads.
* ``log_nct_cdf`` / ``log_nct_sf`` — log-domain non-central t-distribution
  CDF and survival function via the integral representation
  ``E_X[Φ(x · √(X/df) − nc)]`` over the central χ²(df) distribution,
  evaluated on a log-uniform X-grid.

The only scipy primitives assumed to be log-stable are ``norm.logcdf`` and
``norm.logsf`` (backed by ``special.log_ndtr``); everything else is wrapped
through an explicit log-sum-exp.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, special, stats


__all__ = [
    "pairwise_error_prob",
    "log_pairwise_error_prob",
    "log_pairwise_error_prob_vec",
    "log_nct_cdf",
    "log_nct_sf",
]


# ---------------------------------------------------------------------------
# Lemma 1 — linear evaluation
# ---------------------------------------------------------------------------

def pairwise_error_prob(t: float, n: int) -> float:
    """Pairwise error probability `P(ρ̂ ≥ t)` at blocklength n.

    Linear-domain evaluation via ``scipy.integrate.quad``.  For moderate n
    (say n ≲ 500) this agrees with :func:`log_pairwise_error_prob` to the
    integrator's precision; for larger n the linear form eventually
    underflows and the log version should be used instead.

    Parameters
    ----------
    t : float
        Correlation threshold, in (0, 1).
    n : int
        Blocklength.
    """
    if t <= 0:
        return 0.5
    if t >= 1:
        return 0.0

    def integrand(theta):
        return (1 + t ** 2 * np.tan(theta) ** 2) ** ((1 - n) / 2)

    integral_val, _ = integrate.quad(integrand, 0, np.pi / 2, limit=100)
    return (1 / np.pi) * (1 - t ** 2) ** ((n - 1) / 2) * integral_val


# ---------------------------------------------------------------------------
# Lemma 1 — log-domain evaluation (scalar and vector)
# ---------------------------------------------------------------------------

def _lemma1_grid(n_grid: int):
    """θ-grid + Jacobian factors shared between scalar and vector log forms."""
    p = 1.5  # concentration exponent; 1.5 works for n up to several thousand
    s = (np.arange(1, n_grid + 1) - 0.5) / n_grid          # midpoints in (0, 1)
    thetas = (np.pi / 2) * s ** p
    d_theta_d_s = (np.pi / 2) * p * s ** (p - 1)
    ds = 1.0 / n_grid
    return thetas, d_theta_d_s, ds


def log_pairwise_error_prob(t: float, n: int, n_grid: int = 600) -> float:
    """Log-domain pairwise error `log P(ρ̂ ≥ t)`.

    Uses the identity
    ``log P(t) = -log π + (n-1)/2 · log(1-t²)
                + log ∫_0^(π/2) exp[(1-n)/2 · log(1 + t² tan²θ)] dθ``
    evaluated with ``scipy.special.logsumexp`` on a θ-grid concentrated near
    θ = 0 where the integrand is peaked.  Remains accurate for large n and t
    close to 1 where the linear form underflows.
    """
    if t <= 0:
        return np.log(0.5)
    if t >= 1:
        return -np.inf

    log_prefactor = (n - 1) / 2 * np.log1p(-t * t)
    thetas, d_theta_d_s, ds = _lemma1_grid(n_grid)
    log_h = (1 - n) / 2 * np.log1p(t * t * np.tan(thetas) ** 2)
    log_integral = special.logsumexp(log_h + np.log(d_theta_d_s)) + np.log(ds)
    return -np.log(np.pi) + log_prefactor + log_integral


def log_pairwise_error_prob_vec(ts, n: int, n_grid: int = 600) -> np.ndarray:
    """Vectorised :func:`log_pairwise_error_prob` over an array of t values.

    Parameters
    ----------
    ts : array_like
        Correlation thresholds.  Shape ``(...,)``; the returned array has
        the same shape.
    n : int
        Blocklength.
    """
    ts = np.asarray(ts, dtype=float)
    thetas, d_theta_d_s, ds = _lemma1_grid(n_grid)
    log_weight = np.log(d_theta_d_s) + np.log(ds)

    safe_ts = np.clip(ts, 1e-300, 1 - 1e-15)
    t2 = (safe_ts ** 2)[..., None]
    log_h = (1 - n) / 2 * np.log1p(t2 * np.tan(thetas) ** 2)
    log_integral = special.logsumexp(log_h + log_weight, axis=-1)
    out = -np.log(np.pi) + (n - 1) / 2 * np.log1p(-safe_ts ** 2) + log_integral

    out = np.where(ts <= 0.0, np.log(0.5), out)
    out = np.where(ts >= 1.0, -np.inf, out)
    return out


# ---------------------------------------------------------------------------
# Log-domain non-central t CDF / SF
# ---------------------------------------------------------------------------

def _log_nct_template(x: float, df: float, nc: float, n_grid: int,
                      *, phi_fn):
    """Shared implementation for log_nct_cdf / log_nct_sf."""
    if df <= 0:
        raise ValueError("df must be positive")
    if n_grid < 50:
        raise ValueError("n_grid too small for a stable estimate")

    u_min = max(np.finfo(float).tiny, 1e-10 * df)
    u_max = 50.0 * df
    log_us = np.linspace(np.log(u_min), np.log(u_max), n_grid)
    us = np.exp(log_us)
    d_log_u = log_us[1] - log_us[0]

    log_f = stats.chi2.logpdf(us, df=df)
    args = x * np.sqrt(us / df) - nc
    log_phi = phi_fn(args)
    log_terms = log_f + log_phi + log_us + np.log(d_log_u)
    return float(special.logsumexp(log_terms))


def log_nct_cdf(x: float, df: float, nc: float, n_grid: int = 2000) -> float:
    """Log-domain non-central t CDF `log F_NCT(x; df, nc)`.

    Uses the integral representation
    ``F_NCT(x | df, nc) = E_X[Φ(x · √(X/df) − nc)]`` where X ~ χ²(df),
    evaluated on a log-uniform X-grid.  Log-spacing is essential for extreme
    negative x with non-zero nc, where the integrand is peaked in a region of
    tiny χ² density that quantile-midpoint quadrature undersamples.

    scipy's own ``stats.nct.logcdf`` is implemented as ``log(cdf(x))`` and so
    inherits the underflow of the linear ncdf; this routine genuinely stays
    in log domain throughout.
    """
    return _log_nct_template(x, df, nc, n_grid, phi_fn=stats.norm.logcdf)


def log_nct_sf(x: float, df: float, nc: float, n_grid: int = 2000) -> float:
    """Log-domain non-central t survival function `log (1 − F_NCT(x; df, nc))`."""
    return _log_nct_template(x, df, nc, n_grid, phi_fn=stats.norm.logsf)
