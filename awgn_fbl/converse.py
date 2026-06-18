"""
Converse (upper) bounds on the maximum coding rate for the AWGN channel.

Three methods with complementary properties:

* :class:`NoncentralTConverse` — our method, based on the non-central t
  distribution.  Offers two evaluation paths:
    - a *simple* linear-domain path via scipy's ``stats.nct``, fast and
      accurate for moderate n but hitting scipy's NaN wall at n≳1000 and
      high SNR;
    - a *log-domain* path via :func:`awgn_fbl._pairwise.log_nct_cdf`, which
      extends the working range essentially indefinitely and serves as the
      default when numerical robustness matters.
  The two paths agree to ~10⁻⁶ in the shared regime (a cross-check wired
  into the test suite).

* :class:`ChiSquaredConverse` — Polyanskiy's meta-converse evaluated with
  the auxiliary output measure ``Q_Y = N(0, (1+P)·I)``.  Strictly looser
  than the NCT converse at any finite n; kept as a reference because it is
  the standard formulation in the literature and lets us quantify the
  relaxation cost.

* :class:`SolidAngleConverse` — Shannon's 1959 original formulation via the
  solid angle ratio ``Ω_N(θ)/Ω_N(π)``.  Mathematically equivalent to
  :class:`NoncentralTConverse` but numerically fragile for n > 80 (warning
  emitted beyond that).  Kept for pedagogy and as a historical cross-check.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import integrate, optimize, stats

from ._pairwise import (
    log_nct_cdf,
    log_pairwise_error_prob,
    pairwise_error_prob,
)


__all__ = [
    "AWGNConverseBase",
    "NoncentralTConverse",
    "ChiSquaredConverse",
    "SolidAngleConverse",
    "awgn_converse_rate",
    "awgn_converse_error",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AWGNConverseBase:
    """Shared state / helpers for AWGN converse bound classes.

    Parameters
    ----------
    n : int
        Blocklength (number of channel uses).
    snr_db : float
        Signal-to-noise ratio in dB.  The linear SNR ``P = 10^(snr_db/10)``
        is available as ``self.snr``.
    """

    def __init__(self, n: int, snr_db: float):
        if n < 1:
            raise ValueError("Block length n must be positive")
        self.n = n
        self.snr_db = snr_db
        self.snr = 10 ** (snr_db / 10)

    def shannon_capacity(self) -> float:
        """Shannon capacity in bits/channel use."""
        return 0.5 * np.log2(1 + self.snr)

    def converse_rate(self, epsilon: float) -> float:
        """Upper bound on rate given target error probability ε."""
        raise NotImplementedError

    # -- Lemma 1 helpers (used by NCT and SolidAngle subclasses) ----------

    def _pairwise_error_prob(self, t: float) -> float:
        """Linear-domain pairwise error `P(ρ̂ ≥ t)` at blocklength ``n``."""
        return pairwise_error_prob(t, self.n)

    def log_pairwise_error_prob(self, t: float, n_grid: int = 600) -> float:
        """Log-domain pairwise error `log P(ρ̂ ≥ t)` at blocklength ``n``."""
        return log_pairwise_error_prob(t, self.n, n_grid=n_grid)

    def log_pairwise_error_prob_vec(self, ts, n_grid: int = 600):
        """Vectorised version of :meth:`log_pairwise_error_prob`."""
        from ._pairwise import log_pairwise_error_prob_vec as _vec
        return _vec(ts, self.n, n_grid=n_grid)


# ---------------------------------------------------------------------------
# Non-central t converse
# ---------------------------------------------------------------------------

class NoncentralTConverse(AWGNConverseBase):
    """Our NCT-based converse bound.

    Exposes two evaluation paths.  The *simple* ``converse_rate`` /
    ``converse_error`` use scipy's ``stats.nct`` directly in linear domain;
    fast and fine for moderate n, but NaN at large n / high SNR.  The
    *log-domain* ``converse_rate_log`` / ``converse_error_log`` use our
    own log-domain Lemma 1 and log-NCT CDF (see
    :mod:`awgn_fbl._pairwise`); they are the recommended defaults when
    numerical robustness matters.

    The two paths agree to ~10⁻⁶ bits/use in their shared regime, a
    cross-check built into the test suite.
    """

    # =====================================================================
    # Simple (linear) path
    # =====================================================================

    def converse_rate(self, epsilon: float) -> float:
        """Compute converse bound `ε → R` (linear-domain reference path).

        Flow: ε → NCT threshold (via ``nct.ppf``) → β → t → P(t) via linear
        Lemma 1 → R = -log₂ P(t) / n.
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        n = self.n
        mu = np.sqrt(n * self.snr)
        nu = n - 1

        threshold = stats.nct.ppf(epsilon, df=nu, nc=mu)
        if np.isnan(threshold) or np.isinf(threshold):
            return np.nan
        beta = threshold ** 2 / (n - 1)
        t = np.sqrt(beta / (1 + beta))

        P_t = self._pairwise_error_prob(t)
        if P_t <= 0:
            return np.nan

        R_nats = -np.log(P_t) / n
        return R_nats / np.log(2)

    def converse_error(self, rate_bits: float) -> float:
        """Compute converse bound `R → ε` (linear-domain reference path).

        Flow: R → target log P → t via bounded minimize_scalar → threshold
        → ε = ``nct.cdf(threshold)``.
        """
        if rate_bits <= 0:
            raise ValueError("rate_bits must be positive")

        n = self.n
        target_log_prob = -n * rate_bits * np.log(2)

        def objective(t):
            if t <= 0.01 or t >= 0.99:
                return 1e10
            P_t = self._pairwise_error_prob(t)
            if P_t <= 0:
                return 1e10
            return (np.log(P_t) - target_log_prob) ** 2

        result = optimize.minimize_scalar(
            objective, bounds=(0.01, 0.99), method="bounded",
            options={"xatol": 1e-8},
        )
        if not result.success:
            return np.nan

        t = result.x
        beta = t ** 2 / (1 - t ** 2)
        threshold = np.sqrt(beta * (n - 1))
        mu = np.sqrt(n * self.snr)
        return stats.nct.cdf(threshold, df=n - 1, nc=mu)

    # =====================================================================
    # Log-domain path (recommended default)
    # =====================================================================

    def _t_to_threshold(self, t: float) -> float:
        """Map `t ∈ (0, 1)` to the NCT threshold `√(β·(n−1))`."""
        beta = t * t / (1 - t * t)
        return np.sqrt(beta * (self.n - 1))

    def _log_eps_forward(self, t: float) -> float:
        """Forward `t → log ε(t)` via log-NCT CDF."""
        threshold = self._t_to_threshold(t)
        mu = np.sqrt(self.n * self.snr)
        return log_nct_cdf(threshold, df=self.n - 1, nc=mu)

    def converse_rate_log(self, epsilon: float,
                          t_lo: float = 1e-8,
                          t_hi: float = 1 - 1e-8) -> float:
        """Log-domain converse `ε → R`.

        Finds `t` such that `ε(t) = ε` by Brent bisection on the log-NCT
        forward map, then returns `R = −log P(t) / (n·ln 2)` via log-Lemma 1.
        Never calls ``scipy.stats.nct.ppf`` and so is not subject to its
        NaN wall at large n / high SNR.
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        log_eps_target = np.log(epsilon)

        def g(t):
            return self._log_eps_forward(t) - log_eps_target

        g_lo = g(t_lo)
        g_hi = g(t_hi)
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)):
            return np.nan
        if g_lo * g_hi > 0:
            return np.nan  # target unreachable

        t_star = optimize.brentq(g, t_lo, t_hi, xtol=1e-10)
        log_P = self.log_pairwise_error_prob(t_star)
        return -log_P / (self.n * np.log(2))

    def log_converse_error(self, rate_bits: float,
                           t_lo: float = 1e-6,
                           t_hi: float = 1 - 1e-6) -> float:
        """Log-domain converse `R → log ε(R)`.

        Finds `t` with `log P(t) = -n·R·ln 2` via Brent, then applies the
        log-NCT forward map at that `t`.  Precision preserved even when
        ε is far below 10⁻³⁰⁰.
        """
        if rate_bits <= 0:
            raise ValueError("rate_bits must be positive")

        target_log_P = -self.n * rate_bits * np.log(2)

        def h(t):
            return self.log_pairwise_error_prob(t) - target_log_P

        h_lo = h(t_lo)
        h_hi = h(t_hi)
        if not (np.isfinite(h_lo) and np.isfinite(h_hi)):
            return np.nan
        if h_lo * h_hi > 0:
            return np.nan

        t_star = optimize.brentq(h, t_lo, t_hi, xtol=1e-10)
        return self._log_eps_forward(t_star)

    def converse_error_log(self, rate_bits: float) -> float:
        """Log-safe `R → ε` returning ε in linear, clamped to 0 at underflow."""
        log_eps = self.log_converse_error(rate_bits)
        if not np.isfinite(log_eps):
            return np.nan
        if log_eps < -700:
            return 0.0
        return float(np.exp(log_eps))


# ---------------------------------------------------------------------------
# Chi-squared converse (Polyanskiy's relaxation)
# ---------------------------------------------------------------------------

class ChiSquaredConverse(AWGNConverseBase):
    """Polyanskiy's meta-converse with ``Q_Y = N(0,(1+P)·I)``.

    The closed form in terms of non-central chi-squared quantiles is
    easier to evaluate than Shannon's original geometric argument, but
    strictly looser — the Q_Y choice is not β-optimal.  Kept as a
    reference; see the mismatch plots in the thesis chapter for the
    quantified relaxation cost relative to :class:`NoncentralTConverse`.

    Only the `ε → R` direction is supported.  For extended-range
    evaluation at large n and high SNR the NCT log-domain path is
    recommended instead.
    """

    def converse_rate(self, epsilon: float) -> float:
        """Compute converse bound `ε → R` via non-central chi-squared."""
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        A = self.snr  # SNR linear (power ratio)
        n = self.n
        q = 1 - epsilon

        delta_p = n / A
        pp0 = stats.ncx2.ppf(q, df=n, nc=delta_p)
        if np.isnan(pp0) or np.isinf(pp0):
            return np.nan

        gammatil = (1 + A) * n - A * pp0
        lgamma = (
            gammatil * np.log2(np.e) / (2 + 2 * A)
            + n / 2 * np.log2(1 + A)
        )
        qq0 = ((1 + A) * n - gammatil) / ((1 + A) * A)

        delta_q = n * (1 + 1 / A)
        term1 = stats.ncx2.cdf(qq0, df=n, nc=delta_q)
        if term1 <= 0:
            return np.nan

        delta = q - stats.ncx2.cdf(pp0, df=n, nc=delta_p)
        term2 = max(0, delta - 2 * q * np.finfo(float).eps) * 2 ** (-lgamma)

        lbeta = np.log2(term1 + term2)
        return -lbeta / n


# ---------------------------------------------------------------------------
# Shannon's solid-angle converse (1959)
# ---------------------------------------------------------------------------

_SOLID_ANGLE_MAX_N = 80


class SolidAngleConverse(AWGNConverseBase):
    """Shannon's 1959 sphere-packing converse via the solid angle ratio.

    Computes `Ω_N(θ)/Ω_N(π)` for the angle `θ = arccos(t)` corresponding
    to the NCT threshold at target ε.  Mathematically equivalent to
    :class:`NoncentralTConverse`, but numerically fragile for `n > 80`
    because `sin^(N−2)(φ)` decays precipitously.  A warning is issued
    when constructed with `n` above the reliable range.
    """

    def __init__(self, n: int, snr_db: float):
        super().__init__(n, snr_db)
        if n > _SOLID_ANGLE_MAX_N:
            warnings.warn(
                f"SolidAngleConverse is unreliable for n={n} > "
                f"{_SOLID_ANGLE_MAX_N}.  Use NoncentralTConverse "
                f"(log-domain) instead.",
                stacklevel=2,
            )

    def _solid_angle_ratio(self, theta: float) -> float:
        """`Ω_N(θ) / Ω_N(π)` with ``N = self.n``.

        The normalising constants cancel in the ratio, leaving the ratio of
        two integrals of `sin^{N−2}(φ)` on `[0, θ]` and `[0, π]`.
        """
        N = self.n

        def integrand(phi):
            return np.sin(phi) ** (N - 2)

        num, _ = integrate.quad(integrand, 0, theta, limit=200)
        den, _ = integrate.quad(integrand, 0, np.pi, limit=200)
        if den == 0:
            return 0.0
        return num / den

    def converse_rate(self, epsilon: float) -> float:
        """Compute converse bound `ε → R` via the solid-angle ratio."""
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        n = self.n
        mu = np.sqrt(n * self.snr)
        nu = n - 1

        # Recover the NCT threshold via scipy (same step as in
        # NoncentralTConverse.converse_rate).
        def objective(th):
            cdf_val = stats.nct.cdf(th, df=nu, nc=mu)
            return (cdf_val - epsilon) ** 2

        result = optimize.minimize_scalar(
            objective, bounds=(0, mu * 2), method="bounded",
            options={"xatol": 1e-8},
        )
        if not result.success:
            return np.nan

        threshold = result.x
        beta = threshold ** 2 / (n - 1)
        t = np.sqrt(beta / (1 + beta))
        theta = np.arccos(t)
        ratio = self._solid_angle_ratio(theta)

        if ratio <= 0:
            return np.nan
        R_nats = -np.log(ratio) / n
        return R_nats / np.log(2)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_CONVERSE_METHODS = {
    "nct": NoncentralTConverse,
    "chi2": ChiSquaredConverse,
    "solid_angle": SolidAngleConverse,
}


def awgn_converse_rate(n: int, epsilon: float, snr_db: float,
                       method: str = "nct") -> float:
    """Compute the AWGN converse `ε → R` using the chosen method.

    Parameters
    ----------
    n : int
    epsilon : float
    snr_db : float
    method : {"nct", "chi2", "solid_angle"}
        Which converse class to use.  Default ``"nct"`` uses the linear
        path of :class:`NoncentralTConverse`; for the log-safe extended
        range call ``NoncentralTConverse(...).converse_rate_log(eps)``
        directly.
    """
    cls = _CONVERSE_METHODS[method]
    return cls(n=n, snr_db=snr_db).converse_rate(epsilon)


def awgn_converse_error(n: int, rate_bits: float, snr_db: float) -> float:
    """Compute the AWGN converse `R → ε` via :class:`NoncentralTConverse`."""
    return NoncentralTConverse(n=n, snr_db=snr_db).converse_error(rate_bits)
