"""
Achievability (lower) bounds on the maximum coding rate for the AWGN
channel, under i.i.d.\\ power-shell input.

Five classes, grouped in this one module for compactness:

* :class:`RCUAchievable` — our flagship bound, the one-shot metaconverse
  integral `P(log M) = ∫ F(γ)·e^(log M − γ) dγ` that reuses the exact F(R)
  of :class:`~awgn_fbl.converse.NoncentralTConverse`.  Offers a simple
  linear evaluation and a log-safe path using the Elkayam factorisation
  `P(R) = F(R)·J(R)`.

* :class:`KappaBetaAchievable` — our first-pass port of Polyanskiy's κβ
  bound.  Simpler code, uses a one-shot ncx² quantile plus an additive
  correction term for β_q.  Kept as a reference implementation.

* :class:`KappaBetaAchievablePPV` — faithful port of Polyanskiy's
  original MATLAB (``kappabeta_ach.m`` + ``betaq_up_v2.m``).  Newton
  iteration on the β quantile plus a log-domain ncx² fallback.  More
  accurate than the simple version at the edges; the tests cross-check
  the two in the shared regime.

* :class:`GallagerAchievable` — Gallager's random coding bound with the
  two-regime closed form specific to the AWGN power shell.  Ported from
  Polyanskiy's ``gallager_ach.m``.

* :class:`ExactRandomCoding` — Monte-Carlo evaluation of the *exact*
  random-coding error probability
  `E[1 − (1 − G(T))^(M−1)]` for uniform-on-shell input.  Useful at small
  n to quantify the cost of the min-with-1 envelope that RCU⁺ applies.
  The ``rcu_union_error`` method also serves as a cross-check of the
  library's RCU⁺ integral form.

All bounds use the same ``(n, snr_db)`` construction signature.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, stats

from ._pairwise import log_pairwise_error_prob_vec
from .converse import NoncentralTConverse
from .fast_f import FastFREvaluator


__all__ = [
    "RCUAchievable",
    "KappaBetaAchievable",
    "KappaBetaAchievablePPV",
    "GallagerAchievable",
    "ExactRandomCoding",
]


# ===========================================================================
# RCU+
# ===========================================================================

class RCUAchievable:
    """RCU⁺ achievability bound via the one-shot meta-converse integral.

    `P(log M) = ∫_{log M}^∞ F(γ) · exp(log M − γ) dγ`,
    where `F(γ)` is the exact converse curve from
    :class:`~awgn_fbl.converse.NoncentralTConverse`, precomputed on a
    log-ε grid by :class:`~awgn_fbl.fast_f.FastFREvaluator`.

    The class offers a *simple* linear evaluation (``achievable_error``,
    ``achievable_rate``) and a *log-safe* evaluation via the Elkayam
    factorisation `P(R) = F(R) · J(R)` (``log_achievable_error``,
    ``achievable_rate_v2``).  The two agree in their shared regime; the
    log-safe path keeps `log P(R)` well-conditioned far below machine
    precision in linear space.

    **Deep-tail reach.**  The bound's reach is set by the depth of the
    converse curve `F(R)` it integrates: the log-safe path can only report
    error probabilities as small as the smallest ε on the
    :class:`~awgn_fbl.fast_f.FastFREvaluator` grid.  The default grid floors
    at ``eps_min = 1e-10``; pass a smaller ``eps_min`` to reach further into
    the tail (the underlying log-domain converse stays accurate essentially
    arbitrarily deep, so the only cost is a larger — and slower to build —
    grid).  With e.g. ``eps_min=1e-100`` the RCU⁺ waterfall tracks the
    converse down to `P_e ≈ 10⁻¹⁰⁰`.

    Parameters
    ----------
    n, snr_db : int, float
        Blocklength and SNR (dB).
    method : str
        Converse method for the F-grid (``'nct'`` default; ``'chi2'``).
    eps_start, eps_factor, eps_min : float
        Forwarded to :class:`~awgn_fbl.fast_f.FastFREvaluator`; control the
        ε-grid extent.  Lower ``eps_min`` for deeper-tail reach.
    verbose : bool
        Print F-grid precomputation progress.
    """

    def __init__(self, n: int, snr_db: float, *, method: str = "nct",
                 eps_start: float = 1 - 1e-10, eps_factor: float = 0.1,
                 eps_min: float = 1e-10, verbose: bool = False):
        self.n = n
        self.snr_db = snr_db
        self.verbose = verbose
        self.F_eval = FastFREvaluator(n=n, snr_db=snr_db, method=method,
                                      eps_start=eps_start, eps_factor=eps_factor,
                                      eps_min=eps_min, verbose=verbose)

    # ------------------------------------------------------------------
    # Simple linear-domain path
    # ------------------------------------------------------------------

    def achievable_error(self, R_bits: float,
                         integration_limit: float = 15.0) -> float:
        """Achievable error probability at rate R (bits/channel use).

        Evaluates the RCU⁺ integral in linear domain.  Accurate for
        `P(R) ≳ 10⁻¹⁵`; below that use :meth:`achievable_error_v2`.
        """
        n = self.n
        log_M = n * R_bits * np.log(2)
        R_min_nats = self.F_eval.R_min * n * np.log(2)

        def integrand(gamma_nats):
            if gamma_nats < R_min_nats:
                return 0.0
            R_per_use = gamma_nats / (n * np.log(2))
            F_val = self.F_eval(R_per_use)
            exponent = log_M - gamma_nats
            if exponent < -100:
                return 0.0
            return F_val * np.exp(exponent)

        result, _ = integrate.quad(
            integrand, log_M, log_M + integration_limit,
            limit=100, epsabs=1e-12, epsrel=1e-10,
        )
        return result

    def achievable_rate(self, epsilon: float,
                        R_min: float | None = None,
                        R_max: float | None = None,
                        tol: float = 1e-6) -> float:
        """Achievable rate at target ε.

        Root-finds `achievable_error(R) = ε` by Brent bisection.  The
        search is capped at the converse rate (an achievability can never
        exceed the converse), which also rejects out-of-grid ε values.
        """
        if R_min is None:
            R_min = self.F_eval.R_min
        R_converse = self.F_eval.converse_rate(epsilon)
        if np.isnan(R_converse) or np.isinf(R_converse):
            return np.nan
        if R_max is None:
            R_max = min(self.F_eval.R_max, R_converse)
        else:
            R_max = min(R_max, R_converse)
        if R_max <= R_min:
            return np.nan

        def objective(R):
            return self.achievable_error(R) - epsilon

        f_lo = objective(R_min)
        f_hi = objective(R_max)
        if np.isnan(f_lo) or np.isnan(f_hi):
            return np.nan
        if f_lo > 0:
            return 0.0
        if f_hi < 0:
            return R_max

        return optimize.brentq(objective, R_min, R_max, xtol=tol)

    # ------------------------------------------------------------------
    # Log-safe path — Elkayam factorisation  P(R) = F(R) · J(R)
    # ------------------------------------------------------------------

    def log_achievable_error(self, R_bits: float,
                             integration_limit: float = 15.0) -> float:
        """log P(R) in natural log, via `P = F(R) · J(R)`.

        `J(R) = ∫_{log M}^∞ [F(γ)/F(R)] · exp(log M − γ) dγ`, bounded in
        `[1, 1/F(R)]`.  Keeps the deep-tail magnitude separated from a
        well-conditioned O(1)-to-O(−log F(R)) inner integral.
        """
        n = self.n
        log_M = n * R_bits * np.log(2)
        R_min_nats = self.F_eval.R_min * n * np.log(2)
        log_F_R = self.F_eval.log_F(R_bits)

        def inner_integrand(gamma_nats):
            if gamma_nats < R_min_nats:
                return 0.0
            R_per_use = gamma_nats / (n * np.log(2))
            log_F_gamma = self.F_eval.log_F(R_per_use)
            log_ratio = log_F_gamma - log_F_R
            exponent = log_M - gamma_nats
            log_integrand = log_ratio + exponent
            if log_integrand < -100:
                return 0.0
            return np.exp(log_integrand)

        J, _ = integrate.quad(
            inner_integrand, log_M, log_M + integration_limit,
            limit=100, epsabs=1e-12, epsrel=1e-10,
        )
        if J <= 0:
            return -np.inf
        return log_F_R + np.log(J)

    def achievable_error_v2(self, R_bits: float,
                            integration_limit: float = 15.0) -> float:
        """Log-safe alternative to :meth:`achievable_error`.

        Equal in exact arithmetic; stays well-conditioned when `P(R)` is
        many orders of magnitude below 1.
        """
        log_P = self.log_achievable_error(R_bits, integration_limit)
        if log_P < -700:
            return 0.0
        return float(np.exp(log_P))

    def achievable_rate_v2(self, epsilon: float,
                           R_min: float | None = None,
                           R_max: float | None = None,
                           tol: float = 1e-6) -> float:
        """Log-domain counterpart of :meth:`achievable_rate`."""
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        if R_min is None:
            R_min = self.F_eval.R_min
        R_converse = self.F_eval.converse_rate(epsilon)
        if np.isnan(R_converse) or np.isinf(R_converse):
            return np.nan
        if R_max is None:
            R_max = min(self.F_eval.R_max, R_converse)
        else:
            R_max = min(R_max, R_converse)
        if R_max <= R_min:
            return np.nan

        log_eps_target = np.log(epsilon)

        def objective(R):
            return self.log_achievable_error(R) - log_eps_target

        f_lo = objective(R_min)
        f_hi = objective(R_max)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return np.nan
        if f_lo > 0:
            return 0.0
        if f_hi < 0:
            return R_max

        return optimize.brentq(objective, R_min, R_max, xtol=tol)


# ===========================================================================
# κβ — simple reference and faithful PPV port
# ===========================================================================

class _KappaBetaBase:
    """Shared asymptotic κ and driver used by both κβ implementations."""

    def __init__(self, n: int, snr_db: float):
        if n < 1:
            raise ValueError("Block length n must be positive")
        self.n = n
        self.snr_db = snr_db
        self.P = 10 ** (snr_db / 10)
        self.A = np.sqrt(self.P)

    def _kappa_inf(self, tau: float) -> float:
        """Asymptotic κ(τ).  Matches Polyanskiy's ``kappa_inf.m``."""
        if tau <= 0 or tau >= 1:
            return 0.0
        P = self.P
        x0 = stats.norm.ppf((tau + 1) / 2)
        VP = 2 * (1 + 2 * P)
        VQ = 2 * (1 + P) ** 2
        return 2 * stats.norm.cdf(np.sqrt(VP / VQ) * x0) - 1

    def _betaq_upper(self, q: float) -> float:
        """Upper bound on ``log₂ β_q``.  Implementation provided by subclass."""
        raise NotImplementedError

    def achievable_rate(self, epsilon: float, n_tau: int = 40) -> float:
        """κβ achievable rate, ``bits/channel use``.

        Matches ``kappabeta_ach.m`` with asymptotic κ (Polyanskiy's
        ``hack = 1`` default).  Optimised over τ ∈ (0, ε).
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        taus = np.linspace(0, 1, n_tau) * epsilon
        taus = taus[2:-2]  # same endpoint exclusion as the MATLAB reference

        log_M_values = []
        for tau in taus:
            kappa = self._kappa_inf(tau)
            if kappa <= 0:
                continue
            q = min(1 - epsilon + tau, 1 - 1e-10)
            lbeta = self._betaq_upper(q)
            if not np.isfinite(lbeta):
                continue
            log_M_values.append(np.log2(kappa) - lbeta)

        if not log_M_values:
            return np.nan
        return float(np.max(log_M_values)) / self.n


class KappaBetaAchievable(_KappaBetaBase):
    """Simple reference port of Polyanskiy's κβ achievability.

    Uses a one-shot ``ncx2.ppf`` quantile for the β upper bound, with an
    additive correction term of order ε·eps_machine (the correction one
    would expect from the discreteness of the quantile function).  Slightly
    less accurate than :class:`KappaBetaAchievablePPV` in extreme regimes,
    but considerably simpler code and a useful cross-check for the
    faithful version.

    Applies to the *maximal* error probability `ε* = max_m P_e(m)`.
    """

    def _betaq_upper(self, q: float) -> float:
        A = self.A
        n = self.n

        pp0 = stats.ncx2.ppf(q, df=n, nc=n / A ** 2)
        if np.isnan(pp0) or np.isinf(pp0):
            return -np.inf

        gammatil = (1 + A ** 2) * n - A ** 2 * pp0
        lgamma = (
            gammatil * np.log2(np.e) / (2 + 2 * A ** 2)
            + n / 2 * np.log2(1 + A ** 2)
        )
        qq0 = ((1 + A ** 2) * n - gammatil) / ((1 + A ** 2) * A ** 2)
        delta_q = n * (1 + 1 / A ** 2)

        term1 = stats.ncx2.cdf(qq0, df=n, nc=delta_q)
        if term1 <= 0:
            return -lgamma

        delta = q - stats.ncx2.cdf(pp0, df=n, nc=n / A ** 2)
        term2 = max(0, delta - 2 * q * np.finfo(float).eps) * 2 ** (-lgamma)
        return np.log2(term1 + term2)


class KappaBetaAchievablePPV(_KappaBetaBase):
    """Faithful port of Polyanskiy's ``kappabeta_ach.m`` + ``betaq_up_v2.m``.

    The β upper bound uses Newton iteration on the ncx² quantile ``pp0``
    until ``ncx2.cdf(pp0) ≥ q`` (an "overshoot" variant of the quantile),
    then reports ``log₂(term1)`` with no additive correction.  A log-scale
    ``ncx2.logcdf`` fallback is used when ``term1`` underflows.  This is
    the reference implementation one should compare against when
    reproducing Polyanskiy's published numbers.

    Applies to the *maximal* error probability `ε* = max_m P_e(m)`.
    """

    def _betaq_upper(self, q: float, max_iter: int = 50) -> float:
        A = self.A
        n = self.n
        A2 = A ** 2

        # Step 1: Newton-iterate pp0 upward until CDF ≥ q
        pp0 = stats.ncx2.ppf(q, df=n, nc=n / A2)
        if np.isnan(pp0) or np.isinf(pp0):
            return np.inf

        for _ in range(max_iter):
            pgam = stats.ncx2.cdf(pp0, df=n, nc=n / A2)
            if pgam >= q:
                break
            delta = q - pgam
            pdf_val = stats.ncx2.pdf(pp0, df=n, nc=n / A2)
            if pdf_val <= 0:
                break
            pp0 = pp0 + delta / pdf_val

        # Step 2: change of measure pp0 → qq0
        gammatil = (1 + A2) * n - A2 * pp0
        lgamma = (
            gammatil * np.log2(np.e) / (2 + 2 * A2)
            + n / 2 * np.log2(1 + A2)
        )
        qq0 = ((1 + A2) * n - gammatil) / ((1 + A2) * A2)

        # Step 3: term1 under the output measure Q_Y
        term1 = stats.ncx2.cdf(qq0, df=n, nc=n * (1 + 1 / A2))
        if term1 > 0:
            return np.log2(term1)

        # Step 4: log-scale fallback when linear underflows
        log_term1 = stats.ncx2.logcdf(qq0, df=n, nc=n * (1 + 1 / A2))
        if np.isfinite(log_term1):
            return log_term1 * np.log2(np.e)

        # Last-resort: use the NP threshold alone
        return -lgamma


# ===========================================================================
# Gallager random coding
# ===========================================================================

class GallagerAchievable:
    """Gallager's random-coding bound, two-regime closed form for AWGN.

    `P_e ≤ μ(n, ρ) · exp(−n · E_r(R, ρ))`, with two regimes:
    `ρ ∈ (0, 1)` for `R ≥ R_cr`, and the expurgated `ρ = 1` branch for
    `R < R_cr`.  Ported from Polyanskiy's ``gallager_ach.m``.

    Applies to the *average* error probability.
    """

    def __init__(self, n: int, snr_db: float):
        if n < 1:
            raise ValueError("Block length n must be positive")
        self.n = n
        self.snr_db = snr_db
        self.P = 10 ** (snr_db / 10)
        self.A = np.sqrt(self.P)
        self.capacity = 0.5 * np.log2(1 + self.P)

        A2 = self.A ** 2
        self.Rcr_bits = 0.5 * np.log2(
            0.5 + A2 / 4 + 0.5 * np.sqrt(1 + A2 ** 2 / 4)
        )

    def _gallager_pe(self, R_bits: float) -> float:
        """Gallager's upper bound on `P_e` at rate R (bits/channel use)."""
        if R_bits <= 0:
            return 0.0
        n = self.n
        A = self.A
        A2 = A ** 2
        R = R_bits / np.log2(np.e)           # convert to nats
        Rcr = self.Rcr_bits / np.log2(np.e)

        if R >= Rcr:
            beta = np.exp(2 * R)
            sqrt_arg = 1 + 4 * beta / (A2 * (beta - 1))
            sq = np.sqrt(sqrt_arg)
            ro = A2 / (2 * beta) * (1 + sq) - 1
            Er = (
                A2 / (4 * beta) * ((beta + 1) - (beta - 1) * sq)
                + 0.5 * np.log(beta - A2 * (beta - 1) / 2 * (sq - 1))
            )
        else:
            beta = 0.5 * (1 + A2 / 2 + np.sqrt(1 + A2 ** 2 / 4))
            ro = 1.0
            Er = (
                1 - beta + A2 / 2
                + 0.5 * np.log(beta - A2 / 2)
                + 0.5 * np.log(beta)
                - R
            )

        # Gallager 7.4.39 with δ' = 1/s
        s = ro * A2 / (2 * (1 + ro) ** 2 * beta)
        deltap = float(n) if s == 0 else min(1.0 / s, float(n))

        hi = stats.chi2.cdf(n, df=n)
        lo = stats.chi2.cdf(n - deltap, df=n)
        mu = hi - lo
        if mu <= 0:
            return 1.0

        multip = 2 * np.exp(s * deltap) / mu
        return min(multip * np.exp(-n * Er), 1.0)

    def achievable_rate(self, epsilon: float, tol: float = 1e-4) -> float:
        """Largest R for which Gallager guarantees `P_e ≤ ε`."""
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")

        R_up = self.capacity
        R_down = 0.0
        if self._gallager_pe(R_down) > epsilon:
            return 0.0

        precision = tol * R_up
        while (R_up - R_down) > precision:
            R_mid = 0.5 * (R_up + R_down)
            if self._gallager_pe(R_mid) < epsilon:
                R_down = R_mid
            else:
                R_up = R_mid
        return R_down

    def achievable_error(self, R_bits: float) -> float:
        """Gallager upper bound on error probability at rate R."""
        return self._gallager_pe(R_bits)


# ===========================================================================
# Exact random coding (Monte Carlo)
# ===========================================================================

class ExactRandomCoding:
    """Monte-Carlo estimate of the exact random-coding error probability.

    Given M codewords iid uniform on the power shell `{‖x‖² = nP}` and a
    transmitted codeword x₀, the conditional pairwise error probability
    given the correlation `T = ρ̂(x₀, Y)` is `1 − (1 − G(T))^(M−1)` where
    `G(t)` is the Lemma 1 integral.  The unconditioned error probability is
    `P_e^{RC}(n, M) = E_T[1 − (1 − G(T))^(M−1)]`.

    This class samples T under H₀ and averages, using the log-domain
    Lemma 1 for numerical stability.  The companion :meth:`rcu_union_error`
    evaluates `E_T[min(1, (M−1)·G(T))]`, the min-with-1 envelope that
    RCU⁺ applies; the two together quantify the envelope cost.

    Applies to the *average* error probability.  Intended for small n
    where the MC sample size needed is manageable; for large n the
    envelope cost is negligible and `RCUAchievable` is recommended.
    """

    def __init__(self, n: int, snr_db: float, seed: int | None = None):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.snr_db = snr_db
        self.P = 10 ** (snr_db / 10)
        self._rng = np.random.default_rng(seed)

    # -- internal helpers --------------------------------------------------

    def _sample_T(self, n_samples: int) -> np.ndarray:
        """Sample `T = ρ̂(x₀, Y)` under H₀ = "x₀ sent"."""
        mu = np.sqrt(self.n * self.P)
        Z1 = self._rng.standard_normal(n_samples)
        chi2 = self._rng.chisquare(df=self.n - 1, size=n_samples)
        U = mu + Z1
        return U / np.sqrt(U * U + chi2)

    def _log_G_vec(self, ts: np.ndarray) -> np.ndarray:
        return log_pairwise_error_prob_vec(ts, self.n)

    def _log_M_minus_1(self, R_bits: float) -> float:
        log_M = self.n * R_bits * np.log(2)
        if log_M > 40:
            return log_M
        return np.log(np.expm1(log_M))

    @staticmethod
    def _exact_bracket(log_Gs: np.ndarray, log_Mm1: float) -> np.ndarray:
        """`1 − (1 − G)^(M−1)`, vectorised and log-safe."""
        G = np.where(log_Gs > -700, np.exp(np.minimum(log_Gs, 0)), 0.0)
        G = np.clip(G, 0.0, 1.0 - 1e-15)
        log_1mG = np.log1p(-G)
        exponent = np.exp(log_Mm1) * log_1mG
        return np.where(exponent < -700, 1.0, -np.expm1(exponent))

    @staticmethod
    def _union_bracket(log_Gs: np.ndarray, log_Mm1: float) -> np.ndarray:
        """`min(1, (M−1)·G)`."""
        log_terms = log_Mm1 + log_Gs
        return np.where(log_terms >= 0, 1.0, np.exp(log_terms))

    def _mc_mean(self, bracket_fn, R_bits: float, n_samples: int,
                 chunk: int) -> float:
        log_Mm1 = self._log_M_minus_1(R_bits)
        total = 0.0
        processed = 0
        while processed < n_samples:
            this = min(chunk, n_samples - processed)
            Ts = self._sample_T(this)
            log_Gs = self._log_G_vec(Ts)
            total += float(np.sum(bracket_fn(log_Gs, log_Mm1)))
            processed += this
        return total / n_samples

    # -- public methods ----------------------------------------------------

    def exact_error(self, R_bits: float,
                    n_samples: int = 50_000,
                    chunk: int = 200_000) -> float:
        """MC estimate of `E_T[1 − (1 − G(T))^(M−1)]`.

        Precision scales as `1/√n_samples`; for `P_e ≲ 10⁻⁶` use
        `n_samples ≳ 10⁷`.
        """
        return self._mc_mean(self._exact_bracket, R_bits, n_samples, chunk)

    def rcu_union_error(self, R_bits: float,
                        n_samples: int = 50_000,
                        chunk: int = 200_000) -> float:
        """MC estimate of `E_T[min(1, (M−1)·G(T))]`.

        By integration by parts this equals the RCU⁺ integral form
        (up to the trivial M↔M−1 correction), which makes MC agreement
        a cross-check against :class:`RCUAchievable`.
        """
        return self._mc_mean(self._union_bracket, R_bits, n_samples, chunk)
