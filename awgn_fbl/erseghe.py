"""
Robust evaluation of the *relaxed* PPV converse (Polyanskiy's ``Q_Y``) for the
AWGN channel, following Erseghe, "On the Evaluation of the Polyanskiy-Poor-Verdú
Converse Bound for Finite Block-length Coding in AWGN", IEEE Trans. Inf.
Theory 61(12):6578-6590, 2015 (arXiv:1401.7169).

What it computes
----------------
This is the meta-converse with the *capacity-achieving output measure*
``Q_Y = N(0, (1+P)·I)`` — i.e. the same bound as
:class:`~awgn_fbl.converse.ChiSquaredConverse`, **not** the optimal
cone-packing (NCT) converse.  It is strictly looser than the cone-packing
bound at finite ``n`` (the relaxation the chapter's mismatch figures quantify);
it is kept as the standard literature reference, evaluated robustly.

Why this and not scipy ``ncx2``
-------------------------------
Erseghe expresses the false-alarm / missed-detection probabilities of the
meta-converse as non-central chi-squared tails (his Theorem 1) and evaluates
them with Temme's method: a single, well-conditioned integral in which ``n``
appears only in an exponent (his Theorem 4).  This stays accurate where
scipy's ``ncx2.ppf`` returns ``NaN`` (large ``n``, high SNR).  The integral
form here agrees with scipy ``ncx2`` to ~1e-12 in the shared regime and
extends past scipy's NaN wall.

Two evaluation paths are exposed, mirroring the library's simple/robust split:

* ``method="integral"`` (default) — Erseghe's exact single integral (Thm 4).
* ``method="asymptotic"`` — the two-term uniform asymptotic expansion
  (Erseghe eqs 34-36); ~0.1% of the integral for ``n >= 100``, much cheaper.

Only the regime ``P_e < 1/2`` and ``R > 1/n`` is supported (Erseghe's
assumption (33)); both always hold at sensible operating points.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


__all__ = ["ErsegheConverse"]


class ErsegheConverse:
    """Erseghe's robust evaluation of the relaxed (``Q_Y``) PPV converse.

    Parameters
    ----------
    n : int
        Blocklength.
    snr_db : float
        SNR in dB.  The linear SNR ``Omega = 10^(snr_db/10)`` is stored as
        ``self.Omega``.
    method : {"integral", "asymptotic"}
        Evaluation path; ``"integral"`` (default) is exact, ``"asymptotic"``
        is the cheaper two-term expansion.
    n_grid : int
        Quadrature points for the integral path.
    """

    def __init__(self, n: int, snr_db: float, *, method: str = "integral",
                 n_grid: int = 4000):
        if n < 1:
            raise ValueError("Block length n must be positive")
        if method not in ("integral", "asymptotic"):
            raise ValueError("method must be 'integral' or 'asymptotic'")
        self.n = n
        self.snr_db = snr_db
        self.Omega = 10 ** (snr_db / 10)
        self.method = method
        self.n_grid = n_grid

    # ------------------------------------------------------------------
    # gamma parametrisation and the building-block functions (Erseghe eq 16)
    # ------------------------------------------------------------------

    def _gamma_bounds(self, margin: float = 1e-9) -> tuple[float, float]:
        """Open interval (eq 33): P_e < 1/2 and R > 1/n."""
        Omega = self.Omega
        g_lo = 0.5 * np.log1p(Omega / (1 + Omega))
        g_hi = 0.5 * np.log1p(Omega)
        span = g_hi - g_lo
        return g_lo + margin * span, g_hi - margin * span

    def _theta(self, gamma: float, which: str) -> float:
        sg = np.sinh(gamma)
        base = np.log(self.Omega / (2 * sg))
        return base if which == "MD" else base - np.log1p(self.Omega)

    def _v(self, gamma: float, which: str) -> float:
        """Exponential-rate function ``v(gamma)`` (eq 16)."""
        sg = np.sinh(gamma)
        cg = np.cosh(gamma)
        th = self._theta(gamma, which)
        alpha_theta = cg - np.cosh(th) + sg * (th - gamma)
        return -alpha_theta / sg

    # ------------------------------------------------------------------
    # g(gamma): the O(1) prefactor
    # ------------------------------------------------------------------

    def _g_integral(self, gamma: float, which: str) -> float:
        """Exact prefactor via Erseghe Theorem 4 (eqs 18-19)."""
        n, Omega, ng = self.n, self.Omega, self.n_grid
        sg = np.sinh(gamma)
        cg = np.cosh(gamma)
        th = self._theta(gamma, which)

        phi = (np.arange(ng) + 0.5) * (np.pi / ng)        # midpoints on (0, pi)
        sinphi = np.sin(phi)
        sinc = sinphi / phi
        r = np.arcsinh(sg / sinc)
        rp = (1.0 / phi - 1.0 / np.tan(phi)) / np.sqrt(1.0 + (sinc / sg) ** 2)
        alpha_r = cg - np.cosh(r) + sg * (r - gamma)
        h = (1.0 - np.cos(phi)) * np.cosh(r) + alpha_r
        edr = np.exp(th - r)
        gtilde = (edr * (np.cos(phi) + rp * sinphi) - 1.0) / \
                 (edr ** 2 - 2.0 * np.cos(phi) * edr + 1.0)
        weight = np.exp(-n * (2.0 * h) / (4.0 * sg))
        return float(np.mean(gtilde * weight))            # (1/pi)∫_0^pi ... dphi

    def _g_asymptotic(self, gamma: float, which: str) -> tuple[float, float]:
        """Two-term expansion coefficients ``g0, g1`` (eq 36)."""
        sg = np.sinh(gamma)
        tg = np.tanh(gamma)
        th = self._theta(gamma, which)
        em1 = np.expm1(th - gamma)
        g0 = np.sqrt(tg / np.pi) / em1
        g1 = tg * ((9 - 12 * tg + 5 * tg ** 2) / 12
                   + (2 + (3 - tg) * em1) / em1 ** 2)
        return g0, g1

    # ------------------------------------------------------------------
    # log probabilities
    # ------------------------------------------------------------------

    def _log_prob(self, gamma: float, which: str) -> float:
        """``log P_MD`` or ``log P_FA`` at ``gamma`` (Erseghe eq 14 under (33)).

        Integral path: ``P = |g(gamma)| · e^{-n v/2}`` where the Theorem-4
        integral already carries the ``1/sqrt(n)`` width.  Asymptotic path:
        ``P = |g0| · n^{-1/2} · (1 - g1/n) · e^{-n v/2}`` (eqs 34-36).
        """
        v = self._v(gamma, which)
        if self.method == "integral":
            g = self._g_integral(gamma, which)
            return np.log(abs(g)) - 0.5 * self.n * v
        g0, g1 = self._g_asymptotic(gamma, which)
        lp = np.log(abs(g0)) - 0.5 * np.log(self.n) - 0.5 * self.n * v
        corr = 1.0 - g1 / self.n
        if corr > 0:
            lp += np.log(corr)
        return lp

    # ------------------------------------------------------------------
    # public converse maps
    # ------------------------------------------------------------------

    def converse_rate(self, epsilon: float) -> float:
        """Upper bound on rate (bits/use) at target error probability ``epsilon``.

        Solves ``P_MD(gamma) = epsilon`` then returns
        ``R = -log2 P_FA(gamma) / n`` (Erseghe Theorem 2).
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        gamma = self._solve_gamma_for_eps(epsilon)
        if not np.isfinite(gamma):
            return np.nan
        return -self._log_prob(gamma, "FA") / (self.n * np.log(2))

    def log_converse_error(self, rate_bits: float) -> float:
        """``log epsilon`` (natural log) at rate ``rate_bits`` (R -> eps direction)."""
        if rate_bits <= 0:
            raise ValueError("rate_bits must be positive")
        target_log_pfa = -self.n * rate_bits * np.log(2)
        _, g_hi = self._gamma_bounds()
        # log P_FA rises briefly then decreases monotonically; root-find on the
        # decreasing branch from its peak to the upper bound.
        g_peak = self._argmax_logprob("FA")

        def f(g):
            return self._log_prob(g, "FA") - target_log_pfa

        f_peak, f_hi = f(g_peak), f(g_hi)
        if not (np.isfinite(f_peak) and np.isfinite(f_hi)) or f_peak * f_hi > 0:
            return np.nan
        gamma = optimize.brentq(f, g_peak, g_hi, xtol=1e-12)
        return self._log_prob(gamma, "MD")

    def converse_error(self, rate_bits: float) -> float:
        """Error probability at rate ``rate_bits`` (R -> eps), clamped at underflow."""
        log_eps = self.log_converse_error(rate_bits)
        if not np.isfinite(log_eps):
            return np.nan
        return 0.0 if log_eps < -700 else float(np.exp(log_eps))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _argmax_logprob(self, which: str) -> float:
        """gamma maximising ``log P_{which}`` over the valid interval.

        The leading Temme expansion is accurate in the tail but turns over as
        the probability approaches O(1) (near the P_e = 1/2 boundary); the
        argmax marks the end of the monotone branch used for root-finding.
        """
        g_lo, g_hi = self._gamma_bounds()
        res = optimize.minimize_scalar(
            lambda g: -self._log_prob(g, which),
            bounds=(g_lo, g_hi), method="bounded",
            options={"xatol": 1e-10},
        )
        return res.x

    def _solve_gamma_for_eps(self, epsilon: float) -> float:
        target = np.log(epsilon)
        g_lo, _ = self._gamma_bounds()
        # log P_MD rises monotonically up to its peak, then the expansion turns
        # over; root-find only on the rising branch.
        g_peak = self._argmax_logprob("MD")

        def f(g):
            return self._log_prob(g, "MD") - target

        f_lo, f_peak = f(g_lo), f(g_peak)
        if not (np.isfinite(f_lo) and np.isfinite(f_peak)) or f_lo * f_peak > 0:
            return np.nan  # epsilon outside the method's valid (tail) range
        return optimize.brentq(f, g_lo, g_peak, xtol=1e-12)
