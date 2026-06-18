"""
Fast evaluator for the converse curve `F: R ↦ ε`.

Given `(n, SNR)`, :class:`FastFREvaluator` sweeps an adaptive grid of ε
values, evaluates the converse rate at each, and fits a monotone PCHIP
interpolant in log(ε) space.  Subsequent `F(R)` queries are then
microsecond-level lookups, which matters for the RCU⁺ integrand that is
evaluated hundreds of times inside ``scipy.integrate.quad``.

The grid construction defaults to the log-domain converse
:meth:`NoncentralTConverse.converse_rate_log`, which extends the working
range well beyond where scipy's linear NCT breaks down.  Pass
``log_domain=False`` for the linear path when that reference behaviour is
wanted (used by some tests).
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate

from .converse import ChiSquaredConverse, NoncentralTConverse


__all__ = ["FastFREvaluator"]


class FastFREvaluator:
    """Precomputed `F(R)` with monotone PCHIP interpolation in log(ε).

    Parameters
    ----------
    n : int
        Blocklength.
    snr_db : float
        SNR in dB.
    eps_start : float
        Largest ε in the sweep (default `1 − 10⁻¹⁰`).
    eps_factor : float
        Geometric shrink factor: `ε_{i+1} = ε_i · (1 − eps_factor)`.
    eps_min : float
        Smallest ε in the sweep.
    method : str
        ``'nct'`` (default) or ``'chi2'`` — which converse to evaluate.
    log_domain : bool
        If True and ``method='nct'``, precompute via
        :meth:`NoncentralTConverse.converse_rate_log`; otherwise use the
        linear path.  Silently forced to False for ``method='chi2'``.
    verbose : bool
        Print progress during precomputation.
    """

    def __init__(
        self,
        n: int,
        snr_db: float,
        eps_start: float = 1 - 1e-10,
        eps_factor: float = 0.1,
        eps_min: float = 1e-10,
        method: str = "nct",
        log_domain: bool = True,
        verbose: bool = False,
    ):
        self.n = n
        self.snr_db = snr_db
        self.method = method
        self.log_domain = log_domain
        self.verbose = verbose

        cls = NoncentralTConverse if method == "nct" else ChiSquaredConverse
        self._converse = cls(n=n, snr_db=snr_db)

        # Only the NCT class exposes a log-domain converse; ignore the flag
        # for chi².
        if log_domain and method != "nct":
            self.log_domain = False

        self._eps_raw = self._generate_eps_grid(eps_start, eps_factor, eps_min)

        if verbose:
            print(f"Precomputing F(R) grid with {len(self._eps_raw)} points ...")

        self._precompute()

        if verbose:
            print(f"  R range: [{self.R_min:.6f}, {self.R_max:.6f}] bits")
            print(f"  eps range: [{self.eps_min:.2e}, {self.eps_max:.2e}]")
            print(f"  {len(self.R_grid)} valid grid points")

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_eps_grid(eps_start, eps_factor, eps_min) -> np.ndarray:
        eps_values = []
        eps_current = eps_start
        while eps_current > eps_min:
            eps_values.append(eps_current)
            eps_current *= 1 - eps_factor
        eps_values.append(eps_min)
        return np.array(sorted(set(eps_values)))

    def _precompute(self):
        """Sweep ε grid, evaluate the converse at each, build the PCHIP interp."""
        if self.log_domain:
            rate_fn = self._converse.converse_rate_log
        else:
            rate_fn = self._converse.converse_rate

        eps_list, R_list = [], []
        for i, eps in enumerate(self._eps_raw):
            if self.verbose and i % 20 == 0:
                print(f"  {i}/{len(self._eps_raw)}", end="\r")
            R = rate_fn(eps)
            if np.isnan(R) or np.isinf(R):
                continue
            eps_list.append(eps)
            R_list.append(R)

        if not R_list:
            raise RuntimeError("Failed to compute any valid R values")

        eps_arr = np.array(eps_list)
        R_arr = np.array(R_list)

        # Sort by R (required for monotone interpolation)
        order = np.argsort(R_arr)
        R_arr = R_arr[order]
        eps_arr = eps_arr[order]

        # Deduplicate numerically identical R values
        unique_idx = np.concatenate(([True], np.diff(R_arr) > 0))
        self.R_grid = R_arr[unique_idx]
        self.eps_grid = eps_arr[unique_idx]

        log_eps = np.log(self.eps_grid)
        self._interp_core = interpolate.PchipInterpolator(
            self.R_grid, log_eps, extrapolate=False,
        )
        self._log_eps_lo = log_eps[0]
        self._log_eps_hi = log_eps[-1]

        self.R_min = float(self.R_grid[0])
        self.R_max = float(self.R_grid[-1])
        self.eps_min = float(self.eps_grid[0])
        self.eps_max = float(self.eps_grid[-1])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def log_F(self, R):
        """Return `log F(R) = log ε(R)` directly from the interpolant.

        For `R` outside the precomputed range, clamps to the boundary log ε.
        """
        scalar = np.isscalar(R)
        R_arr = np.atleast_1d(np.asarray(R, dtype=float))
        log_eps = self._interp_core(R_arr)
        log_eps = np.where(np.isnan(log_eps) & (R_arr <= self.R_min),
                           self._log_eps_lo, log_eps)
        log_eps = np.where(np.isnan(log_eps) & (R_arr >= self.R_max),
                           self._log_eps_hi, log_eps)
        return float(log_eps[0]) if scalar else log_eps

    def __call__(self, R):
        """Return `F(R) = ε(R)` (the converse error at rate R)."""
        log_eps = self.log_F(R)
        return float(np.exp(log_eps)) if np.isscalar(log_eps) else np.exp(log_eps)

    def converse_rate(self, epsilon: float) -> float:
        """Direct `ε → R` call, bypassing the interpolant.

        Uses whichever path (linear or log-domain) the grid was built with,
        so it is consistent with what :meth:`log_F` returns.
        """
        if self.log_domain:
            return self._converse.converse_rate_log(epsilon)
        return self._converse.converse_rate(epsilon)
