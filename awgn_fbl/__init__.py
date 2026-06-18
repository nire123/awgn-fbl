"""
awgn_fbl — Finite-blocklength bounds for the real-valued AWGN channel.

The package groups:

* Mathematical primitives :mod:`awgn_fbl._pairwise`, :mod:`awgn_fbl.capacity`:
  Lemma 1 (pairwise error probability) in linear and log domain,
  log-domain non-central t CDF/SF, Shannon capacity, channel dispersion.

* Converse bounds :mod:`awgn_fbl.converse`:
  :class:`NoncentralTConverse` (our NCT method with both a simple linear
  path and a log-domain path), :class:`ChiSquaredConverse` (Polyanskiy's
  relaxation), :class:`SolidAngleConverse` (Shannon 1959).

* Achievability bounds :mod:`awgn_fbl.achievable`:
  :class:`RCUAchievable` (our flagship, with linear and log-safe paths),
  :class:`KappaBetaAchievable` (simple) and :class:`KappaBetaAchievablePPV`
  (faithful PPV port), :class:`GallagerAchievable`,
  :class:`ExactRandomCoding` (Monte-Carlo).

* Normal approximation :mod:`awgn_fbl.normal_approx`.

* :class:`awgn_fbl.fast_f.FastFREvaluator` — precomputed converse curve
  F(R) used by :class:`RCUAchievable`.

* :func:`awgn_fbl.plot.plot` and its convenience wrappers.

A "simple reference / efficient production" split is used consistently:
where a bound has both a direct linear implementation and a more involved
log-domain implementation, both are kept — the simple version serves as
a test oracle, the efficient version as the recommended default.
"""

__version__ = "0.1.0"

from .capacity import awgn_dispersion, shannon_capacity
from .converse import (
    AWGNConverseBase,
    ChiSquaredConverse,
    NoncentralTConverse,
    SolidAngleConverse,
    awgn_converse_error,
    awgn_converse_rate,
)
from .achievable import (
    ExactRandomCoding,
    GallagerAchievable,
    KappaBetaAchievable,
    KappaBetaAchievablePPV,
    RCUAchievable,
)
from .normal_approx import normal_approx_error, normal_approx_rate
from .fast_f import FastFREvaluator
from ._pairwise import (
    log_nct_cdf,
    log_nct_sf,
    log_pairwise_error_prob,
    log_pairwise_error_prob_vec,
    pairwise_error_prob,
)
from .plot import (
    ALL_CURVES,
    CURVE_STYLES,
    ERROR_CURVES,
    plot,
    plot_error_vs_n,
    plot_error_vs_snr,
    plot_rate_vs_epsilon,
    plot_rate_vs_n,
    plot_rate_vs_snr,
    plot_snr_vs_n,
)

__all__ = [
    "__version__",
    # capacity
    "shannon_capacity", "awgn_dispersion",
    # converse
    "AWGNConverseBase", "NoncentralTConverse", "ChiSquaredConverse",
    "SolidAngleConverse", "awgn_converse_rate", "awgn_converse_error",
    # achievable
    "RCUAchievable", "KappaBetaAchievable", "KappaBetaAchievablePPV",
    "GallagerAchievable", "ExactRandomCoding",
    # normal approximation
    "normal_approx_rate", "normal_approx_error",
    # helpers
    "FastFREvaluator",
    # math primitives
    "pairwise_error_prob", "log_pairwise_error_prob",
    "log_pairwise_error_prob_vec", "log_nct_cdf", "log_nct_sf",
    # plotting
    "plot", "plot_rate_vs_snr", "plot_rate_vs_epsilon", "plot_rate_vs_n",
    "plot_error_vs_snr", "plot_error_vs_n", "plot_snr_vs_n",
    "CURVE_STYLES", "ALL_CURVES", "ERROR_CURVES",
]
