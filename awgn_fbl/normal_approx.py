"""
Second-order normal approximation for the finite-blocklength coding rate
(Polyanskiy–Poor–Verdú 2010):

    R*(n, ε) ≈ C − √(V/n)·Q⁻¹(ε) + log₂(n) / (2n).

This is not a rigorous bound; it is the standard benchmark against which
rigorous converse and achievability bounds are compared.  Accurate to
`O(log n / n)`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .capacity import shannon_capacity, awgn_dispersion


__all__ = ["normal_approx_rate", "normal_approx_error", "awgn_dispersion"]


def normal_approx_rate(n: int, epsilon: float, snr_db: float) -> float:
    """Normal approximation of the maximum achievable rate.

    ``R*(n, ε) ≈ C − √(V/n)·Q⁻¹(ε) + log₂(n)/(2n)``.

    Parameters
    ----------
    n : int
        Blocklength.
    epsilon : float
        Target error probability, in (0, 1).
    snr_db : float
        SNR in dB.
    """
    snr = 10 ** (snr_db / 10)
    C = shannon_capacity(snr)
    V = awgn_dispersion(snr)
    Qinv = stats.norm.ppf(1 - epsilon)
    return C - np.sqrt(V / n) * Qinv + np.log2(n) / (2 * n)


def normal_approx_error(n: int, rate_bits: float, snr_db: float) -> float:
    """Inverse normal approximation: error probability at rate R.

    ``ε ≈ Q((C − R + log₂(n)/(2n)) / √(V/n))``.

    Parameters
    ----------
    n : int
        Blocklength.
    rate_bits : float
        Rate in bits/channel use.
    snr_db : float
        SNR in dB.
    """
    snr = 10 ** (snr_db / 10)
    C = shannon_capacity(snr)
    V = awgn_dispersion(snr)
    arg = (C - rate_bits + np.log2(n) / (2 * n)) / np.sqrt(V / n)
    return stats.norm.sf(arg)
