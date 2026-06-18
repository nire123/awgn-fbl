"""
Shannon capacity and channel dispersion for the AWGN channel.

These are the two scalar quantities that any finite-blocklength analysis
falls back on in the infinite-blocklength / first-order asymptotics: the
Shannon capacity `C(SNR)` (in bits/channel use) and the channel dispersion
`V(SNR)` that appears in the second-order term of the normal approximation.
"""

from __future__ import annotations

import numpy as np


__all__ = ["shannon_capacity", "awgn_dispersion"]


def shannon_capacity(snr: float) -> float:
    """Shannon capacity `C = ½·log₂(1 + SNR)` of the real AWGN channel.

    Parameters
    ----------
    snr : float
        Linear SNR (power ratio, not dB).
    """
    return 0.5 * np.log2(1 + snr)


def awgn_dispersion(snr: float) -> float:
    """Channel dispersion `V` of the real AWGN channel, in (bits/use)².

    `V = SNR·(2 + SNR) / (2·(1 + SNR)²) · (log₂ e)²`.

    Parameters
    ----------
    snr : float
        Linear SNR (power ratio).
    """
    log2e_sq = np.log2(np.e) ** 2
    return snr * (2 + snr) / (2 * (1 + snr) ** 2) * log2e_sq
