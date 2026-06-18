import numpy as np
import pytest

from awgn_fbl import (
    awgn_dispersion,
    normal_approx_rate,
    normal_approx_error,
)
from awgn_fbl import NoncentralTConverse


class TestDispersion:
    def test_zero_snr(self):
        """At SNR=0, V should be 0 (no channel → no dispersion)."""
        assert awgn_dispersion(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_positive(self):
        assert awgn_dispersion(1.0) > 0
        assert awgn_dispersion(10.0) > 0

    def test_approaches_limit(self):
        """V → (log₂ e)² / 2 as SNR → ∞."""
        V_large = awgn_dispersion(1e6)
        limit = np.log2(np.e) ** 2 / 2
        assert V_large == pytest.approx(limit, rel=0.01)


class TestNormalApproxRate:
    def test_below_capacity(self):
        snr_db = 0.0
        C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
        R = normal_approx_rate(200, 1e-3, snr_db)
        assert 0 < R < C

    def test_approaches_capacity_large_n(self):
        snr_db = 5.0
        C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
        R = normal_approx_rate(100000, 1e-3, snr_db)
        assert R == pytest.approx(C, abs=0.01)

    def test_higher_eps_gives_higher_rate(self):
        R_strict = normal_approx_rate(200, 1e-5, 0.0)
        R_relaxed = normal_approx_rate(200, 1e-1, 0.0)
        assert R_relaxed > R_strict

    def test_reasonable_value_n200(self):
        """At n=200, SNR=0dB, ε=1e-3, should be near the exact bounds."""
        R = normal_approx_rate(200, 1e-3, 0.0)
        assert 0.25 < R < 0.40

    def test_between_achievable_and_converse(self):
        """Normal approx should be near (but not necessarily between) bounds."""
        R_na = normal_approx_rate(200, 1e-3, 0.0)
        R_conv = NoncentralTConverse(n=200, snr_db=0.0).converse_rate(1e-3)
        # Normal approx should be below converse
        assert R_na < R_conv


class TestNormalApproxError:
    def test_round_trip(self):
        R = normal_approx_rate(200, 1e-3, 5.0)
        eps_back = normal_approx_error(200, R, 5.0)
        assert eps_back == pytest.approx(1e-3, rel=0.01)

    def test_higher_rate_gives_higher_error(self):
        eps_lo = normal_approx_error(200, 0.3, 0.0)
        eps_hi = normal_approx_error(200, 0.45, 0.0)
        assert eps_hi > eps_lo
