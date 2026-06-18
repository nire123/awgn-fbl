"""Tests for Gallager random coding achievability."""

import warnings

import numpy as np
import pytest

from awgn_fbl import GallagerAchievable
from awgn_fbl import NoncentralTConverse, ChiSquaredConverse


# Silence SciPy integration warnings that aren't relevant to these tests
pytestmark = pytest.mark.filterwarnings("ignore")


class TestGallagerBasic:
    def test_constructor(self):
        g = GallagerAchievable(n=200, snr_db=0.0)
        assert g.n == 200
        assert g.P == pytest.approx(1.0)
        assert g.A == pytest.approx(1.0)
        assert g.capacity == pytest.approx(0.5)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            GallagerAchievable(n=0, snr_db=0.0)

    def test_invalid_epsilon(self):
        g = GallagerAchievable(n=200, snr_db=0.0)
        with pytest.raises(ValueError):
            g.achievable_rate(0.0)
        with pytest.raises(ValueError):
            g.achievable_rate(1.0)

    def test_Rcr_known_value(self):
        """Rcr at SNR=0 dB (P=1) should be ~0.194 bits."""
        g = GallagerAchievable(n=200, snr_db=0.0)
        assert g.Rcr_bits == pytest.approx(0.194, abs=0.01)


class TestGallagerPolyanskiyReference:
    """Validate against Polyanskiy's explicit numerical reference."""

    def test_reference_point(self):
        """Polyanskiy: gallager_ach(3000, 1e-6, 1) = 1225 (log M in bits)."""
        g = GallagerAchievable(n=3000, snr_db=0.0)
        R = g.achievable_rate(1e-6)
        log_M = R * 3000
        # Expect to match to within ~1 bit
        assert log_M == pytest.approx(1225, abs=3)


class TestGallagerSanity:
    def test_achievable_below_converse(self):
        """Gallager rate must be <= converse rate."""
        n, snr_db, eps = 200, 0.0, 1e-3
        R_ach = GallagerAchievable(n=n, snr_db=snr_db).achievable_rate(eps)
        R_conv = NoncentralTConverse(n=n, snr_db=snr_db).converse_rate(eps)
        assert R_ach <= R_conv + 1e-6

    def test_achievable_below_capacity(self):
        for snr_db in [-2, 0, 3, 5]:
            g = GallagerAchievable(n=200, snr_db=snr_db)
            R = g.achievable_rate(1e-3)
            assert R < g.capacity

    def test_monotonic_in_n(self):
        """Rate should increase with n (at fixed SNR and eps)."""
        rates = [
            GallagerAchievable(n=n, snr_db=0.0).achievable_rate(1e-3)
            for n in [100, 200, 500, 1000]
        ]
        # Monotone non-decreasing
        diffs = np.diff(rates)
        assert np.all(diffs >= -1e-6)

    def test_monotonic_in_epsilon(self):
        """Rate should increase with epsilon (more errors tolerated)."""
        g = GallagerAchievable(n=200, snr_db=0.0)
        rates = [g.achievable_rate(eps) for eps in [1e-6, 1e-4, 1e-3, 1e-2]]
        diffs = np.diff(rates)
        assert np.all(diffs >= -1e-6)

    def test_error_at_rate_zero(self):
        g = GallagerAchievable(n=200, snr_db=0.0)
        assert g.achievable_error(0.0) == 0.0


class TestGallagerErrorDirection:
    def test_error_round_trip(self):
        """If R is the rate for epsilon, Pe(R) should be approximately epsilon."""
        g = GallagerAchievable(n=500, snr_db=0.0)
        eps_target = 1e-3
        R = g.achievable_rate(eps_target)
        pe = g.achievable_error(R)
        # Should be ≤ target (binary search returns rate satisfying Pe ≤ eps)
        assert pe <= eps_target * 1.01

    def test_error_monotone_in_rate(self):
        """Error probability should increase with rate."""
        g = GallagerAchievable(n=500, snr_db=0.0)
        pes = [g.achievable_error(R) for R in [0.1, 0.15, 0.2, 0.25, 0.3]]
        diffs = np.diff(pes)
        assert np.all(diffs >= -1e-10)
