import numpy as np
import pytest

from awgn_fbl import RCUAchievable
from awgn_fbl import NoncentralTConverse


@pytest.fixture(scope="module")
def rcu():
    return RCUAchievable(n=200, snr_db=0.0, method="nct")


# ---------------------------------------------------------------------------
# Known values from documentation
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_achievable_rate_n200_snr0(self, rcu):
        """Doc says R ≈ 0.3346 at n=200, SNR=0dB, ε=1e-3."""
        R = rcu.achievable_rate(1e-3)
        assert 0.30 < R < 0.36

    def test_achievable_error_at_known_rate(self, rcu):
        """Error at a rate near capacity should be small but nonzero."""
        eps = rcu.achievable_error(0.30)
        assert 0 < eps < 1


# ---------------------------------------------------------------------------
# Achievable must be below converse
# ---------------------------------------------------------------------------

class TestBoundOrdering:
    @pytest.mark.parametrize("eps", [1e-2, 1e-3])
    def test_achievable_below_converse(self, rcu, eps):
        R_ach = rcu.achievable_rate(eps)
        R_conv = NoncentralTConverse(n=200, snr_db=0.0).converse_rate(eps)
        assert R_ach < R_conv

    def test_gap_is_small(self, rcu):
        """Gap should be < 0.05 bits at ε=1e-3 (doc says 0.0092)."""
        R_ach = rcu.achievable_rate(1e-3)
        R_conv = NoncentralTConverse(n=200, snr_db=0.0).converse_rate(1e-3)
        assert R_conv - R_ach < 0.05


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_error_increases_with_rate(self, rcu):
        """Higher rate → higher error probability."""
        eps_lo = rcu.achievable_error(0.25)
        eps_hi = rcu.achievable_error(0.32)
        assert eps_hi > eps_lo

    def test_rate_increases_with_epsilon(self, rcu):
        """More relaxed error → higher achievable rate."""
        R_strict = rcu.achievable_rate(1e-4)
        R_relaxed = rcu.achievable_rate(1e-2)
        assert R_relaxed > R_strict


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_error_then_rate(self, rcu):
        """achievable_error(R) then achievable_rate(ε) should recover R."""
        R_in = 0.30
        eps = rcu.achievable_error(R_in)
        if not np.isnan(eps) and 0 < eps < 1:
            R_back = rcu.achievable_rate(eps)
            assert R_back == pytest.approx(R_in, abs=0.005)
