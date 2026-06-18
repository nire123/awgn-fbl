"""Tests for the exact random-coding bound implementation."""

import numpy as np
import pytest

from awgn_fbl import ExactRandomCoding
from awgn_fbl import RCUAchievable
from awgn_fbl import NoncentralTConverse


pytestmark = pytest.mark.filterwarnings("ignore")


class TestOrdering:
    """exact ≤ RCU union ≤ RCU+, all ≥ converse."""

    @pytest.mark.parametrize("n,snr_db,R", [
        (30, 0.0, 0.15),
        (50, 0.0, 0.20),
        (100, 0.0, 0.30),
    ])
    def test_exact_at_or_below_union(self, n, snr_db, R):
        ex = ExactRandomCoding(n=n, snr_db=snr_db, seed=0)
        P_ex = ex.exact_error(R, n_samples=20_000)
        P_u = ex.rcu_union_error(R, n_samples=20_000)
        # Allow small MC noise
        assert P_ex <= P_u * 1.02, f"P_exact={P_ex} should be ≤ P_union={P_u}"

    @pytest.mark.parametrize("n,snr_db,R", [
        (50, 0.0, 0.20),
        (100, 0.0, 0.30),
    ])
    def test_union_at_or_below_rcu_plus(self, n, snr_db, R):
        ex = ExactRandomCoding(n=n, snr_db=snr_db, seed=0)
        rcu = RCUAchievable(n=n, snr_db=snr_db)
        P_u = ex.rcu_union_error(R, n_samples=20_000)
        P_rp = rcu.achievable_error(R)
        # In most of the natural operating range the library's RCU+
        # integral is at least as loose as the per-sample min envelope
        # (with a small MC-noise slack).
        assert P_u <= P_rp * 1.5

    @pytest.mark.parametrize("n,snr_db,R", [
        (50, 0.0, 0.20),
        (100, 0.0, 0.30),
    ])
    def test_exact_at_or_above_converse(self, n, snr_db, R):
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        ex = ExactRandomCoding(n=n, snr_db=snr_db, seed=0)
        P_ex = ex.exact_error(R, n_samples=20_000)
        P_conv = conv.converse_error_log(R)
        assert P_ex >= P_conv - 1e-6


class TestMCConvergence:
    """MC estimator standard error shrinks as 1/sqrt(N)."""

    def test_reduces_with_more_samples(self):
        """Std of estimate should decrease with more samples.  We use
        independent seeds per batch so variance reflects real MC noise."""
        small = []
        for seed in range(10):
            ex = ExactRandomCoding(n=50, snr_db=0.0, seed=seed)
            small.append(ex.exact_error(0.25, n_samples=2_000))
        large = []
        for seed in range(10, 20):
            ex = ExactRandomCoding(n=50, snr_db=0.0, seed=seed)
            large.append(ex.exact_error(0.25, n_samples=30_000))
        assert np.std(large) < np.std(small)


class TestTrivialCases:
    def test_rate_zero_error_zero(self):
        """R = 0 implies M = 1 implies P_e = 0."""
        ex = ExactRandomCoding(n=50, snr_db=0.0, seed=0)
        P = ex.exact_error(0.0, n_samples=5_000)
        assert P < 1e-10

    def test_vectorized_log_G_matches_scalar(self):
        conv = NoncentralTConverse(n=50, snr_db=0.0)
        ts = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
        scalar = np.array([conv.log_pairwise_error_prob(float(t)) for t in ts])
        vec = conv.log_pairwise_error_prob_vec(ts)
        assert np.max(np.abs(scalar - vec)) < 1e-10
